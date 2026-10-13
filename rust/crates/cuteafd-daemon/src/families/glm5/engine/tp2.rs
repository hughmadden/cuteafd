//! GLM's admitted routed prefix uses shared paired backends and the existing FFN slot.
use super::*;

impl<'a> GlmEngine<'a> {
    pub(crate) fn set_tp2_exchange(&mut self, dtype: ExchangeDtype) {
        self.tp2_exchange = dtype;
    }

    pub(crate) fn install_tp2(&mut self, ranks: [Box<dyn RtxExpertLayer + 'a>; 2]) -> Result<()> {
        let peer = self.peer()?;
        let layers = ranks[0].layers();
        ensure!(
            layers.start == self.cfg.first_moe_layer
                && layers.end <= self.weights.layers.len()
                && self.weights.layers[layers.clone()].iter().all(|l| !l.dense),
            "invalid GLM TP2 routed prefix"
        );
        ensure!(
            self.tp2.is_none()
                && self.decode_workspace.borrow().is_none()
                && self.lane_workspaces.borrow().is_empty()
                && peer.decode_workspace.borrow().is_none()
                && peer.lane_workspaces.borrow().is_empty(),
            "install TP2 before graph/workspace allocation"
        );
        let devices = [
            RankDevice {
                device: self.device,
                stream: self.stream,
            },
            RankDevice {
                device: peer.device,
                stream: peer.stream,
            },
        ];
        for (rank, producer) in devices.into_iter().enumerate() {
            self.exchange()?
                .require_cooperative_producer_stream(rank, producer)?;
        }
        self.tp2 = Some(Pair::new(
            self.library,
            devices,
            ranks,
            self.cfg.hidden,
            self.cfg.topk,
            self.prefill_rows.max(DECODE_ROWS),
            self.prefill_lanes,
            self.tp2_exchange,
        )?);
        cuteafd_bench::context::set_resolved(
            "rtx-tp2-layers",
            &format!("{}..{}", layers.start, layers.end),
        );
        Ok(())
    }

    pub(crate) fn has_tp2(&self) -> bool {
        self.tp2.is_some()
    }
    pub(super) fn tp2_layer(&self, layer: usize) -> bool {
        self.tp2
            .as_ref()
            .is_some_and(|pair| pair.layers.contains(&layer))
    }

    fn abort_tp2(&self) -> Result<()> {
        if self.tp2.is_none() {
            return Ok(());
        }
        // A backend failure may occur with rank1 current; always publish on the home device.
        crate::shared::memory::device::Device {
            library: self.library,
            id: self.device,
        }
        .run(|| {
            for exchange in [
                self.exchange()?,
                &self.tp2.as_ref().context("TP2 pair")?.routes,
            ] {
                match exchange.terminal_state() {
                    crate::shared::peer_split::TerminalState::Active => exchange.publish_abort()?,
                    crate::shared::peer_split::TerminalState::Aborting
                    | crate::shared::peer_split::TerminalState::Drained => {}
                    _ => anyhow::bail!("GLM TP2 abort owners retained"),
                }
            }
            Ok(())
        })
    }

    pub(crate) fn retire_tp2(&self) -> Result<()> {
        let Some(pair) = &self.tp2 else {
            return Ok(());
        };
        self.abort_tp2()?;
        self.exchange()?.drain_compute()?;
        pair.routes.drain_compute()?;
        self.on(0, || {
            self.graphs.borrow_mut().clear();
            Ok(())
        })?;
        self.on(1, || {
            self.peer()?.graphs.borrow_mut().clear();
            Ok(())
        })?;
        self.exchange()?.finish_terminal(true);
        pair.routes.finish_terminal(true);
        Ok(())
    }

    pub(crate) fn prime_tp2(&self) -> Result<()> {
        let Some(pair) = &self.tp2 else {
            return Ok(());
        };
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
        }
        while self.lane_workspaces.borrow().len() < self.prefill_lanes {
            let w = self.workspace(self.prefill_rows, false)?;
            self.lane_workspaces.borrow_mut().push(w);
        }
        drop(self.peer_workspaces(true, 1)?);
        drop(self.peer_workspaces(false, self.prefill_lanes)?);
        for rank in 0..2 {
            let owned = if rank == 0 {
                self.lane_workspaces.borrow()
            } else {
                self.peer()?.lane_workspaces.borrow()
            };
            let decode = if rank == 0 {
                self.decode_workspace.borrow()
            } else {
                self.peer()?.decode_workspace.borrow()
            };
            let w = if self.prefill_rows >= DECODE_ROWS {
                &owned[0]
            } else {
                decode.as_ref().context("decode workspace")?
            };
            self.on(rank, || {
                // SAFETY: startup buffers have no outstanding consumers or peer waits.
                unsafe {
                    self.library.cuda_zero_bytes_async(
                        w.x.buffer,
                        w.rows * self.cfg.hidden * 2,
                        self.stream_of(rank),
                    )?;
                }
                let layer = if rank == 0 {
                    &self.weights.layers[pair.layers.start]
                } else {
                    &self.peer()?.layers[pair.layers.start]
                };
                self.moe_front_on(rank, w, layer, w.rows)?;
                let cap = if self.prefill_rows >= DECODE_ROWS {
                    "m4096"
                } else {
                    "m64"
                };
                self.ffn_on(
                    rank,
                    w,
                    layer,
                    self.cfg.moe_intermediate,
                    cap,
                    w.shared.buffer.ptr,
                    Scalar::I32(w.rows as i32),
                )?;
                let input = if pair.partial() == PartialDtype::F32 {
                    ExpertInput::Fp8K32(w.wire.buffer.ptr)
                } else {
                    ExpertInput::Bf16(w.x.buffer.ptr)
                };
                // SAFETY: maximum-capacity initialized startup input, routes, shared and disjoint payload.
                unsafe {
                    pair.prime(
                        rank,
                        input,
                        Routes {
                            ids: w.route_ids.buffer.ptr,
                            weights: w.route_weights.buffer.ptr,
                        },
                        w.rows,
                        w.shared.buffer.ptr.cast(),
                        w.payload.as_ref().context("TP2 payload")?.buffer.ptr,
                    )
                }
            })?;
        }
        Ok(())
    }

    /// Local expert outputs are consumed before another lane can reuse backend scratch.
    pub(super) fn tp2_pair(
        &self,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        w1: &Workspace<'_>,
        rows: usize,
        cap: &str,
    ) -> Result<()> {
        let pair = self.tp2.as_ref().context("TP2 experts")?;
        self.ffn(
            w,
            &self.weights.layers[layer],
            self.cfg.moe_intermediate,
            cap,
            w.shared.buffer.ptr,
            Scalar::I32(rows as i32),
        )?;
        self.moe_front_on(1, w1, &self.peer()?.layers[layer], rows)?;
        self.tp2_rank(pair, 0, layer, lane, w, rows)?;
        self.tp2_rank(pair, 1, layer, lane, w1, rows)?;
        // Rank1 has no final norm/head, but must consume every publication
        // so the next request cannot observe this step's stale FFN sequence.
        if layer + 1 == self.weights.layers.len() {
            self.exchange()?.wait(1, slot(layer, true, lane))?;
        }
        Ok(())
    }

    fn tp2_rank(
        &self,
        pair: &Pair<'_>,
        rank: usize,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        rows: usize,
    ) -> Result<()> {
        let input = if pair.partial() == PartialDtype::F32 {
            ExpertInput::Fp8K32(w.wire.buffer.ptr)
        } else {
            ExpertInput::Bf16(w.x.buffer.ptr)
        };
        let out = w.payload.as_ref().context("TP2 payload")?.buffer.ptr;
        // SAFETY: each workspace is rank-local; payload consumes backend output in stream order.
        unsafe {
            pair.enqueue(
                rank,
                layer,
                lane,
                rows,
                input,
                Routes {
                    ids: w.route_ids.buffer.ptr,
                    weights: w.route_weights.buffer.ptr,
                },
                w.router_logits.buffer,
                w.shared.buffer.ptr.cast(),
                out,
                &mut || {
                    crate::shared::decode_graph::fatal_drain(
                        self.abort_tp2(),
                        "GLM TP2 enqueue abort",
                    )
                },
            )?;
        }
        self.exchange()?.push(
            rank,
            slot(layer, true, lane),
            out,
            pair.combine.payload_bytes(rows, self.cfg.hidden),
        )
    }

    pub(super) fn tp2_norm(
        &self,
        rank: usize,
        w: &Workspace<'_>,
        weight: *mut c_void,
        layer: usize,
        lane: usize,
        rows: Scalar,
    ) -> Result<()> {
        let Scalar::I32(rows_count) = rows else {
            anyhow::bail!("GLM TP2 norm rows must be i32");
        };
        let pair = self.tp2.as_ref().context("TP2 pair")?;
        let local = w.payload.as_ref().context("TP2 payload")?.buffer.ptr;
        let received = self.recv(rank, slot(layer, true, lane))?;
        let (rank0, rank1) = if rank == 0 {
            (local, received)
        } else {
            (received, local)
        };
        self.on(rank, || {
            // SAFETY: FFN wait precedes the rank-ordered sum into disjoint delta storage.
            unsafe {
                pair.combine.sum(
                    rank0,
                    rank1,
                    w.delta.buffer.ptr.cast(),
                    rows_count as usize,
                    self.cfg.hidden,
                    self.stream_of(rank),
                )
            }
        })?;
        self.norm_full(
            rank,
            w,
            weight,
            1,
            rows,
            w.delta.buffer.ptr,
            w.delta.buffer.ptr,
        )
    }

    pub(crate) fn check_tp2_routes(
        &self,
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
    ) -> Result<()> {
        if self.tp2.is_none() {
            return Ok(());
        }
        let rows = 512.min(self.max_context.saturating_sub(1));
        ensure!(rows > 0, "TP2 route diagnostics need two context rows");
        let mut allocator = PageAllocator::new(self);
        let mut placement = allocator.admit(rows + 1)?;
        let tokens: Vec<_> = (0..rows)
            .map(|i| (i % self.cfg.vocab_size) as u32)
            .collect();
        self.force_route_check.set(true);
        let mut experts = experts;
        let result = (|| {
            self.prefill_device(
                &mut placement,
                &tokens,
                experts.as_mut().map(|(l, r)| (&mut **l, *r)),
            )?;
            self.verify_device(
                &mut [(&mut placement, 1)],
                &[1],
                experts.as_mut().map(|(l, r)| (&mut **l, *r)),
            )?;
            Ok(())
        })();
        self.force_route_check.set(false);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{check, Schedule};

    fn schedule(
        lanes: usize,
        local: std::ops::Range<usize>,
        broadcast: bool,
        repeats: usize,
    ) -> Schedule {
        let mut s = Schedule::default();
        for _ in 0..repeats {
            for _ in 0..lanes {
                s.push(0, "main", DIRECT, "embedding");
            }
            let mut pending = std::collections::VecDeque::new();
            let post = |s: &mut Schedule, layer, lane, remote| {
                let ffn = slot(layer, true, lane);
                if remote && layer < 7 {
                    s.push(0, "main", ffn, "Spark sum");
                }
                if layer < 7 || local.contains(&layer) {
                    s.wait(1, "main", ffn, "rank1 FFN");
                }
                s.wait(0, "main", ffn, "rank0 FFN");
            };
            for layer in 0..8 {
                for lane in 0..lanes {
                    while pending.iter().any(|&(_, k)| k == lane) {
                        let (l, k) = pending.pop_front().unwrap();
                        post(&mut s, l, k, true);
                    }
                    if layer == 0 {
                        s.wait(1, "main", DIRECT, "embedding received");
                    }
                    for rank in 0..2 {
                        s.push(rank, "main", slot(layer, false, lane), "attention");
                        s.wait(rank, "main", slot(layer, false, lane), "attention received");
                    }
                    if local.contains(&layer) {
                        if broadcast {
                            let at = 2 * lane + layer % 2;
                            s.push(0, "routes", at, "canonical routes");
                            s.wait(0, "routes", at, "canonical consumed");
                            s.wait(1, "routes", at, "canonical received");
                            s.push(1, "routes", at, "canonical backend finished");
                        }
                        for rank in 0..2 {
                            s.push(rank, "main", slot(layer, true, lane), "paired payload");
                        }
                        post(&mut s, layer, lane, false);
                    } else {
                        s.push(1, "main", slot(layer, true, lane), "shared or dense");
                        if layer < 3 {
                            s.push(0, "main", slot(layer, true, lane), "dense");
                            post(&mut s, layer, lane, false);
                        } else {
                            pending.push_back((layer, lane));
                        }
                    }
                }
            }
            while let Some((l, k)) = pending.pop_front() {
                post(&mut s, l, k, true);
            }
        }
        s
    }

    #[test]
    fn local_prefix_and_spark_suffix_drain_all_admitted_lanes_and_reused_flags() {
        for lanes in 1..=4 {
            for end in 4..=8 {
                for broadcast in [false, true] {
                    assert_eq!(
                        check(&schedule(lanes, 3..end, broadcast, 3)),
                        Ok(()),
                        "lanes={lanes} local=3..{end} broadcast={broadcast}"
                    );
                }
            }
        }
    }
}
