//! Captures borrow only engine-owned weights, state and planes. Uploads and tap
//! scatters remain ordered eager work outside each complete layer segment.
use super::*;
use crate::shared::decode_graph::{GraphBank, device::{self, Basis, Cycle, Preference, Site}};
use cuteafd_loader::serving_capacity::qwen35_graphs::{self, Geometry};
use std::{cell::RefCell, ffi::c_void};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Key {
    segment: usize,
    rows: usize,
    table_stride: usize,
    spec: bool,
}

pub(super) struct Executable<'a>(*mut c_void, Device<'a>);
impl Drop for Executable<'_> {
    fn drop(&mut self) {
        // SAFETY: the owning engine drains before this bank or any backing drops.
        let _ = self.1.run(|| unsafe { self.1.library.cuda_graph_exec_destroy(self.0) });
    }
}

pub(super) struct Graphs<'a> {
    bank: RefCell<GraphBank<Key, Executable<'a>>>,
    variant: String,
    enabled: bool,
    capture_ready: std::cell::Cell<bool>,
    warming: std::cell::Cell<bool>,
}

impl<'a> Graphs<'a> {
    pub(super) fn new(device: Device<'a>, variant: String) -> Result<Self> {
        let reserve = qwen35_graphs::reserve(&variant)?;
        let mut bank = GraphBank::new(None);
        bank.attach_device(device.library, device.id, reserve.bytes, Site {
            name: "qwen3_5.segment", cycle: Cycle::Other, preference: Preference::Adaptive,
            bootstrap: qwen35_graphs::BOOTSTRAP_BYTES, basis: Basis::Bytes, spans_devices: false,
        })?;
        device::set_reserve_policy(device.id, reserve)?;
        tracing::warn!(%variant, reserve = reserve.bytes,
            "Dense Qwen graph cache uses uncalibrated byte basis");
        if qwen35_graphs::layout(&variant).is_none() {
            tracing::warn!(%variant, "Dense Qwen layout has no trace: bootstrap is unpinned");
        }
        Ok(Self { bank: RefCell::new(bank), variant,
            enabled: std::env::var("CUTEAFD_QWEN35_GRAPHS").ok().as_deref() == Some("1"),
            capture_ready: std::cell::Cell::new(false), warming: std::cell::Cell::new(false) })
    }

    pub(super) fn cycle(&self, speculative: bool) {
        self.bank.borrow_mut().cache_cycle(if self.warming.get() { Cycle::Other }
            else if speculative { Cycle::Verify } else { Cycle::Decode });
    }

    pub(super) fn prefill(&self) { self.bank.borrow_mut().cache_cycle(Cycle::Prefill); }
}

impl DenseEngine<'_> {
    pub(super) fn graph_layer(&self, index: usize, step: &Step) -> Result<()> {
        let segment = || Blocks { programs: self.programs, admitted: &self.admitted,
            cfg: &self.cfg, homes: &self.homes, bank: &self.bank, stream: self.stream.raw }
            .run(index, &self.weights.layers[index], self.planes,
                self.kv[index].as_ref().map(|allocation| allocation.buffer), step);
        if !self.graphs.enabled || !self.graphs.capture_ready.get() { return segment(); }
        let key = Key { segment: index, rows: step.rows, table_stride: step.table_stride, spec: step.speculative };
        let library = self.stream.device.library;
        if let Some(raw) = self.graphs.bank.borrow_mut().cached_launch(&key).map(|exec| exec.0) {
            // SAFETY: all captured pointers retain the same engine-owned backing.
            return unsafe { library.cuda_graph_launch(raw, self.stream.raw) };
        }
        let pinned = qwen35_graphs::layout(&self.graphs.variant).is_some_and(|layout|
            layout.traced && layout.entries.iter().any(|entry|
                entry.pin && entry.rows == key.rows && entry.table_stride == key.table_stride && entry.spec == key.spec));
        let Some(ticket) = self.graphs.bank.borrow_mut().cache_ticket(&key, pinned, None) else {
            return segment();
        };
        // SAFETY: blocks only enqueue launches over persistent engine storage.
        if let Err(error) = unsafe { library.cuda_graph_begin_capture(self.stream.raw) } {
            drop(ticket);
            tracing::warn!(%error, "Dense Qwen capture unavailable; eager fallback");
            return segment();
        }
        let captured = segment();
        // SAFETY: ends capture on the same stream even when enqueue failed.
        let ended = unsafe { library.cuda_graph_end_capture_topology(self.stream.raw) };
        if let Err(error) = captured {
            if let Ok((raw, ..)) = ended {
                // SAFETY: this executable has never been launched.
                unsafe { let _ = library.cuda_graph_exec_destroy(raw); }
            }
            return Err(error);
        }
        let (raw, nodes, width, spans, inspection) = match ended {
            Ok(info) => info,
            Err(error) => {
                drop(ticket);
                tracing::warn!(%error, "Dense Qwen instantiate unavailable; eager fallback");
                return segment();
            }
        };
        let exec = Executable(raw, self.stream.device);
        // Unknown topology or a singleton is not useful enough to retain.
        if inspection.is_some() || nodes < 2 {
            drop(exec);
            drop(ticket);
            return segment();
        }
        self.graphs.bank.borrow_mut().cache_commit(key, exec, ticket, nodes as u64, width, spans);
        // SAFETY: successful insertion retains the executable until a drained boundary.
        unsafe { library.cuda_graph_launch(raw, self.stream.raw) }
    }

    pub(super) fn graph_completed(&self) -> Result<()> {
        if !self.graphs.bank.borrow().cache_needs_boundary() {
            self.graphs.bank.borrow_mut().cache_tick();
            return Ok(());
        }
        self.stream.drain()?;
        // SAFETY: the single target stream is drained; the bank owns no captured
        // backing, and persistent weights/state/planes outlive every executable.
        unsafe { device::observe_completed(self.stream.device.library, |visit|
            visit(self.stream.device.id, "qwen3_5.segment", false, &mut *self.graphs.bank.borrow_mut())) }
    }

    pub(crate) fn warm_decode_graphs(&mut self) -> Result<()> {
        if !self.graphs.enabled { return Ok(()); }
        let entries = qwen35_graphs::layout(&self.graphs.variant).map_or_else(Vec::new, |layout| layout.entries.clone());
        self.graphs.warming.set(true);
        let result = (|| {
            let request = [crate::shared::speculation::target::VerifyRow { slot: 0, position: 0, token: Some(0) }];
            let plan = rows::Rows::new(&crate::shared::speculation::target::VerifyRequest {
                rows: &request, tap_layers: &[] }, self.cfg.attention.len(), self.cfg.vocab,
                &[(0, self.slots[0].clone())], self.cfg.max_context, self.pool_tokens)?;
            self.verify_rows(plan, 1, &[], true)?;
            self.verification.pending = None;
            self.reset_slot(0)?;
            device::observe_device(self.stream.device.library, self.stream.device.id)?;
            self.graphs.capture_ready.set(true);
            for Geometry { rows, table_stride, spec, .. } in entries {
                if table_stride * 64 > self.cfg.max_context || rows > 64 { continue; }
                let position = (table_stride - 1) * 64;
                if position + rows > self.cfg.max_context || position + rows > self.pool_tokens { continue; }
                // Prewarm through real device lengths; restore all recurrence
                // afterwards so startup leaves exactly the fresh-engine state.
                self.slots[0].position = position;
                let request = (0..rows).map(|offset| crate::shared::speculation::target::VerifyRow {
                    slot: 0, position: position + offset, token: Some(0) }).collect::<Vec<_>>();
                let plan = rows::Rows::new(&crate::shared::speculation::target::VerifyRequest {
                    rows: &request, tap_layers: &[] }, self.cfg.attention.len(), self.cfg.vocab,
                    &[(0, self.slots[0].clone())], self.cfg.max_context, self.pool_tokens)?;
                self.verify_rows(plan, rows, &[], spec)?;
                self.verification.pending = None;
                self.reset_slot(0)?;
            }
            Ok(())
        })();
        self.graphs.warming.set(false);
        result
    }
}
