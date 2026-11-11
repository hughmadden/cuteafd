//! Weight operations are selected once for loading and CPU header replay.
//! Requests precede execution; the real uploader must return exactly these owners.
use super::*;
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::formats::trellis_core::CoreWeightLayout as SharedCoreLayout;
use cuteafd_loader::placement::planner::{Allocation, Lifetime, Location};

pub(super) enum Operation {
    Tensor(String),
    Rows(Vec<String>),
    Projection {
        names: Vec<String>,
        axis: Axis,
        bf16: bool,
        kmajor: bool,
        replicated: bool,
    },
    KvB {
        layer: usize,
        bf16: bool,
    },
    HeadVector(String),
    Core(SharedCoreLayout),
    NativeAttention(cuteafd_loader::formats::native_attention::NativeLayout),
    TensorScales(Vec<String>),
}

pub(super) struct Route {
    pub ranks: Vec<usize>,
    pub key: &'static str,
    pub operation: Operation,
    pub requests: Vec<Allocation>,
}

struct Routes<'a> {
    catalog: &'a OfficialV41Catalog,
    cfg: &'a GlmDsaConfig,
    layer: usize,
    ranks: usize,
    attention: Vec<usize>,
    routes: Vec<Route>,
}

fn extent(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |n, &factor| {
        n.checked_mul(factor)
            .context("weight request extent overflow")
    })
}

impl Routes<'_> {
    fn weight(&self, op: Op) -> Result<String> {
        self.cfg
            .weight(self.layer, op)
            .context("missing variant weight operand")
    }
    fn meta(&self, name: &str) -> Result<&cuteafd_loader::SafetensorsTensorMetadata> {
        Ok(&self.catalog.tensor(name)?.metadata)
    }
    fn native_bf16(&self, names: &[String], enabled: bool) -> Result<bool> {
        names.iter().try_fold(enabled, |yes, name| {
            Ok(yes && self.meta(name)?.dtype == DType::Bf16)
        })
    }
    fn push(&mut self, attention: bool, key: &'static str, operation: Operation) -> Result<()> {
        let ranks = if attention {
            self.attention.clone()
        } else {
            (0..self.ranks).collect()
        };
        let parts = ranks.len();
        let mut sizes = Vec::<(String, usize)>::new();
        match &operation {
            Operation::Tensor(name) => {
                let meta = self.meta(name)?;
                let bytes = if meta.dtype == DType::F8E4M3 && meta.shape.len() == 2 {
                    extent(&[meta.shape[0], meta.shape[1], 2])?
                } else {
                    meta.byte_length as usize
                };
                sizes.push((key.into(), bytes.max(256)));
            }
            Operation::Rows(names) => {
                ensure!(!names.is_empty(), "empty BF16 row source");
                let mut bytes = 0usize;
                let mut cols = None;
                for name in names {
                    let meta = self.meta(name)?;
                    ensure!(
                        meta.shape.len() == 2
                            && cols.is_none_or(|k| k == meta.shape[1])
                            && matches!(meta.dtype, DType::Bf16 | DType::F8E4M3),
                        "BF16 row source {name}"
                    );
                    cols = Some(meta.shape[1]);
                    bytes = bytes
                        .checked_add(
                            meta.shape[0]
                                .checked_mul(meta.shape[1])
                                .and_then(|n| n.checked_mul(2))
                                .context("row extent overflow")?,
                        )
                        .context("row sum overflow")?;
                }
                sizes.push((key.into(), bytes));
            }
            Operation::Projection {
                names,
                axis,
                bf16,
                kmajor,
                replicated,
            } => {
                ensure!(!names.is_empty(), "empty projection source");
                let parts = if *replicated { 1 } else { parts };
                let mut values = 0usize;
                let mut grids = 0usize;
                let mut rows = 0usize;
                let mut cols = None;
                let mut native = true;
                for (index, name) in names.iter().enumerate() {
                    let meta = self.meta(name)?;
                    ensure!(
                        meta.shape.len() == 2
                            && meta.shape.iter().all(|&n| n > 0)
                            && cols.is_none_or(|k| k == meta.shape[1])
                            && matches!(meta.dtype, DType::Bf16 | DType::F8E4M3),
                        "projection geometry {name}"
                    );
                    let (n, k) = (meta.shape[0], meta.shape[1]);
                    ensure!(
                        match axis {
                            Axis::Rows => n % parts == 0,
                            Axis::Cols => k % parts == 0,
                        },
                        "projection split {name}"
                    );
                    cols = Some(k);
                    rows = rows.checked_add(n).context("projection row overflow")?;
                    values = values
                        .checked_add(n.checked_mul(k).context("projection extent overflow")?)
                        .context("projection sum overflow")?;
                    grids = grids
                        .checked_add(
                            n.div_ceil(128)
                                .checked_mul(k.div_ceil(128))
                                .and_then(|n| n.checked_mul(4))
                                .context("projection scale overflow")?,
                        )
                        .context("projection scales overflow")?;
                    native &= landing::native_fp8_block(self.catalog, name)?;
                    if *bf16 {
                        ensure!(meta.dtype == DType::Bf16, "BF16 projection {name}");
                    } else {
                        ensure!(
                            index + 1 == names.len() || n % 128 == 0,
                            "projection concatenation cuts a block {name}"
                        );
                        if parts > 1 {
                            let (nb, kb) = match axis {
                                Axis::Rows => (128 * parts, 128),
                                Axis::Cols => (128, 128 * parts),
                            };
                            ensure!(
                                n % nb == 0 && k % kb == 0,
                                "projection split cuts a block {name}"
                            );
                        }
                        if meta.dtype == DType::F8E4M3
                            && !landing::native_fp8_block(self.catalog, name)?
                        {
                            let scalar = self.meta(&format!("{name}_scale"))?;
                            ensure!(
                                scalar.dtype == DType::F32 && scalar.byte_length == 4,
                                "FP8 scalar {name}"
                            );
                        }
                    }
                }
                if *bf16 {
                    sizes.push((key.into(), extent(&[values / parts, 2])?.max(256)));
                } else {
                    sizes.push((
                        fp8_operand_names(key).0.into(),
                        if parts == 1 || native {
                            (values / parts).max(256)
                        } else {
                            values / parts
                        },
                    ));
                    sizes.push((
                        fp8_operand_names(key).1.into(),
                        if parts == 1 || native || *axis == Axis::Cols {
                            (grids / parts).max(256)
                        } else {
                            grids / parts
                        },
                    ));
                    if *kmajor {
                        sizes.push((
                            "w_qkv_a_kscale".into(),
                            extent(&[rows, cols.context("empty projection")?.div_ceil(128), 4])?
                                .max(256),
                        ));
                    }
                }
            }
            Operation::KvB { layer, bf16 } => {
                let name = self.cfg.weight(*layer, Op::KvB).context("KV-B operand")?;
                let meta = self.meta(&name)?;
                let (h, d, v, c) = (
                    self.cfg.heads,
                    self.cfg.nope_dim(*layer),
                    self.cfg.v_head_dim,
                    self.cfg.kv_lora_rank,
                );
                let width = d.checked_add(v).context("KV-B width overflow")?;
                ensure!(
                    h % parts == 0 && meta.shape == [extent(&[h, width])?, c],
                    "KV-B geometry {name}"
                );
                for (key, values) in [
                    ("w_uk", extent(&[h / parts, c, d])?),
                    ("w_uv", extent(&[h / parts, v, c])?),
                ] {
                    if *bf16 {
                        ensure!(meta.dtype == DType::Bf16, "BF16 KV-B {name}");
                        sizes.push((key.into(), extent(&[values, 2])?.max(256)));
                    } else {
                        sizes.push((format!("{key}_fp8"), values.max(256)));
                        sizes.push((format!("{key}_scale"), extent(&[values / 64, 4])?.max(256)));
                    }
                }
            }
            Operation::HeadVector(name) => {
                let meta = self.meta(name)?;
                ensure!(
                    meta.dtype == DType::Bf16
                        && meta.shape == [self.cfg.heads]
                        && self.cfg.heads % parts == 0,
                    "query head vector {name}"
                );
                sizes.push((key.into(), (meta.byte_length as usize / parts).max(256)));
            }
            Operation::Core(layout) => {
                let [n, k] = layout.shape;
                sizes.extend([
                    (
                        format!("core.{key}.payload"),
                        extent(&[n, k, layout.bits])? / 8,
                    ),
                    (format!("core.{key}.suh"), extent(&[k, 2])?),
                    (format!("core.{key}.svh"), extent(&[n, 2])?),
                ]);
            }
            Operation::NativeAttention(layout) => {
                let [n, k] = layout.shape;
                sizes.push((format!("native.{key}.values"), extent(&[n, k, if layout.fp8 { 1 } else { 2 }])?.max(256)));
                if layout.fp8 {
                    sizes.push((format!("native.{key}.scales"), extent(&[n.div_ceil(128), k.div_ceil(128), 4])?.max(256)));
                }
            }
            Operation::TensorScales(names) => {
                ensure!(
                    !names.is_empty() && names.len() <= 2,
                    "tensor-scale member count"
                );
                for name in names {
                    let stem = name.strip_suffix("weight").context("tensor-scale stem")?;
                    for scalar in [format!("{name}_scale"), format!("{stem}input_scale")] {
                        let meta = self.meta(&scalar)?;
                        ensure!(
                            meta.dtype == DType::F32 && meta.byte_length == 4,
                            "{scalar}: expected one FP32 value"
                        );
                    }
                }
                sizes.push((key.into(), 256));
            }
        }
        let layer = self.layer;
        let requests = ranks
            .iter()
            .flat_map(|&rank| {
                sizes.iter().map(move |(name, bytes)| {
                    let name = if name.starts_with("core.") || name.starts_with("native.") {
                        name.clone()
                    } else {
                        format!("operand.{name}")
                    };
                    Allocation::buffer(
                        format!("glm.weights.gpu.{rank}.layer.{layer}.{name}"),
                        Location::coordinator(rank as u8),
                        *bytes as u64,
                        Lifetime::RESIDENT,
                        Category::Weights,
                        "GLM target weights",
                    )
                })
            })
            .collect();
        self.routes.push(Route {
            ranks,
            key: key.into(),
            operation,
            requests,
        });
        Ok(())
    }
}

/// The actual loader's format and owner dispatch. A receiver may execute the
/// operation or collect its validated requests without touching tensor payloads.
pub(super) fn layer_routes(
    catalog: &OfficialV41Catalog,
    cfg: &GlmDsaConfig,
    layer: usize,
    checkpoint: &cuteafd_loader::plan::Checkpoint,
    core: &CoreInventory,
    ranks: usize,
    owner: Option<u8>,
    native_shared: bool,
    native_bf16: bool,
    mut receive: impl FnMut(Route) -> Result<()>,
) -> Result<()> {
    ensure!(
        matches!(ranks, 1 | 2)
            && layer < cfg.layers
            && owner.is_none_or(|rank| ranks == 2 && usize::from(rank) < ranks),
        "GLM weight route ownership"
    );
    let mut r = Routes {
        catalog,
        cfg,
        layer,
        ranks,
        attention: owner.map_or_else(|| (0..ranks).collect(), |rank| vec![usize::from(rank)]),
        routes: Vec::new(),
    };
    for (key, op) in [
        ("input_norm", Op::InputNorm),
        ("post_norm", Op::PostAttentionNorm),
    ] {
        r.push(false, key, Operation::Tensor(r.weight(op)?))?;
    }
    let core_attention = [Op::QA, Op::KvA, Op::QB, Op::O].into_iter()
        .filter_map(|op| cfg.stem(layer, op)).all(|stem|
            core.get_weight(&stem).is_some() || core.native.contains_key(&stem));
    cuteafd_loader::families::glm5::core::validate_attention_format(cfg, layer, core)?;
    let attention = [Op::QA, Op::KvA, Op::QB, Op::KvB, Op::O]
        .into_iter()
        .filter_map(|op| cfg.weight(layer, op))
        .collect::<Vec<_>>();
    let bf16 = core_attention || r.native_bf16(&attention, native_bf16)?;
    for (key, op) in [("q_a_norm", Op::QANorm), ("kv_a_norm", Op::KvANorm)] {
        if let Some(name) = cfg.weight(layer, op) {
            r.push(true, key, Operation::Tensor(name))?;
        }
    }
    for (key, op) in [
        ("query_scale_s", Op::QueryScaleS),
        ("query_scale_b", Op::QueryScaleB),
    ] {
        if let Some(name) = cfg.weight(layer, op) {
            r.push(true, key, Operation::HeadVector(name))?;
        }
    }
    if !core_attention {
        let names = [Op::QA, Op::KvA]
            .into_iter()
            .filter_map(|op| cfg.weight(layer, op))
            .collect();
        r.push(
            true,
            "w_qkv_a",
            Operation::Projection {
                names,
                axis: Axis::Rows,
                bf16,
                kmajor: !bf16,
                replicated: true,
            },
        )?;
        for (key, op, axis) in [("w_q_b", Op::QB, Axis::Rows), ("w_o", Op::O, Axis::Cols)] {
            r.push(
                true,
                key,
                Operation::Projection {
                    names: vec![r.weight(op)?],
                    axis,
                    bf16,
                    kmajor: false,
                    replicated: false,
                },
            )?;
        }
    }
    r.push(true, "kv_b", Operation::KvB { layer, bf16 })?;
    if cfg.indexers[layer] == GlmIndexer::Full {
        let name = r.weight(Op::IndexQ)?;
        if core.get_weight(&name).is_none() {
            if !cfg.variant.is_glm() {
                ensure!(
                    landing::native_fp8_block(catalog, &name)?,
                    "{name}: Mistral Large 4 serves the indexer wq_b from the K6 core or its shipped FP8 block-128 form only"
                );
            }
            let bf16 = cfg.variant.is_glm()
                && (core_attention || r.native_bf16(std::slice::from_ref(&name), native_bf16)?);
            r.push(
                true,
                "w_iq",
                Operation::Projection {
                    names: vec![name],
                    axis: Axis::Rows,
                    bf16,
                    kmajor: false,
                    replicated: true,
                },
            )?;
        }
        r.push(
            true,
            "w_ik",
            Operation::Rows(vec![r.weight(Op::IndexK)?, r.weight(Op::IndexWeights)?]),
        )?;
        for (key, name) in cfg.plain_owners(layer).into_iter().filter(|(key, _)| {
            [
                "k_norm_w",
                "k_norm_b",
                "index_q_norm_w",
                "index_q_norm_b",
                "index_scale_s",
                "index_scale_b",
            ]
            .contains(key)
        }) {
            r.push(true, key, Operation::Tensor(name))?;
        }
    }
    let dense = layer < cfg.first_moe_layer;
    let gate = cfg.stem(layer, Op::Gate).context("FFN gate")?;
    if (dense || !native_shared) && core.get_weight(&gate).is_none() {
        let names = vec![r.weight(Op::Gate)?, r.weight(Op::Up)?];
        let down = vec![r.weight(Op::Down)?];
        let bf16 = r.native_bf16(&[names.clone(), down.clone()].concat(), native_bf16)?;
        r.push(
            false,
            "w_gate_up",
            Operation::Projection {
                names: names.clone(),
                axis: Axis::Rows,
                bf16,
                kmajor: false,
                replicated: false,
            },
        )?;
        r.push(
            false,
            "w_down",
            Operation::Projection {
                names: down.clone(),
                axis: Axis::Cols,
                bf16,
                kmajor: false,
                replicated: false,
            },
        )?;
        let scales = |names: &[String]| -> Result<bool> {
            Ok(names.iter().all(|name| {
                catalog
                    .tensor(name)
                    .is_ok_and(|t| t.metadata.dtype == DType::F8E4M3)
                    && catalog.tensor(&format!("{name}_scale")).is_ok()
                    && catalog
                        .tensor(&format!(
                            "{}input_scale",
                            name.strip_suffix("weight").unwrap()
                        ))
                        .is_ok()
            }))
        };
        if dense && scales(&names)? && scales(&down)? {
            r.push(false, "w_gate_up_tscale", Operation::TensorScales(names))?;
            r.push(false, "w_down_tscale", Operation::TensorScales(down))?;
        }
        let numerics = checkpoint.module_numerics(&gate)?;
        for op in [Op::Up, Op::Down] {
            let stem = cfg.stem(layer, op).context("FFN projection")?;
            ensure!(
                checkpoint.module_numerics(&stem)? == numerics,
                "{gate}: mixed declared MLP arithmetic is unsupported"
            );
        }
    }
    if !dense {
        for (key, op) in [("gate", Op::Router), ("gate.bias", Op::RouterBias)] {
            if let Some(name) = cfg.weight(layer, op) {
                let before = r.routes.len();
                r.push(false, key, Operation::Tensor(name))?;
                r.routes[before].ranks.retain(|&rank| rank == 0);
                r.routes[before]
                    .requests
                    .retain(|request| request.location == Location::coordinator(0));
            }
        }
    }
    for (rank, layouts) in core
        .layer_weights(cfg, layer, ranks, owner)?
        .into_iter()
        .enumerate()
    {
        for (key, layout) in layouts {
            r.push(false, key, Operation::Core(layout))?;
            let last = r.routes.last_mut().unwrap();
            last.ranks.retain(|&r| r == rank);
            last.requests
                .retain(|request| request.location == Location::coordinator(rank as u8));
        }
    }
    for (rank, layouts) in core.layer_native(cfg, layer, ranks, owner)?.into_iter().enumerate() {
        for (key, layout) in layouts {
            r.push(false, key, Operation::NativeAttention(layout))?;
            let last = r.routes.last_mut().unwrap();
            last.ranks.retain(|&r| r == rank);
            last.requests.retain(|request| request.location == Location::coordinator(rank as u8));
        }
    }
    for route in r.routes {
        receive(route)?;
    }
    Ok(())
}

/// Global operations share the layer uploader's preallocation contract.
pub(super) fn global_routes(
    catalog: &OfficialV41Catalog,
    cfg: &GlmDsaConfig,
    core: &CoreInventory,
    mut receive: impl FnMut(Route) -> Result<()>,
) -> Result<()> {
    let mut r = Routes {
        catalog,
        cfg,
        layer: 0,
        ranks: 1,
        attention: vec![0],
        routes: Vec::new(),
    };
    r.push(false, "norm", Operation::Tensor(cfg.final_norm().into()))?;
    if let Some(projection) = core.get_weight(cfg.head()) {
        r.push(
            false,
            "head",
            Operation::Core(SharedCoreLayout::new(
                projection,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Output,
                0,
                1,
            )?),
        )?;
    } else {
        r.push(
            false,
            "head",
            Operation::Tensor(format!("{}.weight", cfg.head())),
        )?;
    }
    for mut route in r.routes {
        for request in &mut route.requests {
            request.id = request
                .id
                .replace(".layer.0.operand.", ".")
                .replace(".layer.0.core.", ".core.");
        }
        receive(route)?;
    }
    Ok(())
}
