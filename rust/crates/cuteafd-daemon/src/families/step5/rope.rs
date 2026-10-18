//! Step-5 selects checkpoint recipes; shared code owns and initializes storage.
use crate::shared::memory::device::Device;
use crate::shared::rope::{RotaryTables, Table};
use anyhow::Result;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract;
use std::rc::Rc;

pub(super) struct LayerTables<'a> {
    pub attention: Rc<Table<'a>>,
    pub indexer: Option<Rc<Table<'a>>>,
}

pub(super) struct TargetTables<'a> {
    pub tables: RotaryTables<'a>,
    pub layers: Vec<LayerTables<'a>>,
}

impl<'a> TargetTables<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &Programs<'a>,
        contract: &CoordinatorContract,
        prefix: &str,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        let tables = RotaryTables::load(
            device,
            programs,
            contract.rope_tables.clone(),
            prefix,
            available,
            reservations,
        )?;
        let layers = contract
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerTables {
                    attention: tables.get(layer.rope)?,
                    indexer: layer
                        .indexer_rope
                        .map(|recipe| tables.get(recipe))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tables, layers })
    }
}
