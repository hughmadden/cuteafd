//! Immutable, deduplicated native rotary tables, built once before requests.
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    programs::{Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::rope::{Recipe, Tables};
use std::rc::Rc;

pub(crate) struct Table<'a> {
    pub cos: Allocation<'a>,
    pub sin: Allocation<'a>,
}

pub(crate) struct RotaryTables<'a> {
    pub plan: Tables,
    tables: Vec<Rc<Table<'a>>>,
}

impl<'a> RotaryTables<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &Programs<'a>,
        plan: Tables,
        prefix: &str,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        // Budget and every ABI are checked before the first device allocation.
        plan.check_budget(available, reservations)?;
        for recipe in &plan.recipes {
            let name = format!("{prefix}{}", recipe.program());
            let spec = programs.spec(&name)?;
            ensure!(
                spec.scalar_kinds == b"ifffff",
                "rotary table scalar ABI mismatch"
            );
            ensure!(
                spec.geometry["rotary_span"].as_u64() == Some(recipe.span as u64)
                    && spec.geometry["llama3"].as_bool() == Some(recipe.scaling.is_some()),
                "rotary table geometry mismatch"
            );
            ensure!(
                spec.pointers == ["cos", "sin"],
                "rotary table pointer ABI mismatch"
            );
        }
        let mut tables = Vec::new();
        for recipe in &plan.recipes {
            let program = device.run(|| {
                programs.program(&format!("{prefix}{}", recipe.program()), &["cos", "sin"])
            })?;
            let plane = usize::try_from(recipe.allocation_bytes(plan.extent)? / 2)?;
            let _scope = cuteafd_ffi::memory_ledger::scope("rotary_tables");
            let table = Table {
                cos: Allocation::new(device, plane)?,
                sin: Allocation::new(device, plane)?,
            };
            // The stream is created after storage, so error unwinding drains before freeing.
            let stream = Stream::new(device)?;
            let scaling = recipe.scaling;
            let scalars = [
                Scalar::I32(plan.extent as i32),
                Scalar::F32(recipe.theta),
                Scalar::F32(scaling.map_or(1.0, |s| s.factor)),
                Scalar::F32(scaling.map_or(plan.extent as f32, |s| s.original_max)),
                Scalar::F32(scaling.map_or(1.0, |s| s.low_factor)),
                Scalar::F32(scaling.map_or(32.0, |s| s.high_factor)),
            ];
            device.run(|| {
                // SAFETY: both admitted BF16 planes belong to this device, span the
                // complete extent, and outlive the drained load-only stream.
                unsafe {
                    program.launch(
                        &[table.cos.buffer.ptr, table.sin.buffer.ptr],
                        &scalars,
                        stream.raw,
                    )
                }
            })?;
            stream.drain()?;
            tables.push(Rc::new(table));
        }
        Ok(Self { plan, tables })
    }

    pub fn get(&self, recipe: Recipe) -> Result<Rc<Table<'a>>> {
        let index = self
            .plan
            .recipes
            .iter()
            .position(|r| *r == recipe)
            .context("rotary recipe was not admitted")?;
        Ok(Rc::clone(&self.tables[index]))
    }
}

impl Table<'_> {
    pub fn buffers(&self) -> (CuteafdDeviceBuffer, CuteafdDeviceBuffer) {
        (self.cos.buffer, self.sin.buffer)
    }
}
