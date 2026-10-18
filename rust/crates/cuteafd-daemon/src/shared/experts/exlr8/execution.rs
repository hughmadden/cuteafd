//! Admitted K1 workspace: graph-visible routing storage never moves.
use super::{Exlr8Weights, view};
use crate::shared::memory::DeviceAllocation;
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RoutingIdentity {
    addresses: [usize; 4],
    rows: usize,
    top_k: usize,
    max_groups: usize,
}

#[derive(Default)]
struct RoutingLifetime {
    captured: Option<RoutingIdentity>,
}
impl RoutingLifetime {
    fn check(&self, candidate: RoutingIdentity) -> Result<()> {
        ensure!(
            self.captured.is_none_or(|bound| bound == candidate),
            "EXLR8 routing storage or shape changed while a graph retains it"
        );
        Ok(())
    }
    fn capture(&mut self, identity: RoutingIdentity) -> Result<()> {
        self.check(identity)?;
        self.captured = Some(identity);
        Ok(())
    }
}

#[derive(Debug)]
struct Workspace {
    ranges: [(usize, usize); 10],
    bytes: usize,
    rows: usize,
    top_k: usize,
    max_groups: usize,
    hidden: usize,
    units: usize,
}
impl Workspace {
    fn plan(program: &Program<'_>, rows: usize) -> Result<Self> {
        let spec = program.spec();
        ensure!(
            spec.name == "exlr8_a16_serving" && spec.geometry["abi_version"] == 2,
            "EXLR8 execution requires device-count ABI v2"
        );
        ensure!(
            rows > 0 && rows <= 64 && rows <= spec.capacity_rows as usize,
            "EXLR8 rows outside admitted K1 capacity"
        );
        let dimension = |name| -> Result<usize> {
            usize::try_from(
                spec.geometry[name]
                    .as_u64()
                    .context("missing K1 geometry")?,
            )
            .context("K1 geometry exceeds address space")
        };
        let hidden = dimension("hidden")?;
        let top_k = dimension("top_k")?;
        let max_groups = dimension("max_groups")?;
        let units = dimension("units")?;
        let product = |a: usize, b: usize| a.checked_mul(b).context("EXLR8 workspace overflow");
        let routes = product(rows, top_k)?;
        let scratch = |name: &str| -> Result<usize> {
            usize::try_from(*spec.scratch.get(name).context("missing K1 scratch")?)
                .context("K1 scratch exceeds address space")
        };
        // x, ids, route weights, output, then the four graph-stable route tables,
        // and the K1 intermediate planes. All are admitted in one allocation.
        let sizes = [
            product(product(rows, hidden)?, 2)?,
            product(routes, 4)?,
            product(routes, 4)?,
            product(product(rows, hidden)?, 4)?,
            product(max_groups, 4)?,
            product(product(max_groups, 16)?, 4)?,
            product(routes, 4)?,
            4,
            scratch("fc1")?,
            scratch("intermediate")?
                .checked_add(scratch("fc2")?)
                .context("K1 scratch overflow")?,
        ];
        let mut bytes = 0usize;
        let mut ranges = [(0, 0); 10];
        for (range, size) in ranges.iter_mut().zip(sizes) {
            let start = bytes
                .checked_add(255)
                .context("workspace alignment overflow")?
                & !255;
            bytes = start.checked_add(size).context("workspace size overflow")?;
            *range = (start, size);
        }
        Ok(Self {
            ranges,
            bytes,
            rows,
            top_k,
            max_groups,
            hidden,
            units,
        })
    }
}

pub(crate) struct Exlr8Execution<'a> {
    group: Program<'a>,
    k1: Program<'a>,
    // Private, fixed storage: no request can resize or replace captured pointers.
    arena: DeviceAllocation<'a>,
    layout: Workspace,
    routing: RoutingLifetime,
    intermediate_bytes: usize,
    cta_limit: i32,
}
impl<'a> Exlr8Execution<'a> {
    pub(crate) fn new(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
        rows: usize,
        available: usize,
    ) -> Result<Self> {
        let k1 = programs.program(
            "exlr8_a16_serving",
            &[
                "x",
                "pointers",
                "bits",
                "group_expert",
                "route_slots",
                "route_group_lane",
                "route_weights",
                "fc1",
                "intermediate",
                "fc2",
                "output",
                "group_count",
            ],
        )?;
        let group = programs.program(
            "exlr8_group",
            &[
                "ids",
                "group_expert",
                "route_slots",
                "route_group_lane",
                "group_count",
            ],
        )?;
        let layout = Workspace::plan(&k1, rows)?;
        ensure!(
            group.spec().geometry["top_k"] == k1.spec().geometry["top_k"]
                && group.spec().geometry["max_groups"] == k1.spec().geometry["max_groups"],
            "EXLR8 grouping and K1 geometry differ"
        );
        ensure!(
            layout.bytes <= available,
            "EXLR8 workspace exceeds admitted device bytes"
        );
        let intermediate_bytes = usize::try_from(k1.spec().scratch["intermediate"])?;
        let ctas = i32::try_from(
            k1.spec().geometry["ctas_per_sm"]
                .as_u64()
                .context("CTA hint")?,
        )?;
        let cta_limit = i32::try_from(library.sm_count()?)?
            .checked_mul(ctas)
            .context("CTA limit overflow")?;
        ensure!(cta_limit > 0, "runtime SM count must be positive");
        let arena = DeviceAllocation::named(library, layout.bytes, "experts/workspace/exlr8")?;
        Ok(Self {
            group,
            k1,
            arena,
            layout,
            routing: RoutingLifetime::default(),
            intermediate_bytes,
            cta_limit,
        })
    }
    fn buffer(&self, index: usize) -> Result<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.layout.ranges[index];
        view(self.arena.buffer, offset, bytes)
    }
    fn identity(&self) -> Result<RoutingIdentity> {
        Ok(RoutingIdentity {
            addresses: [
                self.buffer(4)?.ptr as usize,
                self.buffer(5)?.ptr as usize,
                self.buffer(6)?.ptr as usize,
                self.buffer(7)?.ptr as usize,
            ],
            rows: self.layout.rows,
            top_k: self.layout.top_k,
            max_groups: self.layout.max_groups,
        })
    }
    /// Called before capture. Retain this owner until every graph is destroyed
    /// and queued replay work has drained; binding or shape changes are refused.
    pub(crate) fn seal_graph_storage(&mut self) -> Result<()> {
        self.routing.capture(self.identity()?)
    }
    pub(crate) fn input(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(0)
    }
    pub(crate) fn ids(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(1)
    }
    pub(crate) fn route_weights(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(2)
    }
    pub(crate) fn output(&self) -> Result<CuteafdDeviceBuffer> {
        self.buffer(3)
    }

    /// # Safety
    /// Inputs and weights are retained on this device until the caller drains
    /// `stream`. The caller also retains this workspace through graph replay.
    pub(crate) unsafe fn run(
        &self,
        weights: &Exlr8Weights<'_>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.layout.rows,
            "EXLR8 live rows exceed admission"
        );
        ensure!(
            weights.layout.hidden == self.layout.hidden
                && weights.layout.units == self.layout.units,
            "EXLR8 weights and workspace geometry differ"
        );
        ensure!(
            self.group.spec().geometry["experts"].as_u64() == Some(weights.layout.experts as u64),
            "EXLR8 grouping expert count differs from residency"
        );
        self.routing.check(self.identity()?)?;
        let buffers = (0..10)
            .map(|i| self.buffer(i))
            .collect::<Result<Vec<_>>>()?;
        let p = |i: usize| buffers[i].ptr;
        let intermediate = view(buffers[9], 0, self.intermediate_bytes)?;
        let fc2 = view(
            buffers[9],
            self.intermediate_bytes,
            buffers[9].bytes - self.intermediate_bytes,
        )?;
        // SAFETY: private admitted arena views retain the manifest's exact sizes;
        // weights own all referenced descriptors, and caller retains the stream.
        unsafe {
            self.group.launch(
                &[p(1), p(4), p(5), p(6), p(7)],
                &[Scalar::I32(rows as i32)],
                stream,
            )?;
            self.k1.launch(
                &[
                    p(0),
                    weights.pointers()?.ptr,
                    weights.bits()?.ptr,
                    p(4),
                    p(5),
                    p(6),
                    p(2),
                    p(8),
                    intermediate.ptr,
                    fc2.ptr,
                    p(3),
                    p(7),
                ],
                &[Scalar::I32(rows as i32), Scalar::I32(self.cta_limit)],
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_reallocated_route_storage_or_shape_after_capture() {
        let identity = RoutingIdentity {
            addresses: [256, 512, 768, 1024],
            rows: 64,
            top_k: 8,
            max_groups: 512,
        };
        let mut lifetime = RoutingLifetime::default();
        lifetime.capture(identity).unwrap();
        lifetime.check(identity).unwrap();
        for index in 0..4 {
            let mut replacement = identity;
            replacement.addresses[index] += 4096;
            assert!(lifetime.check(replacement).is_err());
            assert!(lifetime.capture(replacement).is_err());
        }
        let mut changed = identity;
        changed.rows = 32;
        assert!(lifetime.check(changed).is_err());
        let mut changed = identity;
        changed.top_k = 4;
        assert!(lifetime.check(changed).is_err());
        let mut changed = identity;
        changed.max_groups = 1024;
        assert!(lifetime.check(changed).is_err());
        lifetime.check(identity).unwrap();
    }
}
