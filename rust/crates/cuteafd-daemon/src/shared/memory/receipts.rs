//! Passive owner inventory. Every extent and address comes from retained storage.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer};
use cuteafd_loader::placement::planner::{Allocation, CoordinatorDevices, Lifetime, Location, PhaseOrder, Receipt, View};
use std::collections::BTreeMap;

/// Cross-domain aliases are observations, never another physical allocation.
#[derive(Debug, Clone)]
pub(crate) struct MappedAlias {
    pub owner: String,
    pub backing: String,
    pub location: Location,
    pub address: u64,
    pub bytes: u64,
}

/// Retained storage only. Declared lifetimes are not observed startup history.
#[derive(Debug, Default)]
pub(crate) struct Census {
    pub receipts: Vec<Receipt>,
    pub aliases: Vec<MappedAlias>,
    /// Diagnostic capacities lie inside the process RSS envelope, not beside it.
    pub heap_diagnostics: Vec<Receipt>,
    pub unclassified: Vec<cuteafd_transport::UnclassifiedTransportOwner>,
}
impl Census {
    pub(crate) fn extend(&mut self, other: Self) {
        self.receipts.extend(other.receipts);
        self.aliases.extend(other.aliases);
        self.heap_diagnostics.extend(other.heap_diagnostics);
        self.unclassified.extend(other.unclassified);
    }
    pub(crate) fn owners(&mut self, receipts: Vec<Receipt>) {
        for receipt in receipts {
            if receipt.allocation.location == Location::HOST {
                self.heap_diagnostics.push(receipt);
            } else {
                self.receipts.push(receipt);
            }
        }
    }
    pub(crate) fn mapped(&mut self, owner: String, backing: String,
        buffer: CuteafdDeviceBuffer, devices: CoordinatorDevices<'_>) -> Result<()> {
        ensure!(!buffer.ptr.is_null() && buffer.bytes > 0, "empty mapped alias {owner}");
        self.aliases.push(MappedAlias { owner, backing, location: devices.location(buffer.device_id)?,
            address: buffer.ptr as u64, bytes: buffer.bytes as u64 });
        Ok(())
    }
    pub(crate) fn mapped_pinned(&mut self, library: &cuteafd_ffi::NativeLibrary,
        devices: CoordinatorDevices<'_>, owner: String, buffer: CuteafdHostBuffer) -> Result<()> {
        ensure!(buffer.flags & cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_MAPPED != 0,
            "retained host owner is not mapped: {owner}");
        let backing = self.receipts.iter().find(|receipt| receipt.allocation.view.is_none()
            && receipt.allocation.location == Location::PINNED && receipt.address == buffer.ptr as u64
            && receipt.allocation.bytes == buffer.bytes as u64)
            .context("mapped pinned owner has no whole retained backing")?.allocation.id.clone();
        let alias = library.cuda_host_buffer_device_alias(buffer)?;
        ensure!(alias.bytes == buffer.bytes, "mapped alias extent differs from backing: {owner}");
        self.mapped(owner, backing, alias, devices)
    }
    pub(crate) fn transport(&mut self, prefix: &str, transport: cuteafd_transport::TransportAllocationCensus,
        devices: CoordinatorDevices<'_>, lifetime: Lifetime) -> Result<()> {
        ensure!(transport.pinned_frame_audits.is_empty(), "unresolved pinned-frame audit in {prefix}");
        self.unclassified.extend(transport.unclassified.into_iter().map(|mut owner| {
            owner.owner = format!("{prefix}.{}", owner.owner);
            owner
        }));
        for host in transport.host {
            let id = format!("{prefix}.{}", host.owner);
            ensure!(host.address != 0 && host.bytes > 0, "empty transport owner {id}");
            if host.flags & cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED == 0 {
                self.unclassified.push(cuteafd_transport::UnclassifiedTransportOwner {
                    owner: id, reason: "transport backing is not identified as pinned".into(),
                });
                continue;
            }
            self.receipts.push(Receipt { allocation: Allocation::buffer(&id, Location::PINNED,
                host.bytes, lifetime, Category::Transport, "transport backing"), address: host.address });
            if let Some(alias) = host.mapped {
                ensure!(alias.address != 0, "null mapped transport alias {id}");
                self.aliases.push(MappedAlias { owner: format!("{id}.mapped"), backing: id,
                    location: devices.location(alias.device)?, address: alias.address, bytes: host.bytes });
            }
        }
        Ok(())
    }

    /// Exact repeated backing becomes a view. Partial overlap without an explicit
    /// backing relationship is an error, not evidence that one owner contains another.
    pub(crate) fn normalize(&mut self, order: PhaseOrder) -> Result<()> {
        // Independent tensor inventories can name the same retained arena.
        // Only an identical physical backing may repeat its stable ID.
        let mut unique: Vec<Receipt> = Vec::with_capacity(self.receipts.len());
        let mut by_id: BTreeMap<String, usize> = BTreeMap::new();
        for receipt in self.receipts.drain(..) {
            ensure!(order.at_or_before(receipt.allocation.lifetime.first, receipt.allocation.lifetime.last),
                "invalid declared lifetime {}", receipt.allocation.id);
            if let Some(&index) = by_id.get(&receipt.allocation.id) {
                let retained: &mut Receipt = &mut unique[index];
                ensure!(retained.allocation.view.is_none() && receipt.allocation.view.is_none()
                    && retained.address == receipt.address
                    && retained.allocation.location == receipt.allocation.location
                    && retained.allocation.bytes == receipt.allocation.bytes
                    && retained.allocation.category == receipt.allocation.category
                    && retained.allocation.group == "weight arena" && receipt.allocation.group == "weight arena",
                    "conflicting repeated backing {}", receipt.allocation.id);
                let lifetime = receipt.allocation.lifetime;
                let current = &mut retained.allocation.lifetime;
                if order.at_or_before(lifetime.first, current.first) { current.first = lifetime.first; }
                if order.at_or_before(current.last, lifetime.last) { current.last = lifetime.last; }
            } else {
                by_id.insert(receipt.allocation.id.clone(), unique.len());
                unique.push(receipt);
            }
        }
        self.receipts = unique;
        let mut ids = BTreeMap::new();
        for (index, receipt) in self.receipts.iter().enumerate() {
            let allocation = &receipt.allocation;
            ensure!(!allocation.id.is_empty() && receipt.address != 0 && allocation.bytes > 0,
                "empty retained owner {}", allocation.id);
            ensure!(ids.insert(allocation.id.clone(), index).is_none(), "duplicate owner {}", allocation.id);
            ensure!(order.at_or_before(allocation.lifetime.first, allocation.lifetime.last),
                "invalid declared lifetime {}", allocation.id);
            receipt.address.checked_add(allocation.bytes).context("retained owner address overflow")?;
        }
        let mut backing: BTreeMap<(Location, u64, u64), usize> = BTreeMap::new();
        let mut canonical = BTreeMap::new();
        for index in 0..self.receipts.len() {
            let receipt = &self.receipts[index];
            if receipt.allocation.view.is_some() { continue; }
            let key = (receipt.allocation.location, receipt.address, receipt.allocation.bytes);
            if let Some(&owner) = backing.get(&key) {
                let id = self.receipts[owner].allocation.id.clone();
                canonical.insert(receipt.allocation.id.clone(), id.clone());
                let lifetime = receipt.allocation.lifetime;
                let retained = &mut self.receipts[owner].allocation.lifetime;
                if order.at_or_before(lifetime.first, retained.first) { retained.first = lifetime.first; }
                if order.at_or_before(retained.last, lifetime.last) { retained.last = lifetime.last; }
                self.receipts[index].allocation.view = Some(View { backing: id, offset: 0 });
            } else {
                backing.insert(key, index);
            }
        }
        let mut previous: Option<(Location, u64)> = None;
        for ((location, address, bytes), _) in backing {
            if let Some((prior_location, end)) = previous {
                ensure!(prior_location != location || end <= address, "overlapping unclassified backing at {address:#x}");
            }
            previous = Some((location, address + bytes));
        }
        for receipt in &mut self.receipts {
            if let Some(view) = &mut receipt.allocation.view {
                if let Some(id) = canonical.get(&view.backing) { view.backing = id.clone(); }
            }
        }
        for receipt in &self.receipts {
            let Some(view) = &receipt.allocation.view else { continue };
            let owner = &self.receipts[*ids.get(&view.backing).context("missing retained backing")?];
            ensure!(owner.allocation.view.is_none() && owner.allocation.location == receipt.allocation.location
                && view.offset.checked_add(receipt.allocation.bytes).is_some_and(|end| end <= owner.allocation.bytes)
                && owner.address.checked_add(view.offset) == Some(receipt.address)
                && order.at_or_before(owner.allocation.lifetime.first, receipt.allocation.lifetime.first)
                && order.at_or_before(receipt.allocation.lifetime.last, owner.allocation.lifetime.last),
                "invalid retained view {}", receipt.allocation.id);
        }
        let mut alias_ids = std::collections::BTreeSet::new();
        for alias in &mut self.aliases {
            ensure!(alias_ids.insert(alias.owner.clone()) && !ids.contains_key(&alias.owner),
                "duplicate mapped alias {}", alias.owner);
            if let Some(id) = canonical.get(&alias.backing) { alias.backing = id.clone(); }
            let owner = &self.receipts[*ids.get(&alias.backing).context("missing mapped backing")?];
            ensure!(owner.allocation.location == Location::PINNED && owner.allocation.view.is_none()
                && alias.location.domain == cuteafd_loader::placement::planner::Domain::Device
                && alias.address != 0 && alias.bytes == owner.allocation.bytes,
                "invalid mapped alias {}", alias.owner);
            alias.address.checked_add(alias.bytes).context("mapped alias address overflow")?;
        }
        Ok(())
    }
}

pub(crate) struct Inventory<'a> {
    pub receipts: Vec<Receipt>,
    devices: CoordinatorDevices<'a>,
    prefix: String,
    lifetime: Lifetime,
    category: Category,
    group: String,
}
impl<'a> Inventory<'a> {
    pub(crate) fn new(devices: CoordinatorDevices<'a>, prefix: &str, lifetime: Lifetime,
        category: Category, group: &str) -> Self {
        Self { receipts: Vec::new(), devices, prefix: prefix.into(), lifetime, category, group: group.into() }
    }
    pub(crate) fn location(&self, physical_device: i32) -> Result<Location> {
        Ok(self.devices.location(physical_device)?)
    }
    pub(crate) fn lifetime(&self) -> Lifetime { self.lifetime }
    pub(crate) fn extend(&mut self, receipts: Vec<Receipt>) -> Result<()> {
        let arenas = receipts.iter().filter(|r| r.allocation.view.is_none()
            && r.allocation.group == "weight arena")
            .map(|r| r.allocation.id.clone()).collect::<std::collections::BTreeSet<_>>();
        for mut receipt in receipts {
            let arena = arenas.contains(&receipt.allocation.id);
            if !arena { receipt.allocation.id = format!("{}.{}", self.prefix, receipt.allocation.id); }
            if let Some(view) = &mut receipt.allocation.view {
                if !arenas.contains(&view.backing) { view.backing = format!("{}.{}", self.prefix, view.backing); }
            }
            if let Some(existing) = self.receipts.iter().find(|r| r.allocation.id == receipt.allocation.id) {
                ensure!(arena && existing.address == receipt.address && existing.allocation == receipt.allocation,
                    "duplicate imported owner {}", receipt.allocation.id);
            } else { self.receipts.push(receipt); }
        }
        Ok(())
    }
    fn add(&mut self, name: &str, location: Location, address: u64, bytes: u64) -> Result<()> {
        ensure!(!name.is_empty() && address != 0 && bytes != 0, "empty actual allocation {name}");
        let id = format!("{}.{name}", self.prefix);
        ensure!(!self.receipts.iter().any(|r| r.allocation.id == id), "duplicate owner {id}");
        self.receipts.push(Receipt { allocation: Allocation::buffer(id, location, bytes,
            self.lifetime, self.category, &self.group), address });
        Ok(())
    }
    pub(crate) fn device(&mut self, name: &str, buffer: CuteafdDeviceBuffer) -> Result<()> {
        self.add(name, self.devices.location(buffer.device_id)?, buffer.ptr as u64, buffer.bytes as u64)
    }
    pub(crate) fn weight_view(&mut self, name: &str, buffer: CuteafdDeviceBuffer,
        backing: super::device::WeightArenaBacking<'_>) -> Result<()> {
        let owner = backing.buffer;
        let location = self.devices.location(owner.device_id)?;
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("weight view before backing")?;
        ensure!(!backing.id.is_empty() && !owner.ptr.is_null() && owner.bytes > 0
            && buffer.device_id == owner.device_id && buffer.bytes > 0
            && offset.checked_add(buffer.bytes as u64).is_some_and(|end| end <= owner.bytes as u64),
            "weight view {name} outside backing {}", backing.id);
        if let Some(receipt) = self.receipts.iter().find(|r| r.allocation.id == backing.id) {
            ensure!(receipt.allocation.view.is_none() && receipt.allocation.location == location
                && receipt.address == owner.ptr as u64 && receipt.allocation.bytes == owner.bytes as u64,
                "weight arena {} changed backing", backing.id);
        } else {
            self.receipts.push(Receipt { allocation: Allocation::buffer(backing.id, location,
                owner.bytes as u64, self.lifetime, backing.kind.category(), "weight arena"), address: owner.ptr as u64 });
        }
        self.device(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View { backing: backing.id.into(), offset });
        Ok(())
    }
    pub(crate) fn backed_pinned(&mut self, name: &str, buffer: CuteafdHostBuffer,
        owner: CuteafdHostBuffer) -> Result<()> {
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("host view before backing")?;
        ensure!(buffer.bytes > 0 && offset.checked_add(buffer.bytes as u64).is_some_and(|n| n <= owner.bytes as u64),
            "host view {name} exceeds backing");
        let backing = self.receipts.iter().find(|receipt| receipt.allocation.view.is_none()
            && receipt.allocation.location == Location::PINNED && receipt.address == owner.ptr as u64
            && receipt.allocation.bytes == owner.bytes as u64).map(|r| r.allocation.id.clone());
        let backing = match backing {
            Some(id) => id,
            None => {
                self.pinned(&format!("{name}.backing"), owner)?;
                format!("{}.{name}.backing", self.prefix)
            }
        };
        self.pinned(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View { backing, offset });
        Ok(())
    }
    /// Existing per-destination stores only: looking up an unprimed owner never
    /// reserves staging or waits for a copy. Actual aliases keep their own keys.
    pub(crate) fn ordered_uploads(&mut self, library: &cuteafd_ffi::NativeLibrary,
        stream: *mut std::ffi::c_void) -> Result<()> {
        use cuteafd_loader::placement::planner::{Device, Domain};
        let mut uploads = Vec::new();
        let mut observed = std::collections::BTreeSet::new();
        let backings = self.receipts.iter().filter_map(|receipt|
            receipt.allocation.view.as_ref().map(|view| view.backing.as_str())).collect::<std::collections::BTreeSet<_>>();
        for receipt in &self.receipts {
            let allocation = &receipt.allocation;
            if allocation.location.domain != Domain::Device { continue; }
            // A physical root is not an upload destination. Its offset-zero
            // alias can share its address, but owns the logical staging key.
            if allocation.view.is_none() && backings.contains(allocation.id.as_str()) { continue; }
            let Device::Coordinator(rank) = allocation.location.device else {
                anyhow::bail!("ordered upload destination is not coordinator-owned");
            };
            let device_id = self.devices.physical_device(rank)?;
            if !observed.insert((device_id, receipt.address)) { continue; }
            let dst = CuteafdDeviceBuffer { ptr: receipt.address as *mut std::ffi::c_void,
                bytes: allocation.bytes.try_into()?, device_id, flags: 0 };
            if let Some(buffer) = library.ordered_h2d_staging_buffer(dst, stream)? {
                let name = allocation.id.strip_prefix(&format!("{}.", self.prefix))
                    .context("ordered upload destination is outside inventory prefix")?;
                uploads.push((format!("{name}.upload"), buffer));
            }
        }
        for (name, buffer) in uploads {
            self.pinned(&name, buffer)?;
        }
        Ok(())
    }
    pub(crate) fn pinned(&mut self, name: &str, buffer: CuteafdHostBuffer) -> Result<()> {
        self.add(name, Location::PINNED, buffer.ptr as u64, buffer.bytes as u64)
    }
    pub(crate) fn host_backing(&mut self, name: &str, address: u64, bytes: u64) -> Result<()> {
        if bytes != 0 { self.add(name, Location::HOST, address, bytes)?; }
        Ok(())
    }
    pub(crate) fn host<T>(&mut self, name: &str, values: &Vec<T>) -> Result<()> {
        let bytes = values.capacity().checked_mul(std::mem::size_of::<T>()).context("host capacity overflow")?;
        if bytes != 0 { self.add(name, Location::HOST, values.as_ptr() as u64, bytes as u64)?; }
        Ok(())
    }
    pub(crate) fn view(&mut self, name: &str, buffer: CuteafdDeviceBuffer, backing: &str,
        owner: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(buffer.device_id == owner.device_id, "view {name} changed device");
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("view before owner")?;
        ensure!(offset.checked_add(buffer.bytes as u64).is_some_and(|end| end <= owner.bytes as u64),
            "view {name} exceeds actual backing");
        self.device(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View {
            backing: format!("{}.{backing}", self.prefix), offset,
        });
        Ok(())
    }
    pub(crate) fn backed_device(&mut self, name: &str, buffer: CuteafdDeviceBuffer,
        owner: CuteafdDeviceBuffer) -> Result<()> {
        let location = self.devices.location(owner.device_id)?;
        let backing = self.receipts.iter().find(|receipt| receipt.allocation.view.is_none()
            && receipt.allocation.location == location && receipt.address == owner.ptr as u64
            && receipt.allocation.bytes == owner.bytes as u64).map(|r| r.allocation.id.clone());
        let backing = match backing {
            Some(id) => id,
            None => {
                let id = format!("{name}.backing");
                self.device(&id, owner)?;
                format!("{}.{id}", self.prefix)
            }
        };
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("region before owner")?;
        ensure!(buffer.device_id == owner.device_id
            && offset.checked_add(buffer.bytes as u64).is_some_and(|end| end <= owner.bytes as u64),
            "region {name} outside actual owner");
        self.device(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View { backing, offset });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn receipt(id: &str, location: Location, address: u64, bytes: u64, lifetime: Lifetime) -> Receipt {
        Receipt { allocation: Allocation::buffer(id, location, bytes, lifetime, Category::Workspace, "fixture"), address }
    }

    #[test]
    fn drafter_arena_plan_and_runtime_receipts_share_category_and_one_backing() -> Result<()> {
        use cuteafd_core::memory_layout::{ArenaTag, MemoryKind};
        use cuteafd_loader::placement::planner::{weight_arena_components, ComponentKind};
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 20)?;
        let device = super::super::device::Device { library: &library, id: 0 };
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("glm/draft/0",
            [("tensor".into(), 1024)])?;
        let components = weight_arena_components(&layout, Location::coordinator(0),
            Lifetime::RESIDENT, 1, MemoryKind::DrafterWeights)?;
        let ComponentKind::Allocation { template: planned, .. } = &components[0].kind else { panic!("arena root") };
        let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device, layout,
            ArenaTag::drafter_weights("fixture/drafter"))?;
        let tensor = arena.view("tensor", 1024)?;
        let mut inventory = Inventory::new(CoordinatorDevices::new(&[0])?, "fixture",
            Lifetime::RESIDENT, MemoryKind::DrafterWeights.category(), "draft owners");
        inventory.weight_view("tensor", tensor.buffer, tensor.weight_backing().context("arena backing")?)?;
        let roots = inventory.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).collect::<Vec<_>>();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].allocation, *planned);
        assert_eq!(roots[0].allocation.category, Category::Drafter);
        assert_eq!(tensor.tag().context("backing tag")?.kind, MemoryKind::DrafterWeights);
        assert_eq!(inventory.receipts[1].allocation.view.as_ref().unwrap().backing, planned.id);
        Ok(())
    }

    #[test]
    fn census_combines_independent_inventories_without_double_charge() -> Result<()> {
        let mut census = Census::default();
        let late = Lifetime { first: cuteafd_loader::placement::planner::Phase::Warmup,
            last: cuteafd_loader::placement::planner::Phase::Growth };
        census.owners(vec![receipt("prefix.pool", Location::coordinator(0), 0x1000, 1024, late),
            receipt("host.capacity", Location::HOST, 0x8000, 128, late)]);
        census.owners(vec![receipt("target.pool", Location::coordinator(0), 0x1000, 1024, Lifetime::TARGET)]);
        let mut view = receipt("target.slot", Location::coordinator(0), 0x1100, 128, late);
        view.allocation.view = Some(View { backing: "target.pool".into(), offset: 256 });
        census.owners(vec![view]);
        census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
        assert_eq!(census.receipts.iter().filter(|r| r.allocation.view.is_none())
            .map(|r| r.allocation.bytes).sum::<u64>(), 1024);
        assert_eq!(census.receipts[0].allocation.lifetime, Lifetime::TARGET);
        assert_eq!(census.receipts[1].allocation.view.as_ref().unwrap().backing, "prefix.pool");
        assert_eq!(census.receipts[2].allocation.view.as_ref().unwrap(), &View { backing: "prefix.pool".into(), offset: 256 });
        assert_eq!(census.heap_diagnostics.len(), 1);
        let before = census.receipts.clone();
        census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
        assert!(census.receipts.iter().zip(before).all(|(a, b)| a.address == b.address && a.allocation == b.allocation));
        Ok(())
    }

    #[test]
    fn census_keeps_mapped_aliases_and_unavailable_owners_separate() -> Result<()> {
        let physical = [1, 0];
        let devices = CoordinatorDevices::new(&physical)?;
        let mut transport = cuteafd_transport::TransportAllocationCensus::unavailable("rank.1.driver", "measured delta missing");
        transport.host.push(cuteafd_transport::TransportHostAllocation { owner: "rank.0.receive".into(),
            address: 0x1000, bytes: 768, flags: cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_PINNED,
            mapped: Some(cuteafd_transport::TransportDeviceAlias { address: 0x9000, device: 0 }), audit_owner: None });
        let mut census = Census::default();
        census.transport("link", transport, devices, Lifetime::TARGET)?;
        census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
        assert_eq!(census.receipts.len(), 1);
        assert_eq!((census.receipts[0].address, census.receipts[0].allocation.bytes), (0x1000, 768));
        assert_eq!(census.receipts[0].allocation.location, Location::PINNED);
        assert_eq!(census.aliases[0].backing, "link.rank.0.receive");
        assert_eq!(census.aliases[0].location, Location::coordinator(1));
        assert_eq!(census.aliases[0].address, 0x9000);
        assert_eq!(census.unclassified[0].owner, "link.rank.1.driver");
        assert_eq!(census.unclassified[0].reason, "measured delta missing");
        Ok(())
    }

    #[test]
    fn census_rejects_overlap_and_unproved_backing() -> Result<()> {
        for receipts in [
            vec![receipt("a", Location::PINNED, 0x1000, 1024, Lifetime::TARGET),
                receipt("b", Location::PINNED, 0x1100, 128, Lifetime::TARGET)],
            vec![receipt("a", Location::PINNED, 0x1000, 1024, Lifetime::TARGET),
                receipt("a", Location::PINNED, 0x2000, 1024, Lifetime::TARGET)],
        ] {
            let mut census = Census::default();
            census.owners(receipts);
            assert!(census.normalize(PhaseOrder::DRAFT_THEN_TARGET).is_err());
        }
        let mut view = receipt("view", Location::coordinator(0), 0x1100, 128, Lifetime::TARGET);
        view.allocation.view = Some(View { backing: "pinned".into(), offset: 256 });
        let mut census = Census::default();
        census.owners(vec![receipt("pinned", Location::PINNED, 0x1000, 1024, Lifetime::TARGET), view]);
        assert!(census.normalize(PhaseOrder::DRAFT_THEN_TARGET).is_err());
        Ok(())
    }

    #[test]
    fn mapped_pinned_census_uses_whole_backing_without_native_work() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let owner = crate::shared::memory::HostAllocation::new(&library, 1024, crate::shared::memory::TEST)?;
        library.cuda_set_device(1)?;
        let devices = [1, 0];
        let devices = CoordinatorDevices::new(&devices)?;
        let mut inventory = Inventory::new(devices, "fixture", Lifetime::TARGET, Category::Transport, "fixture");
        let slot = CuteafdHostBuffer { ptr: (owner.buffer.ptr as usize + 128) as *mut _, bytes: 256, ..owner.buffer };
        inventory.backed_pinned("slot", slot, owner.buffer)?;
        let mut census = Census::default();
        census.owners(inventory.receipts);
        let before = fixture.events()?;
        census.mapped_pinned(&library, devices, "fixture.mapped".into(), owner.buffer)?;
        census.normalize(PhaseOrder::DRAFT_THEN_TARGET)?;
        assert_eq!(census.aliases.len(), 1);
        let alias = &census.aliases[0];
        assert_eq!((alias.address, alias.bytes, alias.location),
            (owner.buffer.ptr as u64, 1024, Location::coordinator(0)));
        assert_eq!(alias.backing, "fixture.slot.backing");
        assert_eq!(census.receipts.iter().filter(|r| r.allocation.view.is_none())
            .map(|r| r.allocation.bytes).sum::<u64>(), 1024);
        assert!(census.mapped_pinned(&library, devices, "bad".into(), slot).is_err());
        assert_eq!(fixture.events()?, before);
        assert_eq!(library.cuda_get_device()?, 1);
        Ok(())
    }

    #[test]
    fn ordered_upload_census_is_absent_until_reserved_and_never_resizes() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let mut buffer = library.alloc_device_buffer(1024, crate::shared::memory::TEST)?;
        let physical = [1, 0];
        let devices = CoordinatorDevices::new(&physical)?;
        let stream = 17;
        let inventory = || -> Result<Inventory<'_>> {
            let mut inventory = Inventory::new(devices, "fixture", Lifetime::TARGET, Category::Drafter, "fixture");
            inventory.device("metadata", buffer)?;
            inventory.device("same_destination", buffer)?;
            Ok(inventory)
        };
        let before = fixture.events()?;
        let mut absent = inventory()?;
        absent.ordered_uploads(&library, stream as *mut _)?;
        assert_eq!(absent.receipts.len(), 2);
        assert_eq!(fixture.events()?, before);
        fixture.reserve_ordered_upload(&library, buffer, stream)?;
        let actual = library.ordered_h2d_staging_buffer(buffer, stream as *mut _)?.unwrap();
        library.cuda_set_device(1)?;
        let before = fixture.events()?;
        let mut present = inventory()?;
        present.ordered_uploads(&library, stream as *mut _)?;
        assert_eq!(present.receipts.len(), 3);
        let receipt = &present.receipts[2];
        assert_eq!((receipt.address, receipt.allocation.bytes, receipt.allocation.location),
            (actual.ptr as u64, actual.bytes as u64, Location::PINNED));
        assert_eq!(fixture.events()?, before, "no reserve, resize, synchronize or copy during census");
        assert_eq!(library.cuda_get_device()?, 1);
        fixture.finish_ordered_uploads(&library)?;
        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn views_derive_real_offsets_and_host_capacity_is_not_length() -> Result<()> {
        let devices = [1, 0];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "fixture",
            Lifetime::TARGET, Category::Workspace, "fixture");
        let owner = CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut(0x1000), bytes: 1024, device_id: 0, flags: 0 };
        inventory.device("arena", owner)?;
        inventory.view("slice", CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut(0x1100), bytes: 128,
            ..owner }, "arena", owner)?;
        assert_eq!(inventory.receipts[1].allocation.view, Some(View { backing: "fixture.arena".into(), offset: 256 }));
        assert!(inventory.view("past", CuteafdDeviceBuffer { bytes: 2048, ..owner }, "arena", owner).is_err());
        let mut host = Vec::<u32>::with_capacity(11);
        host.push(3);
        inventory.host("host", &host)?;
        assert_eq!(inventory.receipts[2].allocation.bytes, (host.capacity() * 4) as u64);
        Ok(())
    }
}
