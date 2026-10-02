//! Pure startup capacity policy. Families describe physical allocation costs;
//! the daemon supplies memory measured before its allocations. Every device
//! constrains the same logical pool, whether KV is replicated or partitioned.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityPolicy {
    pub concurrency: u32,
    pub requested_full_contexts: u32,
    pub gpu_occupancy_percent: u32,
    /// None chooses the checkpoint length bounded by verified kernel support.
    /// Explicit requests beyond either capability are rejected, never clipped.
    pub max_context_tokens: Option<u64>,
    /// Explicit benchmark overrides retain their requested pool size, rounded
    /// up to the family's allocation quantum and checked before allocation.
    pub pool_tokens: Option<u64>,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            concurrency: 16,
            requested_full_contexts: 8,
            gpu_occupancy_percent: 97,
            max_context_tokens: None,
            pool_tokens: None,
        }
    }
}

impl CapacityPolicy {
    pub fn state_slots(self) -> Result<u32, CapacityError> {
        let slots = u64::from(self.concurrency)
            .checked_mul(5)
            .ok_or(CapacityError::Overflow("state slots"))?
            .div_ceil(4);
        u32::try_from(slots).map_err(|_| CapacityError::Overflow("state slots"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextLimits {
    pub checkpoint_max_tokens: u64,
    /// None only for a family whose context extent is dynamic. Indexed
    /// families must supply the validated compiled manifest bound.
    pub compiled_index_max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemory {
    pub device: u32,
    pub total_bytes: u64,
    /// Read before loading this engine's weights, modules or workspaces.
    pub baseline_free_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReservation {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCosts {
    pub device: u32,
    /// Include actual loaded representations (such as BF16 + FP8 copies),
    /// all workspace shapes/lanes, capture storage, transport, drafts, active
    /// state and retained marks. Header weight bytes alone are insufficient.
    pub reservations: Vec<MemoryReservation>,
    /// This device's bytes per logical pool allocation unit, including all
    /// persistent KV/index records and pool-sized tables/workspace metadata.
    /// A replica has the full cost on each device; partitioned heads use
    /// each device's actual share. Zero means this device does not own KV.
    pub pool_unit_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityProfile {
    pub context: ContextLimits,
    pub pool_unit_rows: u64,
    pub devices: Vec<DeviceCosts>,
    /// A resolved inactive host snapshot quota; reported separately and
    /// never included in the active GPU token capacity calculation.
    pub host_prefix_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDeviceCapacity {
    pub device: u32,
    pub total_bytes: u64,
    pub non_engine_bytes: u64,
    pub engine_budget_bytes: u64,
    pub reservations: Vec<MemoryReservation>,
    pub reserved_bytes: u64,
    pub pool_bytes: u64,
    pub unused_budget_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedCapacity {
    pub concurrency: u32,
    pub state_slots: u32,
    pub checkpoint_max_context_tokens: u64,
    pub compiled_index_max_context_tokens: Option<u64>,
    pub effective_max_context_tokens: u64,
    pub requested_kv_floor_tokens: u64,
    pub feasible_gpu_kv_tokens: u64,
    pub allocated_gpu_kv_tokens: u64,
    pub requested_floor_fits_hardware: bool,
    pub requested_floor_allocated: bool,
    pub requested_floor_shortfall_tokens: u64,
    /// Full effective-context sequences the pool can hold, capped at Cmax.
    /// Short requests can still reach Cmax when this count is lower.
    pub active_max_context_sequences: u32,
    pub host_prefix_bytes: u64,
    pub explicit_pool_override: bool,
    pub devices: Vec<ResolvedDeviceCapacity>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CapacityError {
    #[error("invalid capacity input: {0}")]
    Invalid(&'static str),
    #[error("capacity arithmetic overflows: {0}")]
    Overflow(&'static str),
    #[error("requested context {requested} exceeds checkpoint maximum {checkpoint}")]
    CheckpointContextExceeded { requested: u64, checkpoint: u64 },
    #[error("requested context {requested} exceeds compiled index extent {compiled}; export matching wider programs")]
    CompiledContextExceeded { requested: u64, compiled: u64 },
    #[error("GPU {device} fixed reservations need {reserved} bytes but its engine budget is {budget}; reduce placement/workspaces or choose a smaller compatible checkpoint")]
    ReservationsExceeded {
        device: u32,
        reserved: u64,
        budget: u64,
    },
    #[error(
        "explicit aligned pool {requested} tokens exceeds feasible GPU pool {feasible} tokens"
    )]
    PoolExceeded { requested: u64, feasible: u64 },
}

/// Resolve once before allocation, then hand this exact result to startup.
/// A missed eight-context floor is a reported physical limit, not fabricated
/// capacity; host prefix copies do not change it.
pub fn resolve_capacity(
    policy: CapacityPolicy,
    profile: &CapacityProfile,
    hardware: &[DeviceMemory],
) -> Result<ResolvedCapacity, CapacityError> {
    if policy.concurrency == 0 || policy.requested_full_contexts == 0 {
        return Err(CapacityError::Invalid(
            "concurrency and full-context target must be positive",
        ));
    }
    if !(1..=100).contains(&policy.gpu_occupancy_percent) {
        return Err(CapacityError::Invalid(
            "GPU occupancy percent must be in 1..=100",
        ));
    }
    if profile.context.checkpoint_max_tokens == 0
        || profile.context.compiled_index_max_tokens == Some(0)
        || profile.pool_unit_rows == 0
        || profile.devices.is_empty()
    {
        return Err(CapacityError::Invalid(
            "positive checkpoint/kernel context and pool quantum required",
        ));
    }
    let context = policy.max_context_tokens.unwrap_or_else(|| {
        profile
            .context
            .compiled_index_max_tokens
            .map_or(profile.context.checkpoint_max_tokens, |limit| {
                limit.min(profile.context.checkpoint_max_tokens)
            })
    });
    if context == 0 {
        return Err(CapacityError::Invalid("requested context must be positive"));
    }
    if context > profile.context.checkpoint_max_tokens {
        return Err(CapacityError::CheckpointContextExceeded {
            requested: context,
            checkpoint: profile.context.checkpoint_max_tokens,
        });
    }
    if let Some(compiled) = profile.context.compiled_index_max_tokens {
        if context > compiled {
            return Err(CapacityError::CompiledContextExceeded {
                requested: context,
                compiled,
            });
        }
    }
    let floor = profile
        .context
        .checkpoint_max_tokens
        .checked_mul(u64::from(policy.requested_full_contexts))
        .ok_or(CapacityError::Overflow("requested KV floor"))?;
    let mut memory_by_device = BTreeMap::new();
    for &memory in hardware {
        if memory.total_bytes == 0
            || memory.baseline_free_bytes > memory.total_bytes
            || memory_by_device.insert(memory.device, memory).is_some()
        {
            return Err(CapacityError::Invalid(
                "invalid or duplicate physical GPU memory sample",
            ));
        }
    }
    if memory_by_device.len() != profile.devices.len() {
        return Err(CapacityError::Invalid(
            "one cost profile per selected physical GPU required",
        ));
    }
    let mut feasible_units = u64::MAX;
    let mut has_kv = false;
    let mut devices = Vec::with_capacity(profile.devices.len());
    for costs in &profile.devices {
        let memory = memory_by_device
            .remove(&costs.device)
            .ok_or(CapacityError::Invalid(
                "missing or duplicate physical GPU cost profile",
            ))?;
        let non_engine = memory.total_bytes - memory.baseline_free_bytes;
        // The ceiling is a fraction of TOTAL, less pre-existing usage. A
        // fraction of free memory would also scale pre-existing usage.
        let ceiling = (u128::from(memory.total_bytes) * u128::from(policy.gpu_occupancy_percent)
            / 100) as u64;
        let budget = ceiling.saturating_sub(non_engine);
        let reserved = costs
            .reservations
            .iter()
            .try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .ok_or(CapacityError::Overflow("fixed device reservations"))?;
        if reserved > budget {
            return Err(CapacityError::ReservationsExceeded {
                device: costs.device,
                reserved,
                budget,
            });
        }
        if costs.pool_unit_bytes > 0 {
            has_kv = true;
            feasible_units = feasible_units.min((budget - reserved) / costs.pool_unit_bytes);
        }
        devices.push(ResolvedDeviceCapacity {
            device: costs.device,
            total_bytes: memory.total_bytes,
            non_engine_bytes: non_engine,
            engine_budget_bytes: budget,
            reservations: costs.reservations.clone(),
            reserved_bytes: reserved,
            pool_bytes: 0,
            unused_budget_bytes: budget - reserved,
        });
    }
    if !has_kv {
        return Err(CapacityError::Invalid(
            "at least one physical GPU must own the logical KV pool",
        ));
    }
    let feasible = feasible_units
        .checked_mul(profile.pool_unit_rows)
        .ok_or(CapacityError::Overflow("feasible KV tokens"))?;
    let units = match policy.pool_tokens {
        Some(0) => {
            return Err(CapacityError::Invalid(
                "explicit pool tokens must be positive",
            ))
        }
        Some(tokens) => tokens.div_ceil(profile.pool_unit_rows),
        None => feasible_units,
    };
    let allocated = units
        .checked_mul(profile.pool_unit_rows)
        .ok_or(CapacityError::Overflow("aligned pool tokens"))?;
    if allocated > feasible {
        return Err(CapacityError::PoolExceeded {
            requested: allocated,
            feasible,
        });
    }
    for (device, costs) in devices.iter_mut().zip(&profile.devices) {
        device.pool_bytes = units
            .checked_mul(costs.pool_unit_bytes)
            .ok_or(CapacityError::Overflow("device pool bytes"))?;
        device.unused_budget_bytes -= device.pool_bytes;
    }
    let full_sequence_units = context.div_ceil(profile.pool_unit_rows);
    let active = (units / full_sequence_units).min(u64::from(policy.concurrency)) as u32;
    Ok(ResolvedCapacity {
        concurrency: policy.concurrency,
        state_slots: policy.state_slots()?,
        checkpoint_max_context_tokens: profile.context.checkpoint_max_tokens,
        compiled_index_max_context_tokens: profile.context.compiled_index_max_tokens,
        effective_max_context_tokens: context,
        requested_kv_floor_tokens: floor,
        feasible_gpu_kv_tokens: feasible,
        allocated_gpu_kv_tokens: allocated,
        requested_floor_fits_hardware: feasible >= floor,
        requested_floor_allocated: allocated >= floor,
        requested_floor_shortfall_tokens: floor.saturating_sub(allocated),
        active_max_context_sequences: active,
        host_prefix_bytes: profile.host_prefix_bytes,
        explicit_pool_override: policy.pool_tokens.is_some(),
        devices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    fn device(id: u32, kv_unit_bytes: u64, reserved: u64) -> DeviceCosts {
        DeviceCosts {
            device: id,
            pool_unit_bytes: kv_unit_bytes,
            reservations: vec![MemoryReservation {
                name: "weights + all workspace shapes + marks".into(),
                bytes: reserved,
            }],
        }
    }
    fn hardware(id: u32, occupied: u64) -> DeviceMemory {
        DeviceMemory {
            device: id,
            total_bytes: 96 * GIB,
            baseline_free_bytes: 96 * GIB - occupied,
        }
    }
    fn profile(devices: Vec<DeviceCosts>) -> CapacityProfile {
        CapacityProfile {
            context: ContextLimits {
                checkpoint_max_tokens: 1024,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: 64,
            devices,
            host_prefix_bytes: 0,
        }
    }

    #[test]
    fn total_gpu_ceiling_subtracts_existing_usage_before_joint_reservations() {
        let p = profile(vec![device(0, GIB, 20 * GIB)]);
        let result =
            resolve_capacity(CapacityPolicy::default(), &p, &[hardware(0, 10 * GIB)]).unwrap();
        let gpu = &result.devices[0];
        assert_eq!(gpu.engine_budget_bytes, 96 * GIB * 97 / 100 - 10 * GIB);
        assert_eq!(result.allocated_gpu_kv_tokens, 63 * 64);
        assert_eq!((result.concurrency, result.state_slots), (16, 20));
        assert!(gpu.non_engine_bytes + gpu.reserved_bytes + gpu.pool_bytes <= 96 * GIB * 97 / 100);
        assert!(gpu.unused_budget_bytes < GIB);
    }

    #[test]
    fn replicated_kv_never_adds_gpu_capacities_and_partitioned_heads_use_actual_shares() {
        let one = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB)]),
            &[hardware(0, 0)],
        )
        .unwrap();
        let copies = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB), device(1, GIB, 20 * GIB)]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert_eq!(copies.allocated_gpu_kv_tokens, one.allocated_gpu_kv_tokens);
        let split = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![
                device(0, GIB / 2, 20 * GIB),
                device(1, GIB / 2, 20 * GIB),
            ]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert!(split.allocated_gpu_kv_tokens >= 2 * one.allocated_gpu_kv_tokens);
        let bottleneck = resolve_capacity(
            CapacityPolicy::default(),
            &profile(vec![device(0, GIB, 20 * GIB), device(1, GIB, 40 * GIB)]),
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert!(bottleneck.allocated_gpu_kv_tokens < copies.allocated_gpu_kv_tokens);
    }

    #[test]
    fn pro_eight_context_target_remains_visible_when_hardware_cannot_fit_it() {
        let mut p = profile(vec![
            device(0, 64 * 14400, 10 * GIB),
            device(1, 64 * 14400, 10 * GIB),
        ]);
        p.context.checkpoint_max_tokens = 1 << 20;
        let result = resolve_capacity(
            CapacityPolicy::default(),
            &p,
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert_eq!(result.requested_kv_floor_tokens, 8 << 20);
        assert!(!result.requested_floor_fits_hardware && !result.requested_floor_allocated);
        assert_eq!(
            result.requested_floor_shortfall_tokens,
            (8 << 20) - result.allocated_gpu_kv_tokens
        );
        assert!(result.active_max_context_sequences < 8);
        assert_eq!(result.effective_max_context_tokens, 1 << 20);
        p.host_prefix_bytes = 500 * GIB;
        let host = resolve_capacity(
            CapacityPolicy::default(),
            &p,
            &[hardware(0, 0), hardware(1, 0)],
        )
        .unwrap();
        assert_eq!(host.allocated_gpu_kv_tokens, result.allocated_gpu_kv_tokens);
        assert_eq!(
            host.active_max_context_sequences,
            result.active_max_context_sequences
        );
    }

    #[test]
    fn checkpoint_and_compiled_context_are_separate_and_explicit_limits_never_clip() {
        let mut p = profile(vec![device(0, 64 * 11803, 10 * GIB)]);
        p.context = ContextLimits {
            checkpoint_max_tokens: 1 << 20,
            compiled_index_max_tokens: Some(131072),
        };
        let result = resolve_capacity(CapacityPolicy::default(), &p, &[hardware(0, 0)]).unwrap();
        assert_eq!(result.effective_max_context_tokens, 131072);
        assert_eq!(result.requested_kv_floor_tokens, 8 << 20);
        let policy = CapacityPolicy {
            max_context_tokens: Some(131073),
            ..Default::default()
        };
        assert_eq!(
            resolve_capacity(policy, &p, &[hardware(0, 0)]).unwrap_err(),
            CapacityError::CompiledContextExceeded {
                requested: 131073,
                compiled: 131072
            }
        );
    }

    #[test]
    fn explicit_pool_override_rounds_to_real_units_and_is_admitted_before_allocation() {
        let p = profile(vec![device(0, GIB, 20 * GIB)]);
        let result = resolve_capacity(
            CapacityPolicy {
                pool_tokens: Some(65),
                ..Default::default()
            },
            &p,
            &[hardware(0, 0)],
        )
        .unwrap();
        assert_eq!(result.allocated_gpu_kv_tokens, 128);
        assert!(result.explicit_pool_override);
        assert!(!result.requested_floor_allocated);
        assert!(matches!(
            resolve_capacity(
                CapacityPolicy {
                    pool_tokens: Some(10000),
                    ..Default::default()
                },
                &p,
                &[hardware(0, 0)]
            ),
            Err(CapacityError::PoolExceeded { .. })
        ));
        assert!(matches!(
            resolve_capacity(
                CapacityPolicy::default(),
                &profile(vec![device(0, GIB, 95 * GIB)]),
                &[hardware(0, 0)]
            ),
            Err(CapacityError::ReservationsExceeded { device: 0, .. })
        ));
    }

    #[test]
    fn invalid_inputs_and_capacity_overflow_fail_typed() {
        let p = profile(vec![device(0, GIB, 0)]);
        assert!(resolve_capacity(
            CapacityPolicy::default(),
            &p,
            &[hardware(0, 0), hardware(0, 0)]
        )
        .is_err());
        assert!(resolve_capacity(
            CapacityPolicy {
                concurrency: 0,
                ..Default::default()
            },
            &p,
            &[hardware(0, 0)]
        )
        .is_err());
        let mut huge = p.clone();
        huge.context.checkpoint_max_tokens = u64::MAX;
        assert_eq!(
            resolve_capacity(CapacityPolicy::default(), &huge, &[hardware(0, 0)]),
            Err(CapacityError::Overflow("requested KV floor"))
        );
    }
}
