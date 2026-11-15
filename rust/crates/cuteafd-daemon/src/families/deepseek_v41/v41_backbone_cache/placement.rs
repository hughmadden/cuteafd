//! Cache ownership uses the same validated map as admission and native weights.
#[cfg(test)]
use anyhow::Result;
pub(crate) use cuteafd_loader::placement::families::deepseek_v41::CachePlacement;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tail14_planner_cache_bytes_equal_runtime_storage_bytes() -> Result<()> {
        use cuteafd_loader::placement::families::deepseek_v41::EncoderPlacement;
        use super::super::BackboneCache;
        let config = serde_json::from_str(include_str!(
            "../../../../../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"))?;
        for option in [EncoderPlacement::Ced, EncoderPlacement::Tail14] {
            let map = option.resolve(2)?;
            for concurrency in [1, 16] {
                for retained in [0, 128] {
                    let groups = 4096 + concurrency + 2 * retained;
                    let bytes = BackboneCache::distributed_device_bytes(map, concurrency, [groups, groups, groups, 2 * groups])?;
                    let plan = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes_with_placement(
                        &config, 2, concurrency as u64, groups as u64, false, option)?;
                    assert_eq!(bytes.map(|bytes| bytes as u64).as_slice(), plan);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn execution_plan_binds_single_and_ced_ownership() -> Result<()> {
        use cuteafd_loader::placement::families::deepseek_v41::admission::{
            live_budgets, production_request, ProductionInput,
        };
        use cuteafd_loader::placement::planner::{Operation, PrefillShape};
        use cuteafd_loader::placement::*;
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        ))?;
        for ranks in [1, 2] {
            let prefill = PrefillShape { rows: 128, waves: 1 };
            let placement = ProductionInput {
        encoder_placement: Default::default(),
                coordinator_arch: cuteafd_core::GpuArch::Sm120,
            blas: Vec::new(),
            expert_usage: false,
                config: &config, gpus: ranks, context: 4096, maximum_pool_tokens: 8192,
                prefill, concurrency: 1, retained_turns: 0, replicated: false, spark_ranks: 4,
                onboard_minimum: 0, onboard_counts: None,
                experts: vec![ExpertCost { whole: Bytes2::default(), half: [Bytes2::default(); 2],
                    tp2: ranks == 2, whole_supported: true, spark_ok: true }; 40],
                workspace: 0, owners: Vec::new(), weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(),
                draft: None, encoders: Vec::new(), media_cache_bytes: 0,
                resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
                operation: Operation::Serving,
                budgets: live_budgets(&vec![(1 << 40, 1 << 40, 1 << 39); ranks],
                    &vec![cuteafd_loader::plan::layout::DEFAULT_SPARK_DEVICE_BYTES; 4])?,
            }.prepare()?.admit(production_request(4096, Some(4096), Onboard::Layers(0), prefill,
                Operation::Serving, u64::MAX, Default::default()))?.placement;
            let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(ranks);
            let plan = LayerExecutionPlan::new(&placement, spec)?;
            let map = CachePlacement::from_execution_plan(&plan)?;
            for layer in 0..40 {
                assert_eq!(map.attention(layer)?, usize::from(ranks == 2 && layer >= 20));
            }
            assert_eq!(plan.hops().len(), ranks - 1);
            let mut wrong = placement.clone();
            wrong.residual[1] = ResidualHome::Replicated;
            assert!(LayerExecutionPlan::new(&wrong, spec).is_err());
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn distributed_cache_allocations_and_leases_follow_placement() -> Result<()> {
        cache_allocations(false)
    }
    #[test]
    #[ignore = "requires SM peer copy and two CUDA GPUs"]
    fn replicated_cache_allocations_preserve_capacity_and_reset_both_devices() -> Result<()> {
        cache_allocations(true)
    }
    fn cache_allocations(replicated:bool) -> Result<()> {
        use super::super::BackboneCache;
        use cuteafd_ffi::NativeLibrary;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        for boundary in [20,14] {
            let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= boundary)))?;
            let pages = [2,2,2,4];
            let bytes = BackboneCache::distributed_device_bytes(placement,2,pages)?;
            assert_eq!(bytes.iter().sum::<usize>(),BackboneCache::device_bytes(2,pages)?);
            assert!(BackboneCache::new_distributed(&lib,placement,2,pages,[bytes[0]-1,bytes[1]]).is_err());
            let mut bank = if replicated {
                let replicated_bytes=BackboneCache::replicated_device_bytes(placement,2,pages)?;
                assert!(BackboneCache::new_replicated(&lib,placement,2,pages,
                    [replicated_bytes[0]-1,replicated_bytes[1]]).is_err());
                BackboneCache::new_replicated(&lib,placement,2,pages,replicated_bytes)?
            } else { BackboneCache::new_distributed(&lib,placement,2,pages,bytes)? };
            for cycle in 0..2 {
                let lease = bank.begin_request(0,42+cycle)?;
                assert_eq!(bank.committed_end(lease)?,0);
                let request = bank.request(lease)?;
                for layer in 0..40 {
                    let view = bank.windows[layer].view(request.windows[layer])?;
                    let expected = placement.attention(layer)? as i32;
                    assert_eq!(bank.attention_device(layer)?.id,expected);
                    assert_eq!(view.device_end.device_id,expected);
                    let mut end = [255;8];
                    bank.attention_device(layer)?.run(|| lib.copy_d2h(&mut end,view.device_end))?;
                    assert_eq!(end,[0;8]);
                    if replicated {
                        let replica=bank.windows[layer].replica().unwrap();
                        let peer=unsafe { replica.view(&bank.windows[layer],request.windows[layer])? };
                        assert_eq!(peer.device_end.device_id,1-expected);
                        replica.device().run(||lib.copy_d2h(&mut end,peer.device_end))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                for (index,gpu) in placement.sources().into_iter().enumerate() {
                    let kv = bank.sources[index].kv_cache(request.sources[index])?;
                    let keys = bank.sources[index].index_cache(request.sources[index])?;
                    for buffer in [kv.values,kv.scales,kv.device_pages,kv.device_rows,keys.packed,keys.scales] {
                        assert_eq!(buffer.device_id,gpu as i32);
                    }
                    assert_eq!(kv.rows,0);
                    assert_eq!(bank.sources[index].source_cache().capacity,pages[index]*256);
                    if replicated {
                        let replica=bank.sources[index].replica().unwrap();
                        let peer=unsafe { replica.view(bank.sources[index].source_cache(),0,0)? };
                        assert_eq!(peer.values.device_id,1-gpu as i32);
                        let mut end=[255;8];
                        replica.device().run(||lib.copy_d2h(&mut end,peer.device_rows))?;
                        assert_eq!(end,[0;8]);
                    }
                }
                bank.release(&[lease])?;
                assert!(bank.committed_end(lease).is_err());
                assert_eq!(lib.cuda_get_device()?,0);
            }
            drop(bank);
            assert_eq!(lib.cuda_get_device()?,0);
        }
        Ok(())
    }
    #[test]
    fn placement_keeps_sources_with_consumers_and_allows_rebalancing() -> Result<()> {
        let initial = CachePlacement::encoder_decoder();
        assert_eq!(initial.sources(),[0,0,0,1]);
        let mut rebalanced = initial.layers();
        rebalanced[14..20].fill(1);
        assert_eq!(CachePlacement::new(rebalanced)?.sources(),[0,0,1,1]);
        rebalanced[19] = 0;
        assert!(CachePlacement::new(rebalanced).is_err());
        rebalanced[19] = 1;
        rebalanced[0] = 2;
        assert!(CachePlacement::new(rebalanced).is_err());
        assert!(initial.attention(40).is_err());
        Ok(())
    }
}
