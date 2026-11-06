//! Native projection export selection shared by CPU placement and preweight loading.
use super::Refusal;
use crate::{families::glm5_flash::core::NativeLayout, placement::inventory::ProgramsManifest};
use cuteafd_ffi::draft_projection::ProjectionOutput;
use std::collections::BTreeSet;

pub struct NativeProgram<'a> {
    pub layout: &'a NativeLayout,
    /// A split is not necessarily a reduction partial; the caller owns this fact.
    pub output: ProjectionOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeResolution { Assumed, Exact }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePrograms {
    pub programs: Vec<String>,
    pub fp8_scratch_bytes: u64,
    pub resolution: NativeResolution,
}

impl NativePrograms {
    pub fn resolve(requests: &[NativeProgram<'_>], rows: usize, manifest: Option<&ProgramsManifest>)
        -> Result<Self, Refusal> {
        let mut programs = BTreeSet::new();
        let mut fp8_scratch_bytes = 0;
        for request in requests {
            let [n, k] = request.layout.shape;
            let fp8 = request.layout.fp8;
            if rows == 0 || rows > i32::MAX as usize || k == 0 || n == 0
                || k % if fp8 { 128 } else { 8 } != 0
                || n % if fp8 { 128 } else { 8 } != 0 || (fp8 && rows > 4096) {
                return Err(Refusal::InvalidFact(format!(
                    "native projection {}: invalid shape [{n}, {k}] or row ceiling {rows}", request.layout.stem)));
            }
            let name = if fp8 {
                let stage = if rows <= 64 { 64 } else { 4096 };
                let suffix = if request.output == ProjectionOutput::F32 { "_f32" } else { "" };
                format!("native_attention_n{n}_k{k}_m{stage}{suffix}")
            } else { request.output.program_name(k, n) };
            if let Some(manifest) = manifest {
                let refusal = |detail| Refusal::MissingExport { export: name.clone(),
                    geometry: format!("native [{n}, {k}], {:?}", request.output), context: rows as u64, detail };
                let entry = manifest.get(&name).ok_or_else(|| refusal("absent from PROGRAMS.json".into()))?;
                if fp8 {
                    cuteafd_ffi::native_attention::validate_spec(&entry.spec, k, n, rows, request.output)
                        .map_err(|error| refusal(error.to_string()))?;
                    fp8_scratch_bytes = fp8_scratch_bytes.max(entry.spec.scratch["scratch"]);
                } else {
                    cuteafd_ffi::draft_projection::validate_spec(&entry.spec, k, n, rows, request.output)
                        .map_err(|error| refusal(error.to_string()))?;
                }
            } else if fp8 { fp8_scratch_bytes = fp8_scratch_bytes.max(256); }
            programs.insert(name);
        }
        Ok(Self { programs: programs.into_iter().collect(), fp8_scratch_bytes,
            resolution: if manifest.is_some() { NativeResolution::Exact } else { NativeResolution::Assumed } })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::trellis_core::CoreSplit;
    use serde_json::{json, Value};

    fn layout(fp8: bool) -> NativeLayout {
        NativeLayout { stem: "o".into(), shape: [8192, 4096], fp8, rank: 0, ranks: 2, split: CoreSplit::Input }
    }

    pub(crate) fn entry(fp8: bool, output: ProjectionOutput, stage: usize, scratch: u64) -> Value {
        let suffix = if output == ProjectionOutput::F32 { "_f32" } else { "" };
        if fp8 {
            json!({"name":format!("native_attention_n8192_k4096_m{stage}{suffix}"),"capacity_rows":stage,
                "pointers":[{"name":"x"},{"name":"weight"},{"name":"scales"},{"name":"out"},{"name":"scratch"}],
                "scalars":[{"type":"int32"}],"scratch_bytes_at_capacity":{"scratch":scratch},
                "geometry":{"input_features":4096,"output_features":8192,"output_dtype":output.torch_dtype(),
                    "max_rows":stage,"decode_rows":64,"scale_block":128,"storage":"row_major_e4m3",
                    "reduction":"canonical_fp8_gemv64_w8a16"}})
        } else {
            json!({"name":output.program_name(4096,8192),"capacity_rows":4096,
                "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],"scalars":[{"type":"int32"}],
                "scratch_bytes_at_capacity":{},"geometry":{"width":4096,"outputs":8192,
                    "output_dtype":output.torch_dtype(),"reduction":"ordered_k16","tiles_by_rows":[[1,16,32,2]]}})
        }
    }

    #[test]
    fn selects_only_runtime_fp8_stage_for_each_ceiling() {
        let layout = layout(true);
        for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
            for rows in [1, 64, 65, 2048, 4096] {
                let stage = if rows <= 64 { 64 } else { 4096 };
                let selected = entry(true, output, stage, 1024);
                let manifest = ProgramsManifest::from_manifest(&json!({"programs":[selected.clone()]})).unwrap();
                let requests = [NativeProgram { layout: &layout, output }];
                let result = NativePrograms::resolve(&requests, rows, Some(&manifest)).unwrap();
                assert_eq!(result.programs, [selected["name"].as_str().unwrap()]);
                assert_eq!(result.fp8_scratch_bytes, 1024);
                assert_eq!(result.resolution, NativeResolution::Exact);
                let other = if stage == 64 { 4096 } else { 64 };
                let manifest = ProgramsManifest::from_manifest(&json!({"programs":[entry(true,output,other,1024)]})).unwrap();
                assert!(NativePrograms::resolve(&requests, rows, Some(&manifest)).is_err());
            }
        }
    }

    #[test]
    fn explicit_outputs_deduplicate_and_share_maximum_scratch() {
        let bf16 = layout(false);
        let fp8 = layout(true);
        let requests = [NativeProgram { layout: &bf16, output: ProjectionOutput::Bf16 },
            NativeProgram { layout: &bf16, output: ProjectionOutput::F32 },
            NativeProgram { layout: &fp8, output: ProjectionOutput::Bf16 },
            NativeProgram { layout: &fp8, output: ProjectionOutput::F32 },
            NativeProgram { layout: &fp8, output: ProjectionOutput::F32 }];
        let manifest = ProgramsManifest::from_manifest(&json!({"programs":[
            entry(false,ProjectionOutput::Bf16,4096,0),entry(false,ProjectionOutput::F32,4096,0),
            entry(true,ProjectionOutput::Bf16,4096,512),entry(true,ProjectionOutput::F32,4096,2048)]})).unwrap();
        let exact = NativePrograms::resolve(&requests, 2048, Some(&manifest)).unwrap();
        assert_eq!(exact.programs.len(), 4);
        assert!(exact.programs.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(exact.fp8_scratch_bytes, 2048);
        let assumed = NativePrograms::resolve(&requests, 2048, None).unwrap();
        assert_eq!(assumed.programs, exact.programs);
        assert_eq!(assumed.fp8_scratch_bytes, 256);
        assert_eq!(assumed.resolution, NativeResolution::Assumed);
        assert_eq!(NativePrograms::resolve(&requests[..2], 2048, None).unwrap().fp8_scratch_bytes, 0);
    }

    #[test]
    fn exact_refuses_missing_stale_abi_capacity_and_scratch() {
        for fp8 in [false, true] {
            let layout = layout(fp8);
            let requests = [NativeProgram { layout: &layout, output: ProjectionOutput::F32 }];
            let valid = entry(fp8, ProjectionOutput::F32, 4096, 512);
            for (field, value) in [("pointers",json!([])),("scalars",json!([])),
                ("capacity_rows",json!(64)),("scratch_bytes_at_capacity",json!({"bad":512})),
                ("geometry",json!({}))] {
                let mut bad = valid.clone(); bad[field] = value;
                let manifest = ProgramsManifest::from_manifest(&json!({"programs":[bad]})).unwrap();
                assert!(NativePrograms::resolve(&requests,2048,Some(&manifest)).is_err(),"{fp8} {field}");
            }
            let empty = ProgramsManifest::default();
            assert!(NativePrograms::resolve(&requests,2048,Some(&empty)).is_err());
            for rows in [0,4097] {
                assert!(NativePrograms::resolve(&requests,rows,Some(&ProgramsManifest::from_manifest(
                    &json!({"programs":[valid.clone()]})).unwrap())).is_err());
            }
        }
    }

    #[test]
    fn typed_specs_match_manifest_entries_without_losing_metadata() {
        use cuteafd_ffi::programs::ProgramSpec;
        let mut values = Vec::new();
        let mut specs = Vec::new();
        for fp8 in [false, true] {
            for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
                for stage in if fp8 { &[64, 4096][..] } else { &[4096][..] } {
                    let mut value = entry(fp8, output, *stage, 2048);
                    value["scalars"] = json!([{"type":"int32"},{"type":"int64"},{"type":"float32"}]);
                    specs.push(ProgramSpec {
                        name: value["name"].as_str().unwrap().into(),
                        pointers: if fp8 { ["x", "weight", "scales", "out", "scratch"].as_slice() }
                            else { ["x", "w", "out"].as_slice() }.iter().map(|name| (*name).into()).collect(),
                        scalar_kinds: vec![b'i', b'l', b'f'],
                        scratch: if fp8 { [("scratch".into(), 2048)].into() } else { Default::default() },
                        capacity_rows: if fp8 { *stage as u32 } else { 4096 },
                        geometry: value["geometry"].clone(),
                    });
                    values.push(value);
                }
            }
        }
        let parsed = ProgramsManifest::from_manifest(&json!({"programs":values})).unwrap();
        let typed = ProgramsManifest::from_specs(specs.iter()).unwrap();
        for spec in &specs {
            let left = &parsed.get(&spec.name).unwrap().spec;
            let right = &typed.get(&spec.name).unwrap().spec;
            assert_eq!(left.name, right.name);
            assert_eq!(left.pointers, right.pointers);
            assert_eq!(left.scalar_kinds, right.scalar_kinds);
            assert_eq!(left.scratch, right.scratch);
            assert_eq!(left.capacity_rows, right.capacity_rows);
            assert_eq!(left.geometry, right.geometry);
        }
        specs[0].geometry = json!({"changed_after_copy":true});
        assert_ne!(typed.get(&specs[0].name).unwrap().spec.geometry, specs[0].geometry);
    }

    #[test]
    fn typed_specs_narrowly_filter_and_refuse_native_duplicates() {
        let value = entry(false, ProjectionOutput::Bf16, 4096, 0);
        let parsed = ProgramsManifest::from_manifest(&json!({"programs":[value]})).unwrap();
        let name = ProjectionOutput::Bf16.program_name(4096, 8192);
        let native = parsed.get(&name).unwrap().spec.clone();
        let mut unrelated = native.clone();
        unrelated.name = "muse_attention".into();
        unrelated.geometry = Value::Null;
        let mut near_match = unrelated.clone();
        near_match.name = "native_attention_other".into();
        let specs = [&unrelated, &unrelated, &near_match, &native];
        let typed = ProgramsManifest::from_specs(specs.into_iter()).unwrap();
        assert!(typed.get(&native.name).is_some());
        assert!(typed.get(&unrelated.name).is_none());
        assert!(typed.get(&near_match.name).is_none());
        let error = ProgramsManifest::from_specs([&native, &native].into_iter()).unwrap_err();
        assert!(error.to_string().contains(&format!("duplicate native program {name}")));
        assert!(ProgramsManifest::from_specs(std::iter::empty()).unwrap().get(&name).is_none());
    }

    #[test]
    fn manifest_narrowly_parses_native_entries_and_refuses_duplicates() {
        let valid = entry(false,ProjectionOutput::Bf16,4096,0);
        assert!(ProgramsManifest::from_manifest(&json!({"programs":[{"name":"unrelated","opaque":true},valid.clone()]})).is_ok());
        assert!(ProgramsManifest::from_manifest(&json!({"programs":[valid.clone(),valid.clone()]})).is_err());
        for field in ["capacity_rows","pointers","scalars","scratch_bytes_at_capacity","geometry"] {
            let mut bad = valid.clone(); bad.as_object_mut().unwrap().remove(field);
            assert!(ProgramsManifest::from_manifest(&json!({"programs":[bad]})).is_err(),"{field}");
        }
    }
}
