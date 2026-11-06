//! Native attention projections over row-major E4M3 weights and exact F32 block-128 scales.
//!
//! BF16 activations stay BF16 at every row count: canonical GEMV through 64,
//! W8A16 TMA above 64, never W8A8. Changing prefill to W8A8 changes the admitted
//! HQA arithmetic. F32 output preserves unrotated TP2 partial accumulators.
use crate::{CuteafdDeviceBuffer, draft_projection::ProjectionOutput,
    programs::{Program, ProgramSpec, Programs, Scalar}};
use std::{collections::BTreeMap, ffi::c_void};
use anyhow::Context;

#[derive(Debug, thiserror::Error)]
pub enum NativeProjectionError {
    #[error("invalid native FP8 projection: {0}")]
    Invalid(String),
    #[error(transparent)]
    Output(#[from] crate::draft_projection::ProjectionError),
    #[error("native FP8 projection: {0}")]
    Program(#[from] anyhow::Error),
    #[error("native FP8 projection extent overflow: {0}")]
    Extent(#[from] std::num::TryFromIntError),
}

type Result<T> = std::result::Result<T, NativeProjectionError>;

fn require(condition: bool, reason: impl Into<String>) -> Result<()> {
    if condition { Ok(()) } else { Err(NativeProjectionError::Invalid(reason.into())) }
}

pub struct Fp8Projections<'a> {
    programs: BTreeMap<(usize, usize), Program<'a>>,
    output: ProjectionOutput,
}

fn name(k: usize, n: usize, capacity: usize, output: ProjectionOutput) -> String {
    let suffix = if output == ProjectionOutput::F32 { "_f32" } else { "" };
    format!("native_attention_n{n}_k{k}_m{capacity}{suffix}")
}

fn capacity(rows: usize) -> Result<usize> {
    require((1..=4096).contains(&rows), "native FP8 projection capacity must be 1..4096")?;
    Ok(if rows <= 64 { 64 } else { 4096 })
}

fn validate(spec: &ProgramSpec, k: usize, n: usize, output: ProjectionOutput) -> Result<()> {
    let g = &spec.geometry;
    let capacity = capacity(spec.capacity_rows as usize)?;
    require(spec.capacity_rows as usize == capacity && spec.name == name(k, n, capacity, output)
        && k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0
        && spec.pointers.iter().map(String::as_str).eq(["x", "weight", "scales", "out", "scratch"])
        && spec.scalar_kinds == b"i"
        && spec.scratch.len() == 1 && spec.scratch.get("scratch").is_some_and(|&b| b >= 256)
        && g["input_features"].as_u64() == Some(k as u64)
        && g["output_features"].as_u64() == Some(n as u64)
        && g["output_dtype"].as_str() == Some(output.torch_dtype())
        && g["scale_block"] == 128 && g["decode_rows"] == 64
        && g["max_rows"].as_u64() == Some(capacity as u64)
        && g["storage"] == "row_major_e4m3"
        && g["reduction"] == "canonical_fp8_gemv64_w8a16",
        format!("{}: expected native block-128 FP8 ABI with canonical decode64 and W8A16 prefill", spec.name))?;
    Ok(())
}

/// Validate the cached typed entry using the same capacity selection as runtime loading.
pub fn validate_spec(spec: &ProgramSpec, k: usize, n: usize, rows: usize,
    output: ProjectionOutput) -> Result<()> {
    validate(spec, k, n, output)?;
    require(spec.capacity_rows as usize == capacity(rows)?,
        "native FP8 projection does not match the requested row-capacity stage")
}

impl<'a> Fp8Projections<'a> {
    /// Resolve all programs and scratch metadata before capture/allocation.
    /// Shapes use the existing projection convention (K,N).
    pub fn load(programs: &'a Programs<'a>, shapes: &[(usize, usize)], max_rows: usize,
        output: ProjectionOutput) -> Result<Self> {
        let capacity = capacity(max_rows)?;
        for &(k, n) in shapes {
            validate_spec(programs.spec(&name(k,n,capacity,output))?, k, n, max_rows, output)?;
        }
        let mut loaded = BTreeMap::new();
        for &(k,n) in shapes {
            if loaded.contains_key(&(k,n)) { continue; }
            let program = programs.program(&name(k,n,capacity,output),
                &["x", "weight", "scales", "out", "scratch"])?;
            loaded.insert((k,n),program);
        }
        Ok(Self { programs: loaded, output })
    }

    /// Loaded programs participating in the family's packing exactness proof.
    pub fn program_names(&self) -> impl Iterator<Item = &str> {
        self.programs.values().map(|program| program.spec().name.as_str())
    }

    /// Charge this named manifest owner before allocating the projection workspace.
    pub fn scratch_bytes(&self, k: usize, n: usize) -> Result<usize> {
        let spec = self.programs.get(&(k,n)).context("native FP8 shape not admitted")?.spec();
        usize::try_from(*spec.scratch.get("scratch").context("native FP8 scratch missing")?)
            .context("native FP8 scratch extent overflow").map_err(Into::into)
    }

    /// # Safety
    /// All buffers are nonoverlapping device memory on `stream`'s device. Input
    /// is BF16 [rows,K], weight is E4M3 [N,K], scales are F32 [N/128,K/128].
    /// Every owner survives the queued work; no allocation or resolution occurs here.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(&self, x: *const c_void, weight: *const c_void, scales: *const c_void,
        out: CuteafdDeviceBuffer, scratch: CuteafdDeviceBuffer, rows: usize, k: usize, n: usize,
        stream: *mut c_void) -> Result<()> {
        let program = self.programs.get(&(k,n)).context("native FP8 shape not admitted")?;
        self.output.require_output(out,rows,n)?;
        require(rows <= program.spec().capacity_rows as usize && rows <= i32::MAX as usize
            && scratch.bytes >= self.scratch_bytes(k,n)? && out.device_id == scratch.device_id
            && [x,weight,scales,scratch.ptr.cast_const()].iter()
                .all(|p| !p.is_null() && *p as usize % 16 == 0),
            "native FP8 projection buffers or rows invalid")?;
        // SAFETY: caller retains the documented owners and stream; dtype/extent/capacity checked above.
        unsafe { program.launch(&[x.cast_mut(),weight.cast_mut(),scales.cast_mut(),out.ptr,scratch.ptr],
            &[Scalar::I32(rows as i32)],stream)?; }
        Ok(())
    }
}

/// CPU-only admission uses the same manifest geometry as the runtime registry.
pub fn validate_package_geometry(manifest: &serde_json::Value, k: usize, n: usize, rows: usize,
    output: ProjectionOutput) -> Result<(String, usize)> {
    let capacity = capacity(rows)?;
    let name = name(k,n,capacity,output);
    let entries = manifest["programs"].as_array().context("native FP8 program inventory")?;
    let mut matches = entries.iter().filter(|entry| entry["name"].as_str() == Some(&name));
    let entry = matches.next().context("native FP8 projection export missing")?;
    require(matches.next().is_none(), "duplicate native FP8 projection export")?;
    let pointers = entry["pointers"].as_array().context("native FP8 pointers")?.iter()
        .map(|p| p["name"].as_str().map(str::to_owned).context("native FP8 pointer name").map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    let scalars = entry["scalars"].as_array().context("native FP8 scalar ABI")?;
    require(scalars.len() == 1 && scalars[0]["type"] == "int32", "native FP8 scalar ABI")?;
    let scratch = entry["scratch_bytes_at_capacity"].as_object().context("native FP8 scratch")?.iter()
        .map(|(k,v)| Ok((k.clone(),v.as_u64().context("native FP8 scratch bytes")?)))
        .collect::<Result<_>>()?;
    let spec = ProgramSpec { name:name.clone(), pointers, scalar_kinds:b"i".to_vec(), scratch,
        capacity_rows:u32::try_from(entry["capacity_rows"].as_u64().context("native FP8 capacity")?)?,
        geometry:entry["geometry"].clone() };
    validate_spec(&spec,k,n,rows,output)?;
    Ok((name,usize::try_from(spec.scratch["scratch"])?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest(output: ProjectionOutput) -> serde_json::Value {
        serde_json::json!({"programs":[{"name":name(4096,8192,64,output),"capacity_rows":64,
            "pointers":[{"name":"x"},{"name":"weight"},{"name":"scales"},{"name":"out"},{"name":"scratch"}],
            "scalars":[{"type":"int32"}],"scratch_bytes_at_capacity":{"scratch":256},
            "geometry":{"input_features":4096,"output_features":8192,"output_dtype":output.torch_dtype(),
                "max_rows":64,"decode_rows":64,"scale_block":128,"storage":"row_major_e4m3",
                "reduction":"canonical_fp8_gemv64_w8a16"}}]})
    }
    #[test]
    fn program_names_reports_only_loaded_programs() {
        for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
            let registry = Fp8Projections { programs: BTreeMap::new(), output };
            assert_eq!(registry.program_names().collect::<Vec<&str>>(), Vec::<&str>::new());
        }
    }

    #[test]
    fn typed_spec_selects_exact_runtime_stage() {
        for output in [ProjectionOutput::Bf16, ProjectionOutput::F32] {
            for limit in [64, 4096] {
                let mut geometry = manifest(output)["programs"][0]["geometry"].clone();
                geometry["max_rows"] = limit.into();
                let spec = ProgramSpec { name: name(4096, 8192, limit, output),
                    pointers: ["x", "weight", "scales", "out", "scratch"].map(str::to_owned).to_vec(),
                    scalar_kinds: b"i".to_vec(), scratch: [("scratch".into(), 256)].into_iter().collect(),
                    capacity_rows: limit as u32, geometry };
                let rows = if limit == 64 { 1 } else { 65 };
                assert!(validate_spec(&spec, 4096, 8192, rows, output).is_ok());
                assert!(validate_spec(&spec, 4096, 8192, limit, output).is_ok());
                assert!(validate_spec(&spec, 4096, 8192, 0, output).is_err());
                assert!(validate_spec(&spec, 4096, 8192, 4097, output).is_err());
                let other_stage = if limit == 64 { 65 } else { 64 };
                assert!(validate_spec(&spec, 4096, 8192, other_stage, output).is_err());
            }
        }
    }

    #[test]
    fn manifest_checks_numerics_dtype_capacity_and_scratch() {
        for output in [ProjectionOutput::Bf16,ProjectionOutput::F32] {
            let m=manifest(output);
            assert_eq!(validate_package_geometry(&m,4096,8192,64,output).unwrap().1,256);
            assert!(validate_package_geometry(&m,4096,8192,65,output).is_err());
            let other=if output == ProjectionOutput::Bf16 {ProjectionOutput::F32} else {ProjectionOutput::Bf16};
            assert!(validate_package_geometry(&m,4096,8192,1,other).is_err());
            for (field,value) in [("decode_rows",serde_json::json!(16)),("scale_block",serde_json::json!(1)),
                ("output_dtype",serde_json::json!("torch.float16")),("reduction",serde_json::json!("w8a8"))] {
                let mut bad=m.clone();bad["programs"][0]["geometry"][field]=value;
                assert!(validate_package_geometry(&bad,4096,8192,1,output).is_err(),"{field}");
            }
            let mut bad=m.clone();bad["programs"][0]["scratch_bytes_at_capacity"]=serde_json::json!({});
            assert!(validate_package_geometry(&bad,4096,8192,1,output).is_err());
        }
    }
}
