//! Retained EXL3 capacity arenas, read from the matching exported manifests.
use super::{product, sum, CacheGeometryError};
use serde_json::Value;
use std::collections::BTreeMap;

pub fn exl3_workspace_bytes(manifests: &[Value], fp8_k32: bool) -> Result<u64, CacheGeometryError> {
    let invalid = |name: &str, what: &str| CacheGeometryError::ResidentTensor {
        name: format!("EXL3 workspace/{name}"), what: what.into(),
    };
    if manifests.is_empty() { return Err(invalid("manifests", "no capacity specializations")); }
    let mut shared = BTreeMap::<String, (u64, String)>::new();
    let mut private = Vec::new();
    for manifest in manifests {
        let buffers = manifest["buffers"].as_object().ok_or_else(|| invalid("buffers", "missing allocation map"))?;
        private.push(manifest["trellis_lut"]["bytes"].as_u64()
            .ok_or_else(|| invalid("trellis_lut", "missing asset extent"))?.max(16));
        for (name, spec) in buffers {
            let allocation = spec["allocation"].as_str().ok_or_else(|| invalid(name, "missing allocation owner"))?;
            if name != allocation { continue; }
            let bytes = spec["bytes"].as_u64().ok_or_else(|| invalid(name, "missing byte extent"))?.max(16);
            let zero = spec["zero_on_create"].as_bool().ok_or_else(|| invalid(name, "missing initialization policy"))?;
            if zero { private.push(bytes); continue; }
            let dtype = spec["dtype"].as_str().ok_or_else(|| invalid(name, "missing dtype"))?;
            let entry = shared.entry(name.clone()).or_insert((0, dtype.into()));
            if entry.1 != dtype { return Err(invalid(name, "shared capacity arena has conflicting dtypes")); }
            entry.0 = entry.0.max(bytes);
        }
        match manifest["input_format"].as_str() {
            Some("e4m3_k32") => (),
            None | Some("bf16") if fp8_k32 => {
                let rows = manifest["capacity"].as_u64().ok_or_else(|| invalid("capacity", "missing row extent"))?;
                let hidden = manifest["hidden"].as_u64().ok_or_else(|| invalid("hidden", "missing hidden extent"))?;
                private.push(product("EXL3 wire decode", &[rows, hidden, 2])?);
            }
            None | Some("bf16") => (),
            _ => return Err(invalid("input_format", "unknown wire representation")),
        }
    }
    private.extend(shared.into_values().map(|(bytes, _)| bytes));
    sum("EXL3 capacity arenas", &private)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn aliases_share_scratch_but_capacity_state_and_wire_decode_stay_private() {
        let manifest = |rows, bytes| json!({"capacity": rows, "hidden": 8,
            "trellis_lut": {"bytes": 32}, "buffers": {
                "scratch": {"bytes": bytes, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
                "alias": {"bytes": 99999, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
                "state": {"bytes": 0, "allocation": "state", "dtype": "i32", "zero_on_create": true}}});
        let manifests = [manifest(1, 64), manifest(16, 1024)];
        assert_eq!(exl3_workspace_bytes(&manifests, true).unwrap(), 1024 + 2 * (32 + 16) + 17 * 8 * 2);
        assert_eq!(exl3_workspace_bytes(&manifests, false).unwrap(), 1024 + 2 * (32 + 16));
        let mut conflict = manifests.clone();
        conflict[1]["buffers"]["scratch"]["dtype"] = json!("f32");
        assert!(exl3_workspace_bytes(&conflict, true).is_err());
    }
}
