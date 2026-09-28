use serde_json::Value;
use std::collections::HashSet;

use crate::error::{invalid_request, ApiError};

const MAX_STRICT_SCHEMA_DEPTH: usize = 64;
const MAX_STRICT_SCHEMA_NODES: usize = 4096;

/// Shared OpenAI strict-mode subset check. `pub(crate)` so the native
/// `serve-native` path can apply the same contract as `validate_request`
/// (the adapter crate deserializes `json_schema` into a fieldless variant).
pub(crate) fn validate_strict_json_schema(
    schema: &Value,
    parameter: impl Into<String>,
) -> Result<(), ApiError> {
    fn visit(
        schema: &Value,
        path: &str,
        depth: usize,
        visited_nodes: &mut usize,
    ) -> Result<(), String> {
        if depth > MAX_STRICT_SCHEMA_DEPTH {
            return Err(format!(
                "{path} exceeds the strict JSON Schema nesting limit of {MAX_STRICT_SCHEMA_DEPTH}"
            ));
        }
        *visited_nodes = visited_nodes
            .checked_add(1)
            .ok_or_else(|| format!("{path} has too many schema nodes"))?;
        if *visited_nodes > MAX_STRICT_SCHEMA_NODES {
            return Err(format!(
                "{path} exceeds the strict JSON Schema node limit of {MAX_STRICT_SCHEMA_NODES}"
            ));
        }
        let Some(object) = schema.as_object() else {
            return Err(format!("{path} must be a JSON Schema object"));
        };
        let is_object = object.get("type").is_some_and(|kind| {
            kind == "object"
                || kind
                    .as_array()
                    .is_some_and(|kinds| kinds.iter().any(|candidate| candidate == "object"))
        }) || object.contains_key("properties");
        if is_object {
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("{path}.properties must be an object in strict mode"))?;
            if object.get("additionalProperties") != Some(&Value::Bool(false)) {
                return Err(format!(
                    "{path}.additionalProperties must be false in strict mode"
                ));
            }
            let required = object
                .get("required")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!("{path}.required must list every property in strict mode")
                })?;
            let required = required
                .iter()
                .map(|name| {
                    name.as_str()
                        .ok_or_else(|| format!("{path}.required entries must be strings"))
                })
                .collect::<Result<HashSet<_>, _>>()?;
            if required.len() != properties.len()
                || properties
                    .keys()
                    .any(|name| !required.contains(name.as_str()))
            {
                return Err(format!(
                    "{path}.required must contain every property exactly once in strict mode"
                ));
            }
            for (name, property) in properties {
                visit(
                    property,
                    &format!("{path}.properties.{name}"),
                    depth + 1,
                    visited_nodes,
                )?;
            }
        }
        if let Some(items) = object.get("items") {
            visit(items, &format!("{path}.items"), depth + 1, visited_nodes)?;
        }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let Some(variants) = object.get(keyword).and_then(Value::as_array) {
                for (index, variant) in variants.iter().enumerate() {
                    visit(
                        variant,
                        &format!("{path}.{keyword}[{index}]"),
                        depth + 1,
                        visited_nodes,
                    )?;
                }
            }
        }
        for keyword in ["$defs", "definitions"] {
            if let Some(definitions) = object.get(keyword).and_then(Value::as_object) {
                for (name, definition) in definitions {
                    visit(
                        definition,
                        &format!("{path}.{keyword}.{name}"),
                        depth + 1,
                        visited_nodes,
                    )?;
                }
            }
        }
        Ok(())
    }

    let parameter = parameter.into();
    let root = schema.as_object().ok_or_else(|| {
        invalid_request(
            "strict JSON Schema root must be an object",
            Some(parameter.clone()),
        )
    })?;
    let root_is_object =
        root.get("type").is_some_and(|kind| kind == "object") || root.contains_key("properties");
    if !root_is_object {
        return Err(invalid_request(
            "strict JSON Schema root type must be object",
            Some(parameter),
        ));
    }
    let mut visited_nodes = 0;
    visit(schema, &parameter, 0, &mut visited_nodes)
        .map_err(|message| invalid_request(message, Some(parameter)))
}
