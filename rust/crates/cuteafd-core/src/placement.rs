use crate::{CuteafdError, TensorRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlacementPolicy {
    Modulo,
    Range,
}

impl FromStr for PlacementPolicy {
    type Err = CuteafdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "modulo" => Ok(PlacementPolicy::Modulo),
            "range" => Ok(PlacementPolicy::Range),
            other => Err(CuteafdError::UnknownPlacementPolicy(other.to_owned())),
        }
    }
}

pub fn owner_for_expert(
    _layer_id: usize,
    expert_id: usize,
    routed_experts: usize,
    hosts: &[String],
    policy: PlacementPolicy,
) -> Option<String> {
    if hosts.is_empty() || routed_experts == 0 || expert_id >= routed_experts {
        return None;
    }
    let index = match policy {
        PlacementPolicy::Modulo => expert_id % hosts.len(),
        PlacementPolicy::Range => {
            let chunk = routed_experts.div_ceil(hosts.len());
            (expert_id / chunk).min(hosts.len() - 1)
        }
    };
    hosts.get(index).cloned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorAssignment {
    pub tensor_name: String,
    pub owner: String,
    pub role: TensorRole,
    pub layer_id: Option<u32>,
    pub expert_id: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertOwnerLookup {
    owners_by_expert: BTreeMap<(usize, usize), String>,
}

impl ExpertOwnerLookup {



    pub fn owner_for(&self, layer_id: usize, expert_id: usize) -> Option<&str> {
        self.owners_by_expert
            .get(&(layer_id, expert_id))
            .map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.owners_by_expert.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners_by_expert.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPlan {
    pub model_id: String,
    pub placement_version: String,
    pub policy: PlacementPolicy,
    pub coordinator_host: String,
    pub expert_hosts: Vec<String>,
    pub assignments: Vec<TensorAssignment>,
}

impl LoadPlan {
}
