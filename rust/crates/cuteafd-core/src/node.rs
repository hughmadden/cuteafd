use crate::CuteafdError;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeRole {
    Coordinator,
    Expert,
    Dev,
}

impl NodeRole {
    /// The GPU architectures this role's images are built for.
    pub fn cuda_arches(self) -> &'static [GpuArch] {
        match self {
            NodeRole::Coordinator | NodeRole::Dev => &[GpuArch::Sm89, GpuArch::Sm120],
            NodeRole::Expert => &[GpuArch::Sm121],
        }
    }

    /// [`Self::cuda_arches`] joined with `|`, as `doctor` reports it.
    pub fn expected_cuda_arch(self) -> String {
        self.cuda_arches().iter().map(|arch| arch.as_str()).collect::<Vec<_>>().join("|")
    }
}

impl fmt::Display for NodeRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeRole::Coordinator => write!(f, "coordinator"),
            NodeRole::Expert => write!(f, "expert"),
            NodeRole::Dev => write!(f, "dev"),
        }
    }
}

impl FromStr for NodeRole {
    type Err = CuteafdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "coordinator" => Ok(NodeRole::Coordinator),
            "expert" | "spark" => Ok(NodeRole::Expert),
            "dev" => Ok(NodeRole::Dev),
            other => Err(CuteafdError::UnknownRole(other.to_owned())),
        }
    }
}

/// A GPU architecture the engine plans for and runs on. Anything else is
/// rejected where it enters (a device's compute capability, a CLI option),
/// so no planner or admission path can fall back to another card's numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuArch {
    /// Ada (RTX 4090 class): coordinator.
    #[serde(rename = "sm_89")]
    Sm89,
    /// Blackwell RTX (RTX 5090, RTX PRO 6000): coordinator.
    #[serde(rename = "sm_120")]
    Sm120,
    /// GB10 (DGX Spark): expert ranks.
    #[serde(rename = "sm_121")]
    Sm121,
}

impl GpuArch {
    pub const ALL: [GpuArch; 3] = [GpuArch::Sm89, GpuArch::Sm120, GpuArch::Sm121];

    pub const fn as_str(self) -> &'static str {
        match self {
            GpuArch::Sm89 => "sm_89",
            GpuArch::Sm120 => "sm_120",
            GpuArch::Sm121 => "sm_121",
        }
    }

    /// The architecture of a device reporting compute capability `major.minor`.
    pub fn from_compute_capability(major: i32, minor: i32) -> Result<Self, CuteafdError> {
        match (major, minor) {
            (8, 9) => Ok(GpuArch::Sm89),
            (12, 0) => Ok(GpuArch::Sm120),
            (12, 1) => Ok(GpuArch::Sm121),
            _ => Err(CuteafdError::UnsupportedGpuArch(format!("sm_{major}{minor}"))),
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GpuArch {
    type Err = CuteafdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        GpuArch::ALL.into_iter().find(|arch| arch.as_str() == value)
            .ok_or_else(|| CuteafdError::UnsupportedGpuArch(value.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_arch_round_trips_and_rejects_unknown_devices() {
        for arch in GpuArch::ALL {
            assert_eq!(arch.as_str().parse::<GpuArch>().unwrap(), arch);
            assert_eq!(serde_json::to_value(arch).unwrap(), arch.as_str());
        }
        assert_eq!(GpuArch::from_compute_capability(8, 9).unwrap(), GpuArch::Sm89);
        assert_eq!(GpuArch::from_compute_capability(12, 0).unwrap(), GpuArch::Sm120);
        assert_eq!(GpuArch::from_compute_capability(12, 1).unwrap(), GpuArch::Sm121);
        assert!(GpuArch::from_compute_capability(8, 6).is_err());
        assert!(GpuArch::from_compute_capability(10, 0).is_err());
        assert!("sm_86".parse::<GpuArch>().is_err());
        assert!("sm_120a".parse::<GpuArch>().is_err());
    }

    #[test]
    fn roles_name_their_image_arches() {
        assert_eq!(NodeRole::Coordinator.expected_cuda_arch(), "sm_89|sm_120");
        assert_eq!(NodeRole::Expert.expected_cuda_arch(), "sm_121");
    }
}
