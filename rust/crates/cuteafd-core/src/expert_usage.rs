//! Per-engine executed-route telemetry. Counts describe actual expert work,
//! including rejected verifier rows, not committed/output tokens.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

pub const SNAPSHOTS: usize = 61;
pub const SHORT_SECONDS: u64 = 5;
pub const LONG_SECONDS: u64 = 60;
pub const WORKER_STACK_BYTES: usize = 256 * 1024;
/// Named bounds for worker/channel handles and per-device snapshot descriptors.
/// Payloads and the explicitly sized thread stack are charged separately.
pub const SHARED_METADATA_BYTES: usize = 4096;
pub const DEVICE_METADATA_BYTES: usize = 512;

pub fn device_owner(gpu: u8, buffer: &str) -> String {
    format!("expert-usage.gpu{gpu}.{buffer}")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    pub routed_layers: Vec<usize>,
}
impl Geometry {
    pub fn cells(&self) -> Option<usize> {
        self.layers.checked_mul(self.experts)
    }
    pub fn table_bytes(&self) -> Option<usize> {
        self.cells()?.checked_mul(16)
    }
    /// One merged history plus the worker's next merged sample, ring metadata,
    /// two geometries, bounded worker/device metadata and the explicit stack.
    /// Device samples merge directly from pinned storage, with no extra matrices.
    pub fn host_bytes(&self, devices: usize) -> Option<usize> {
        self.table_bytes()?
            .checked_mul(SNAPSHOTS + 1)?
            .checked_add(SNAPSHOTS.checked_mul(std::mem::size_of::<Snapshot>())?)?
            .checked_add(
                self.routed_layers
                    .len()
                    .checked_mul(2 * std::mem::size_of::<usize>())?,
            )?
            .checked_add(std::mem::size_of::<History>() + std::mem::size_of::<Geometry>())?
            .checked_add(SHARED_METADATA_BYTES)?
            .checked_add(devices.checked_mul(DEVICE_METADATA_BYTES)?)?
            .checked_add(WORKER_STACK_BYTES)
    }
    pub fn validate(&self) -> bool {
        self.layers > 0
            && self.experts > 0
            && self.topk > 0
            && self.topk <= self.experts
            && self.table_bytes().is_some()
            && !self.routed_layers.is_empty()
            && self.routed_layers.windows(2).all(|p| p[0] < p[1])
            && self.routed_layers.iter().all(|&l| l < self.layers)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub elapsed_s: f64,
    /// Flattened layer-major counts: one increment per real row selecting this expert.
    pub prefill: Vec<u64>,
    pub decode: Vec<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Summary {
    pub touched: usize,
    pub possible: usize,
    pub coverage: f64,
    /// Sum of squared within-layer shares, averaged over active routed layers.
    pub concentration: f64,
    pub routed_rows: u64,
}
pub fn summarize(geometry: &Geometry, counts: &[u64]) -> Summary {
    let mut touched = 0;
    let mut selections = 0u64;
    let mut concentration = 0.0;
    let mut active = 0;
    for &layer in &geometry.routed_layers {
        let row = &counts[layer * geometry.experts..(layer + 1) * geometry.experts];
        let total = row.iter().fold(0u64, |n, &count| n.saturating_add(count));
        let denominator = row.iter().map(|&count| count as f64).sum::<f64>();
        touched += row.iter().filter(|&&n| n > 0).count();
        selections = selections.saturating_add(total);
        if total > 0 {
            active += 1;
            concentration += row
                .iter()
                .map(|&n| (n as f64 / denominator).powi(2))
                .sum::<f64>();
        }
    }
    let possible = geometry.routed_layers.len() * geometry.experts;
    Summary {
        touched,
        possible,
        coverage: touched as f64 / possible as f64,
        concentration: if active == 0 {
            0.0
        } else {
            concentration / active as f64
        },
        routed_rows: selections / geometry.topk as u64,
    }
}

#[derive(Debug)]
struct History {
    geometry: Geometry,
    samples: VecDeque<Snapshot>,
    error: Option<String>,
}
/// Cloneable API reader; no CUDA access, process globals or serving-loop locks.
#[derive(Clone, Debug, Default)]
pub struct Reader(Arc<RwLock<Option<History>>>);
impl Reader {
    pub fn initialize(&self, geometry: Geometry) -> Result<(), &'static str> {
        if !geometry.validate() {
            return Err("invalid expert usage geometry");
        }
        let mut history = self.0.write().unwrap_or_else(|e| e.into_inner());
        if history.is_some() {
            return Err("expert usage reader already initialized");
        }
        let cells = geometry.cells().unwrap();
        let zero = Snapshot {
            elapsed_s: 0.0,
            prefill: vec![0; cells],
            decode: vec![0; cells],
        };
        let mut samples = VecDeque::with_capacity(SNAPSHOTS);
        samples.push_back(zero);
        *history = Some(History {
            geometry,
            samples,
            error: None,
        });
        Ok(())
    }
    pub fn publish(&self, sample: Snapshot) -> Result<(), &'static str> {
        let mut guard = self.0.write().unwrap_or_else(|e| e.into_inner());
        let history = guard.as_mut().ok_or("expert usage not initialized")?;
        let cells = history.geometry.cells().unwrap();
        let previous = history.samples.back().unwrap();
        if !sample.elapsed_s.is_finite()
            || sample.elapsed_s <= previous.elapsed_s
            || sample.prefill.len() != cells
            || sample.decode.len() != cells
            || sample
                .prefill
                .iter()
                .zip(&previous.prefill)
                .chain(sample.decode.iter().zip(&previous.decode))
                .any(|(a, b)| a < b)
        {
            return Err("invalid/non-monotonic expert usage snapshot");
        }
        if history.samples.len() == SNAPSHOTS {
            history.samples.pop_front();
        }
        history.samples.push_back(sample);
        Ok(())
    }
    pub fn fail(&self, error: String) {
        if let Some(history) = self.0.write().unwrap_or_else(|e| e.into_inner()).as_mut() {
            history.error = Some(error);
        }
    }
    pub fn snapshot(&self) -> serde_json::Value {
        let guard = self.0.read().unwrap_or_else(|e| e.into_inner());
        let Some(history) = guard.as_ref() else {
            return serde_json::json!({"enabled": false});
        };
        let latest = history.samples.back().unwrap();
        let window = |seconds: Option<u64>| {
            let baseline = seconds.and_then(|s| {
                let cutoff = latest.elapsed_s - s as f64;
                history
                    .samples
                    .iter()
                    .rev()
                    .find(|p| p.elapsed_s <= cutoff)
                    .or_else(|| history.samples.front())
            });
            let delta = |current: &[u64], old: Option<&[u64]>| -> Vec<u64> {
                current
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| n - old.map_or(0, |p| p[i]))
                    .collect()
            };
            let prefill = delta(&latest.prefill, baseline.map(|p| p.prefill.as_slice()));
            let decode = delta(&latest.decode, baseline.map(|p| p.decode.as_slice()));
            let combined: Vec<_> = prefill
                .iter()
                .zip(&decode)
                .map(|(a, b)| a.saturating_add(*b))
                .collect();
            serde_json::json!({"elapsed_s": latest.elapsed_s - baseline.map_or(0.0, |p| p.elapsed_s),
                "prefill_routed_rows": prefill, "decode_routed_rows": decode,
                "prefill": summarize(&history.geometry, &prefill), "decode": summarize(&history.geometry, &decode),
                "combined": summarize(&history.geometry, &combined)})
        };
        serde_json::json!({"enabled": true, "semantics": "executed_routed_rows", "geometry": history.geometry,
            "sampled_s": latest.elapsed_s, "error": history.error,
            "windows": {"short": window(Some(SHORT_SECONDS)), "long": window(Some(LONG_SECONDS)), "total": window(None)}})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry() -> Geometry {
        Geometry {
            layers: 3,
            experts: 4,
            topk: 2,
            routed_layers: vec![1, 2],
        }
    }
    fn sample(t: f64, n: u64) -> Snapshot {
        Snapshot {
            elapsed_s: t,
            prefill: vec![n; 12],
            decode: vec![2 * n; 12],
        }
    }
    #[test]
    fn independent_geometry_and_bounded_windows() {
        let a = Reader::default();
        let b = Reader::default();
        a.initialize(geometry()).unwrap();
        b.initialize(Geometry {
            layers: 1,
            experts: 2,
            topk: 1,
            routed_layers: vec![0],
        })
        .unwrap();
        for n in 1..=100 {
            a.publish(sample(n as f64, n)).unwrap();
        }
        assert_eq!(
            a.0.read().unwrap().as_ref().unwrap().samples.len(),
            SNAPSHOTS
        );
        let value = a.snapshot();
        assert_eq!(value["windows"]["short"]["prefill_routed_rows"][0], 5);
        assert_eq!(value["windows"]["long"]["prefill_routed_rows"][0], 60);
        assert_eq!(value["windows"]["total"]["prefill_routed_rows"][0], 100);
        assert_eq!(b.snapshot()["geometry"]["experts"], 2);
    }
    #[test]
    fn invalid_snapshots_do_not_corrupt_totals() {
        let reader = Reader::default();
        reader.initialize(geometry()).unwrap();
        reader.publish(sample(1.0, 10)).unwrap();
        assert!(reader.publish(sample(2.0, 9)).is_err());
        assert!(reader.publish(sample(f64::NAN, 12)).is_err());
        assert!(reader.initialize(geometry()).is_err());
        assert_eq!(
            reader.snapshot()["windows"]["total"]["decode_routed_rows"][0],
            20
        );
    }
    #[test]
    fn large_totals_saturate_without_distorting_concentration() {
        let geometry = Geometry {
            layers: 1,
            experts: 2,
            topk: 1,
            routed_layers: vec![0],
        };
        let reader = Reader::default();
        reader.initialize(geometry.clone()).unwrap();
        reader
            .publish(Snapshot {
                elapsed_s: 1.0,
                prefill: vec![u64::MAX; 2],
                decode: vec![u64::MAX; 2],
            })
            .unwrap();
        let value = reader.snapshot();
        assert_eq!(
            value["windows"]["total"]["combined"]["routed_rows"],
            u64::MAX
        );
        assert_eq!(value["windows"]["total"]["combined"]["concentration"], 0.5);
        assert_eq!(
            value["windows"]["total"]["prefill_routed_rows"][0],
            u64::MAX
        );
        reader.fail("snapshot timed out; stale".into());
        let stale = reader.snapshot();
        assert_eq!(stale["sampled_s"], 1.0);
        assert_eq!(stale["windows"], value["windows"]);
        assert!(stale["error"].as_str().unwrap().contains("stale"));
        assert_eq!(
            geometry.host_bytes(2).unwrap() - geometry.host_bytes(1).unwrap(),
            DEVICE_METADATA_BYTES
        );
    }

    #[test]
    fn dense_layers_are_excluded_from_coverage() {
        let summary = summarize(&geometry(), &[0, 0, 0, 0, 2, 2, 0, 0, 4, 0, 0, 0]);
        assert_eq!(
            (summary.touched, summary.possible, summary.routed_rows),
            (3, 8, 4)
        );
        assert_eq!(summary.concentration, 0.75);
    }
}
