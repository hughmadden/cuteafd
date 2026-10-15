//! Exact, startup-only plan publication and worker acknowledgement.
use anyhow::{ensure, Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Families retain their launcher's wire schema and placement constraints.
/// The shared handoff owns launch identity, bounded IO and the commit protocol.
pub(crate) trait HandoffPlan: Serialize + DeserializeOwned + PartialEq {
    fn validate(&self) -> Result<()>;
    fn nonce(&self) -> &str;
    fn set_nonce(&mut self, nonce: String);
    fn same_placement(&self, other: &Self) -> bool;
}

pub(crate) struct PlacementHandoff<P> {
    directory: PathBuf,
    plan: P,
}

impl<P: HandoffPlan> PlacementHandoff<P> {
    /// A committed plan may resume only with the identical placement. New
    /// publications reject stale state rather than treating it as readiness.
    pub(crate) fn publish(directory: &Path, mut plan: P) -> Result<Self> {
        plan.set_nonce(format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        Self::validate(&plan)?;
        fs::create_dir_all(directory)?;
        if let Some(committed) = Self::resumed(directory)? {
            ensure!(
                committed.same_placement(&plan),
                "container restart cannot change acknowledged worker boundary"
            );
            return Ok(Self {
                directory: directory.to_owned(),
                plan: committed,
            });
        }
        ensure!(
            !directory.join("plan.json").exists() && !directory.join("ready.json").exists(),
            "placement handoff directory contains stale state"
        );
        Self::write_plan(directory, ".plan-pending", "plan.json", &plan, true)?;
        Ok(Self {
            directory: directory.to_owned(),
            plan,
        })
    }

    fn validate(plan: &P) -> Result<()> {
        plan.validate()?;
        ensure!(
            !plan.nonce().is_empty() && plan.nonce().len() <= 64,
            "invalid placement launch identity"
        );
        Ok(())
    }

    fn read_plan(path: &Path) -> Result<Option<P>> {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).context("opening placement state"),
        };
        let mut bytes = Vec::new();
        file.take(4097).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 4096, "placement state exceeds size limit");
        let plan: P = serde_json::from_slice(&bytes)?;
        Self::validate(&plan)?;
        Ok(Some(plan))
    }

    fn write_plan(
        directory: &Path,
        pending: &str,
        final_name: &str,
        plan: &P,
        fresh: bool,
    ) -> Result<()> {
        let bytes = serde_json::to_vec(plan)?;
        ensure!(bytes.len() <= 4096, "placement state exceeds size limit");
        let temporary = directory.join(pending);
        let mut options = fs::OpenOptions::new();
        options.write(true);
        if fresh {
            options.create_new(true);
        } else {
            options.create(true).truncate(true);
        }
        let mut file = options
            .open(&temporary)
            .context("creating placement handoff")?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(temporary, directory.join(final_name))?;
        Ok(())
    }

    pub(crate) fn resumed(directory: &Path) -> Result<Option<P>> {
        Self::read_plan(&directory.join("committed.json"))
    }

    fn acknowledged(&self) -> Result<bool> {
        if let Some(committed) = Self::resumed(&self.directory)? {
            ensure!(committed == self.plan, "committed placement changed");
            return Ok(true);
        }
        let Some(ready) = Self::read_plan(&self.directory.join("ready.json"))? else {
            return Ok(false);
        };
        ensure!(
            ready == self.plan,
            "worker acknowledgement does not match this placement and launch"
        );
        Self::write_plan(
            &self.directory,
            ".committed-pending",
            "committed.json",
            &self.plan,
            false,
        )?;
        Ok(true)
    }

    /// Startup only, before transport connection or serving lanes exist.
    pub(crate) fn wait_ready(&self, timeout: Duration) -> Result<()> {
        let start = Instant::now();
        loop {
            if self.acknowledged()? {
                return Ok(());
            }
            ensure!(
                start.elapsed() < timeout,
                "timed out waiting for Spark placement readiness"
            );
            std::thread::sleep(
                Duration::from_millis(50).min(timeout.saturating_sub(start.elapsed())),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Plan {
        nonce: String,
        owners: Vec<u8>,
    }
    impl HandoffPlan for Plan {
        fn validate(&self) -> Result<()> {
            ensure!(
                !self.owners.is_empty() && self.owners.iter().all(|&owner| owner < 2),
                "invalid owners"
            );
            Ok(())
        }
        fn nonce(&self) -> &str {
            &self.nonce
        }
        fn set_nonce(&mut self, nonce: String) {
            self.nonce = nonce;
        }
        fn same_placement(&self, other: &Self) -> bool {
            self.owners == other.owners
        }
    }
    fn plan() -> Plan {
        Plan {
            nonce: String::new(),
            owners: vec![0, 1],
        }
    }

    #[test]
    fn publication_exact_acknowledgement_and_resume() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let handoff = PlacementHandoff::publish(directory.path(), plan())?;
        assert!(PlacementHandoff::<Plan>::resumed(directory.path())?.is_none());
        assert!(!directory.path().join(".plan-pending").exists());
        assert!(handoff.wait_ready(Duration::ZERO).is_err());
        fs::copy(
            directory.path().join("plan.json"),
            directory.path().join("ready.json"),
        )?;
        handoff.wait_ready(Duration::ZERO)?;
        fs::remove_file(directory.path().join("ready.json"))?;
        PlacementHandoff::publish(directory.path(), plan())?.wait_ready(Duration::ZERO)?;
        assert!(PlacementHandoff::publish(
            directory.path(),
            Plan {
                nonce: String::new(),
                owners: vec![1, 0]
            }
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn stale_malformed_oversized_and_wrong_launch_are_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let handoff = PlacementHandoff::publish(directory.path(), plan())?;
        assert!(PlacementHandoff::publish(directory.path(), plan()).is_err());
        let mut wrong = handoff.plan.clone();
        wrong.nonce.push('x');
        fs::write(
            directory.path().join("ready.json"),
            serde_json::to_vec(&wrong)?,
        )?;
        assert!(handoff.wait_ready(Duration::ZERO).is_err());
        for bytes in [
            vec![b' '; 4097],
            b"{}".to_vec(),
            br#"{"nonce":"x","owners":[0],"unknown":1}"#.to_vec(),
        ] {
            fs::write(directory.path().join("ready.json"), bytes)?;
            assert!(handoff.wait_ready(Duration::ZERO).is_err());
        }
        Ok(())
    }

    #[test]
    fn oversized_publication_is_rejected_before_pending_file() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let oversized = Plan {
            nonce: String::new(),
            owners: vec![0; 4096],
        };
        assert!(PlacementHandoff::publish(directory.path(), oversized).is_err());
        assert!(!directory.path().join(".plan-pending").exists());
        assert!(!directory.path().join("plan.json").exists());
        Ok(())
    }
}
