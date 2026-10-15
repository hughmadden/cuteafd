//! Run history in SQLite under `~/.cache/cuteafd/bench/` (out of Git):
//! reports, the baselines they carried and saved custom profiles.
use crate::profiles::Profile;
use crate::report::Report;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

pub struct Store {
    connection: Connection,
    pub path: Option<PathBuf>,
}

/// `CUTEAFD_BENCH_DIR`, else `~/.cache/cuteafd/bench`.
pub fn default_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CUTEAFD_BENCH_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    Path::new(&home).join(".cache/cuteafd/bench")
}

/// A run row for listings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunRow {
    pub id: String,
    pub created: String,
    pub fingerprint: String,
    pub profile: String,
    pub status: String,
    pub model: String,
    pub hardware: String,
    pub quality_failed: bool,
}

impl RunRow {
    /// The row with the verdict its stored report earns under the rule in force
    /// now. `stored` is the verdict the run was saved with, kept for a report
    /// this build cannot parse.
    fn with_current_verdict(mut self, text: &str, stored: bool) -> Self {
        self.quality_failed = serde_json::from_str::<Report>(text)
            .map_or(stored, |report| crate::quality::under_current_rule(&report).quality_failed());
        self
    }
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join("bench.sqlite");
        let connection = Connection::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let store = Self { connection, path: Some(path) };
        store.migrate()?;
        Ok(store)
    }

    /// An existing store, read only (another process, often root in a container, owns it).
    pub fn open_read(dir: &Path) -> Result<Self> {
        let path = dir.join("bench.sqlite");
        let connection = Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self { connection, path: Some(path) })
    }

    /// A store that forgets everything at exit (no writable directory).
    pub fn memory() -> Result<Self> {
        let store = Self { connection: Connection::open_in_memory()?, path: None };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, created TEXT NOT NULL, fingerprint TEXT NOT NULL, profile TEXT NOT NULL,
                status TEXT NOT NULL, model TEXT NOT NULL, hardware TEXT NOT NULL, quality_failed INTEGER NOT NULL,
                report TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS runs_fingerprint ON runs(fingerprint, created);
             CREATE TABLE IF NOT EXISTS profiles (name TEXT PRIMARY KEY, profile TEXT NOT NULL);")?;
        Ok(())
    }

    pub fn save(&self, report: &Report) -> Result<()> {
        // The verdict is stored as the record of what this run was measured
        // as; a listing re-derives it from the report under the rule in force
        // then, so the column is only the fallback for a report that will not
        // parse.
        let status = serde_json::to_value(report.status)?.as_str().unwrap_or("unknown").to_string();
        self.connection.execute(
            "INSERT INTO runs (id, created, fingerprint, profile, status, model, hardware, quality_failed, report)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET status = excluded.status, quality_failed = excluded.quality_failed,
                report = excluded.report",
            params![report.id, report.created, report.fingerprint, report.profile, status, report.server.model,
                report.server.hardware.line(), report.quality_failed() as i64, serde_json::to_string(report)?])?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Option<Report>> {
        let text: Option<String> = self.connection
            .query_row("SELECT report FROM runs WHERE id = ?1", params![id], |row| row.get(0)).optional()?;
        text.map(|t| serde_json::from_str(&t).context("stored report")).transpose()
    }

    /// The run records, newest first.
    ///
    /// The row's verdict is derived here, from the report's own measured data
    /// under the rule in force now, never from the status the run was stored
    /// with: the dashboard lists this (the warning mark on a failed quality
    /// gate) beside the run it renders, and a run measured before its
    /// checkpoint was reclassified must not be the one surface still calling it
    /// a failure. A report that will not parse keeps the verdict it was saved
    /// with.
    pub fn list(&self, limit: usize) -> Result<Vec<RunRow>> {
        let mut statement = self.connection.prepare(
            "SELECT id, created, fingerprint, profile, status, model, hardware, quality_failed, report FROM runs
             ORDER BY created DESC LIMIT ?1")?;
        let rows = statement.query_map(params![limit as i64], |row| {
            let record = RunRow { id: row.get(0)?, created: row.get(1)?, fingerprint: row.get(2)?, profile: row.get(3)?,
                status: row.get(4)?, model: row.get(5)?, hardware: row.get(6)?,
                quality_failed: row.get::<_, i64>(7)? != 0 };
            let stored = record.quality_failed;
            Ok(record.with_current_verdict(&row.get::<_, String>(8)?, stored))
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Reports on `fingerprint`, newest first (accumulating panels read them).
    pub fn reports_for(&self, fingerprint: &str, limit: usize) -> Result<Vec<Report>> {
        let mut statement = self.connection.prepare(
            "SELECT report FROM runs WHERE fingerprint = ?1 ORDER BY created DESC LIMIT ?2")?;
        let texts = statement.query_map(params![fingerprint, limit as i64], |row| row.get::<_, String>(0))?;
        Ok(texts.filter_map(|t| t.ok()).filter_map(|t| serde_json::from_str(&t).ok()).collect())
    }

    pub fn save_profile(&self, profile: &Profile) -> Result<()> {
        self.connection.execute("INSERT INTO profiles (name, profile) VALUES (?1, ?2)
            ON CONFLICT(name) DO UPDATE SET profile = excluded.profile",
            params![profile.name, serde_json::to_string(profile)?])?;
        Ok(())
    }

    pub fn delete_profile(&self, name: &str) -> Result<bool> {
        Ok(self.connection.execute("DELETE FROM profiles WHERE name = ?1", params![name])? > 0)
    }

    pub fn profiles(&self) -> Result<Vec<Profile>> {
        let mut statement = self.connection.prepare("SELECT profile FROM profiles ORDER BY name")?;
        let texts = statement.query_map([], |row| row.get::<_, String>(0))?;
        Ok(texts.filter_map(|t| t.ok()).filter_map(|t| serde_json::from_str(&t).ok()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{RunStatus, ServerInfo, SCHEMA};

    fn report(id: &str, fingerprint: &str, created: &str) -> Report {
        Report { schema: SCHEMA.into(), id: id.into(), created: created.into(), finished: None,
            status: RunStatus::Done, profile: "share".into(), plan: vec![],
            server: ServerInfo { model: "m".into(), ..ServerInfo::default() }, fingerprint: fingerprint.into(),
            baseline: None, panels: vec![], fabric: vec![], error: None }
    }

    /// The listing beside the dashboard's rendered run derives the verdict the
    /// same way the render itself does: a stored failure of a checkpoint the
    /// rule in force now calls informational is not listed as a failure, the
    /// same numbers from a checkpoint the bar still gates are, and a calibrated
    /// checkpoint with a gate the bar still fails stays a failure.
    #[test]
    fn a_listed_run_reads_its_verdict_under_the_rule_in_force_now() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        for (id, algo, failing) in [("nvfp4", "NVFP4", ""), ("weight_only", "W4A16_NVFP4", ""),
            ("cache", "NVFP4", "cache_exact"), ("spec", "NVFP4", "spec_lossless")] {
            // Each run owns its checkpoint directory: the rule reads the config
            // beside the run, not one a later run overwrote.
            let mut report = crate::sample::qwen_nvfp4_max(&dir.path().join(id), algo, failing);
            report.id = id.into();
            assert!(report.quality_failed(), "the run was stored as a failure");
            store.save(&report).unwrap();
        }
        let rows = store.list(10).unwrap();
        let row = |id: &str| rows.iter().find(|r| r.id == id).unwrap();
        assert!(!row("nvfp4").quality_failed);
        assert!(row("weight_only").quality_failed);
        // A failure beside the reclassified check is still a failure.
        assert!(row("cache").quality_failed);
        assert!(row("spec").quality_failed);
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn runs_and_profiles_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        store.save(&report("a", "f1", "2026-10-01T00:00:00Z")).unwrap();
        store.save(&report("b", "f1", "2026-10-02T00:00:00Z")).unwrap();
        let mut c = report("c", "f2", "2026-10-03T00:00:00Z");
        store.save(&c).unwrap();
        c.status = RunStatus::Cancelled;
        store.save(&c).unwrap();
        assert_eq!(store.list(10).unwrap().iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), vec!["c", "b", "a"]);
        assert_eq!(store.load("c").unwrap().unwrap().status, RunStatus::Cancelled);
        assert_eq!(store.reports_for("f1", 10).unwrap().len(), 2);
        let profile = Profile { fabric: false, name: "mine".into(), title: "Mine".into(), description: String::new(),
            panels: vec![], builtin: false };
        store.save_profile(&profile).unwrap();
        assert_eq!(store.profiles().unwrap(), vec![profile]);
        assert!(store.delete_profile("mine").unwrap());
        drop(store);
        // Reopening keeps the runs.
        assert_eq!(Store::open(dir.path()).unwrap().list(10).unwrap().len(), 3);
    }
}
