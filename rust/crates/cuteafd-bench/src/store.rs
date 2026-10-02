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

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join("bench.sqlite");
        let connection = Connection::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let store = Self { connection, path: Some(path) };
        store.migrate()?;
        Ok(store)
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

    pub fn list(&self, limit: usize) -> Result<Vec<RunRow>> {
        let mut statement = self.connection.prepare(
            "SELECT id, created, fingerprint, profile, status, model, hardware, quality_failed FROM runs
             ORDER BY created DESC LIMIT ?1")?;
        let rows = statement.query_map(params![limit as i64], |row| Ok(RunRow {
            id: row.get(0)?, created: row.get(1)?, fingerprint: row.get(2)?, profile: row.get(3)?,
            status: row.get(4)?, model: row.get(5)?, hardware: row.get(6)?, quality_failed: row.get::<_, i64>(7)? != 0,
        }))?;
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
            baseline: None, panels: vec![], error: None }
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
        let profile = Profile { name: "mine".into(), title: "Mine".into(), description: String::new(),
            panels: vec![], builtin: false };
        store.save_profile(&profile).unwrap();
        assert_eq!(store.profiles().unwrap(), vec![profile]);
        assert!(store.delete_profile("mine").unwrap());
        drop(store);
        // Reopening keeps the runs.
        assert_eq!(Store::open(dir.path()).unwrap().list(10).unwrap().len(), 3);
    }
}
