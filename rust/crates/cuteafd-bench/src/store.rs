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
    /// Panels with results in this run (stored column; comma-separated ids).
    pub panels: String,
}

/// Version of the quality rule the stored `quality_failed` column was derived
/// under. Bump it whenever `quality::under_current_rule` changes: opening the
/// store then re-derives every row once (a backfill), and listings read only
/// the stored columns, never the report JSON. 2: summaries of runs saved by
/// a build without these columns after an earlier backfill are re-derived.
pub const VERDICT_RULE: i64 = 2;

/// The summary columns a run's report earns under the rule in force now.
struct Summary { quality_failed: bool, panels: String, has_baseline: bool }
fn summarize(report: &Report) -> Summary {
    let current = crate::quality::under_current_rule(report);
    let panels = report.panels.iter()
        // Results, and attempts that ended with a reason (the merged view shows
        // the newest attempt, so a failure is never hidden behind an older pass).
        .filter(|p| p.status == crate::report::PanelStatus::Done || !p.passes.is_empty() || p.error.is_some())
        .map(|p| p.id.as_str()).collect::<Vec<_>>().join(",");
    Summary { quality_failed: current.quality_failed(), panels, has_baseline: report.baseline.is_some() }
}

/// Where each panel of an accumulated view comes from: the newest run on the
/// fingerprint holding it, and the newest run holding a baseline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sources {
    pub baseline: Option<String>,
    /// (panel id, run id), newest first by panel's run.
    pub panels: Vec<(String, String)>,
    /// The newest run on the fingerprint (server info, fabric).
    pub newest: Option<String>,
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
             CREATE INDEX IF NOT EXISTS runs_created ON runs(created);
             CREATE TABLE IF NOT EXISTS profiles (name TEXT PRIMARY KEY, profile TEXT NOT NULL);")?;
        let columns: Vec<String> = self.connection.prepare("PRAGMA table_info(runs)")?
            .query_map([], |row| row.get::<_, String>(1))?.collect::<std::result::Result<_, _>>()?;
        for (name, kind) in [("verdict_rule", "INTEGER NOT NULL DEFAULT 0"), ("panels", "TEXT NOT NULL DEFAULT ''"),
            ("has_baseline", "INTEGER NOT NULL DEFAULT 0")] {
            if !columns.iter().any(|c| c == name) {
                self.connection.execute_batch(&format!("ALTER TABLE runs ADD COLUMN {name} {kind}"))?;
            }
        }
        // Covering indexes: the run list and an accumulated view's sources read
        // only these columns, never the row's report (which sits in overflow
        // pages, so a table scan for a later column pages the whole store in).
        self.connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS runs_listing ON runs(created, id, fingerprint, profile, status, model, hardware, quality_failed, panels);
             CREATE INDEX IF NOT EXISTS runs_sources ON runs(fingerprint, created, id, panels, has_baseline);
             CREATE INDEX IF NOT EXISTS runs_rule ON runs(verdict_rule);")?;
        Ok(())
    }

    /// Whether some rows still carry summaries from an older rule.
    pub fn stale(&self) -> Result<bool> {
        Ok(self.connection.query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE verdict_rule != ?1)",
            params![VERDICT_RULE], |row| row.get::<_, i64>(0))? != 0)
    }

    /// Re-derive the summary columns of rows stored under an older rule, once,
    /// in short transactions (a concurrent save never waits long). Off the
    /// serving path: the server runs it on a background thread after start.
    pub fn backfill(&self) -> Result<usize> {
        let stale: Vec<String> = self.connection.prepare("SELECT id FROM runs WHERE verdict_rule != ?1")?
            .query_map(params![VERDICT_RULE], |row| row.get(0))?.collect::<std::result::Result<_, _>>()?;
        if stale.is_empty() { return Ok(0); }
        let started = std::time::Instant::now();
        for chunk in stale.chunks(32) {
        let transaction = self.connection.unchecked_transaction()?;
        for id in chunk {
            let text: String = transaction.query_row("SELECT report FROM runs WHERE id = ?1", params![id], |row| row.get(0))?;
            // A report this build cannot parse keeps the verdict it was saved with.
            let report = serde_json::from_str::<Report>(&text).ok();
            // A run still going (perhaps under an older build that will save it
            // again without these columns) stays stale until it has finished.
            let rule = if report.as_ref().is_some_and(|r| !r.status.finished()) { 0 } else { VERDICT_RULE };
            match report.map(|r| summarize(&r)) {
                Some(s) => transaction.execute("UPDATE runs SET quality_failed = ?2, panels = ?3, has_baseline = ?4, verdict_rule = ?5 WHERE id = ?1",
                    params![id, s.quality_failed as i64, s.panels, s.has_baseline as i64, rule])?,
                None => transaction.execute("UPDATE runs SET verdict_rule = ?2 WHERE id = ?1", params![id, VERDICT_RULE])?,
            };
        }
        transaction.commit()?;
        }
        tracing::info!(rows = stale.len(), seconds = started.elapsed().as_secs_f64(), "bench store: run summaries re-derived");
        Ok(stale.len())
    }

    pub fn save(&self, report: &Report) -> Result<()> {
        // The verdict is stored as the record of what this run was measured
        // as; a listing re-derives it from the report under the rule in force
        // then, so the column is only the fallback for a report that will not
        // parse.
        let status = serde_json::to_value(report.status)?.as_str().unwrap_or("unknown").to_string();
        let summary = summarize(report);
        self.connection.execute(
            "INSERT INTO runs (id, created, fingerprint, profile, status, model, hardware, quality_failed, report,
                verdict_rule, panels, has_baseline)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(id) DO UPDATE SET status = excluded.status, quality_failed = excluded.quality_failed,
                report = excluded.report, verdict_rule = excluded.verdict_rule, panels = excluded.panels,
                has_baseline = excluded.has_baseline",
            params![report.id, report.created, report.fingerprint, report.profile, status, report.server.model,
                report.server.hardware.line(), summary.quality_failed as i64, serde_json::to_string(report)?,
                VERDICT_RULE, summary.panels, summary.has_baseline as i64])?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Option<Report>> {
        let text: Option<String> = self.connection
            .query_row("SELECT report FROM runs WHERE id = ?1", params![id], |row| row.get(0)).optional()?;
        text.map(|t| serde_json::from_str(&t).context("stored report")).transpose()
    }

    /// The run records, newest first, from the stored summary columns only.
    ///
    /// `quality_failed` is the verdict under the rule in force now: derived
    /// when the run is saved and re-derived by the one-time backfill whenever
    /// [`VERDICT_RULE`] moves, so a run measured before its checkpoint was
    /// reclassified is not listed as a failure.
    pub fn list(&self, limit: usize) -> Result<Vec<RunRow>> {
        let mut statement = self.connection.prepare(
            "SELECT id, created, fingerprint, profile, status, model, hardware, quality_failed, panels FROM runs
             ORDER BY created DESC LIMIT ?1")?;
        let rows = statement.query_map(params![limit as i64], |row| {
            Ok(RunRow { id: row.get(0)?, created: row.get(1)?, fingerprint: row.get(2)?, profile: row.get(3)?,
                status: row.get(4)?, model: row.get(5)?, hardware: row.get(6)?,
                quality_failed: row.get::<_, i64>(7)? != 0, panels: row.get(8)? })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// The runs an accumulated view of `fingerprint` reads, from stored columns.
    pub fn sources(&self, fingerprint: &str) -> Result<Sources> {
        let mut statement = self.connection.prepare(
            "SELECT id, panels, has_baseline FROM runs WHERE fingerprint = ?1 ORDER BY created DESC")?;
        let rows = statement.query_map(params![fingerprint], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?,
            row.get::<_, i64>(2)? != 0)))?;
        let mut sources = Sources::default();
        for row in rows {
            let (id, panels, baseline) = row?;
            sources.newest.get_or_insert_with(|| id.clone());
            if baseline && sources.baseline.is_none() { sources.baseline = Some(id.clone()); }
            for panel in panels.split(',').filter(|p| !p.is_empty()) {
                if !sources.panels.iter().any(|(p, _)| p == panel) { sources.panels.push((panel.into(), id.clone())); }
            }
        }
        Ok(sources)
    }

    /// Panel `panel`'s stored results on `fingerprint`, newest run first, at
    /// most `limit` runs: SQLite extracts the one panel object from each run
    /// that lists it, so neither the other panels nor other runs are parsed.
    pub fn panel_history(&self, fingerprint: &str, panel: &str, limit: usize) -> Result<Vec<crate::report::PanelResult>> {
        let mut statement = self.connection.prepare(
            "SELECT p.value FROM (SELECT report, created FROM runs WHERE fingerprint = ?1 AND (',' || panels || ',') LIKE ?2
                 ORDER BY created DESC LIMIT ?3) r, json_each(r.report, '$.panels') p
             WHERE json_extract(p.value, '$.id') = ?4 ORDER BY r.created DESC")?;
        let pattern = format!("%,{panel},%");
        let texts = statement.query_map(params![fingerprint, pattern, limit as i64, panel], |row| row.get::<_, String>(0))?;
        Ok(texts.filter_map(|t| t.ok()).filter_map(|t| serde_json::from_str(&t).ok()).collect())
    }

    /// One panel of one stored run, extracted by SQLite (comparisons read this).
    pub fn panel(&self, id: &str, panel: &str) -> Result<Option<crate::report::PanelResult>> {
        let text: Option<String> = self.connection.query_row(
            "SELECT p.value FROM runs r, json_each(r.report, '$.panels') p WHERE r.id = ?1 AND json_extract(p.value, '$.id') = ?2",
            params![id, panel], |row| row.get(0)).optional()?;
        text.map(|t| serde_json::from_str(&t).context("stored panel")).transpose()
    }

    /// The newest stored baseline on `fingerprint`, extracted by SQLite.
    pub fn latest_baseline(&self, fingerprint: &str) -> Result<Option<crate::report::Baseline>> {
        let text: Option<String> = self.connection.query_row(
            "SELECT json_extract(report, '$.baseline') FROM runs WHERE fingerprint = ?1 AND has_baseline = 1
             ORDER BY created DESC LIMIT 1", params![fingerprint], |row| row.get(0)).optional()?.flatten();
        text.map(|t| serde_json::from_str(&t).context("stored baseline")).transpose()
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

    /// A row stored under an older rule (or before the summary columns) is
    /// re-derived once on open; list() never reads the report again.
    #[test]
    fn backfill_rederives_old_rows_once_and_list_reads_columns_only() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let mut report = crate::sample::qwen_nvfp4_max(&dir.path().join("nvfp4"), "NVFP4", "");
        report.id = "old".into();
        store.save(&report).unwrap();
        // As written by an older build: stored failure, no summary columns.
        store.connection.execute("UPDATE runs SET quality_failed = 1, verdict_rule = 0, panels = '', has_baseline = 0", []).unwrap();
        drop(store);
        let store = Store::open(dir.path()).unwrap();
        assert!(store.stale().unwrap());
        assert_eq!(store.backfill().unwrap(), 1);
        assert!(!store.stale().unwrap());
        let row = store.list(10).unwrap().into_iter().next().unwrap();
        assert!(!row.quality_failed, "re-derived under the rule in force now");
        // A list never parses report JSON: corrupt it and the listing still works.
        store.connection.execute("UPDATE runs SET report = 'not json'", []).unwrap();
        assert_eq!(store.list(10).unwrap().len(), 1);
        let rule: i64 = store.connection.query_row("SELECT verdict_rule FROM runs", [], |r| r.get(0)).unwrap();
        assert_eq!(rule, VERDICT_RULE);
    }

    #[test]
    fn sources_pick_the_newest_run_per_panel_and_baseline() {
        use crate::report::{PanelResult, PanelStatus};
        let store = Store::memory().unwrap();
        let panel = |id: &str| PanelResult { id: id.into(), status: PanelStatus::Done, passes: vec![serde_json::json!({})], ..Default::default() };
        let mut a = report("a", "f", "2026-10-01T00:00:00Z"); a.panels = vec![panel("speed"), panel("tools")];
        let mut b = report("b", "f", "2026-10-02T00:00:00Z"); b.panels = vec![panel("speed")];
        let c = report("c", "f", "2026-10-03T00:00:00Z");
        for r in [&a, &b, &c] { store.save(r).unwrap(); }
        let sources = store.sources("f").unwrap();
        assert_eq!(sources.newest.as_deref(), Some("c"));
        assert_eq!(sources.panels, vec![("speed".into(), "b".into()), ("tools".into(), "a".into())]);
        assert!(sources.baseline.is_none());
    }

    /// Manual timing on a copy of a real store: `CUTEAFD_BENCH_STORE_COPY=DIR cargo test -- --ignored store_copy`.
    #[test]
    #[ignore]
    fn store_copy_backfill_and_list_timing() {
        let Some(dir) = std::env::var_os("CUTEAFD_BENCH_STORE_COPY") else { return };
        let started = std::time::Instant::now();
        let store = Store::open(Path::new(&dir)).unwrap();
        eprintln!("open: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        let started = std::time::Instant::now();
        eprintln!("backfill: {} rows in {:.2} s", store.backfill().unwrap(), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let rows = store.list(200).unwrap();
        eprintln!("list(200): {} rows in {:.1} ms", rows.len(), started.elapsed().as_secs_f64() * 1e3);
        let started = std::time::Instant::now();
        let sources = store.sources(&rows[0].fingerprint).unwrap();
        eprintln!("sources: {} panels in {:.1} ms", sources.panels.len(), started.elapsed().as_secs_f64() * 1e3);
        let started = std::time::Instant::now();
        drop(store);
        let store = Store::open(Path::new(&dir)).unwrap();
        eprintln!("reopen (no backfill): {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        drop(store);
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
        // History and baselines come out of SQLite one panel at a time.
        let mut d = report("d", "f1", "2026-10-04T00:00:00Z");
        let panel = |id: &str, passes: Vec<serde_json::Value>| -> crate::report::PanelResult { serde_json::from_value(serde_json::json!(
            {"id": id, "title": id, "status": if passes.is_empty() { "pending" } else { "done" }, "passes": passes})).unwrap() };
        d.panels = vec![panel("code", vec![]), panel("math", vec![serde_json::json!({"correct": 3})])];
        d.baseline = crate::sample::report(false).baseline;
        store.save(&d).unwrap();
        let history = store.panel_history("f1", "math", 10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].passes[0]["correct"], 3);
        assert!(store.panel_history("f1", "code", 10).unwrap().is_empty(), "pending, no passes: not listed");
        assert!(store.panel_history("f2", "math", 10).unwrap().is_empty());
        assert!(store.latest_baseline("f1").unwrap().is_some());
        assert_eq!(store.panel("d", "math").unwrap().unwrap().passes.len(), 1);
        assert!(store.panel("d", "nope").unwrap().is_none() && store.panel("zz", "math").unwrap().is_none());
        assert!(store.latest_baseline("f2").unwrap().is_none());
        let profile = Profile { fabric: false, name: "mine".into(), title: "Mine".into(), description: String::new(),
            panels: vec![], builtin: false };
        store.save_profile(&profile).unwrap();
        assert_eq!(store.profiles().unwrap(), vec![profile]);
        assert!(store.delete_profile("mine").unwrap());
        drop(store);
        // Reopening keeps the runs.
        assert_eq!(Store::open(dir.path()).unwrap().list(10).unwrap().len(), 4);
    }
}
