//! `cuteafd bench publish`: the root README's results table (the basic
//! profile per family, checkpoint and reference hardware) and the
//! `benchmarks/README.md` index, both rebuilt from the `report.json` files
//! under `benchmarks/<family>/<date>-<profile>-<hardware>/`.
use crate::render::{rate, seconds};
use crate::report::{CheckStatus, HardwareClass, Report};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const RESULTS_BEGIN: &str = "<!-- results:begin -->";
pub const RESULTS_END: &str = "<!-- results:end -->";
pub const INDEX_BEGIN: &str = "<!-- reports:begin -->";
pub const INDEX_END: &str = "<!-- reports:end -->";

/// A report placed for publication.
#[derive(Debug, Clone)]
pub struct Placed {
    /// `benchmarks/<family>/<dir>` relative to the repository root.
    pub dir: PathBuf,
    pub report: Report,
}

pub fn family_title(family: &str) -> &str {
    match family {
        "deepseek_v41" => "DeepSeek V4.1",
        "deepseek_v4" => "DeepSeek V4",
        "glm5" => "GLM 5.3",
        "glm5_flash" => "GLM 5.3 Flash",
        "mimo_v2" => "MiMo V2",
        "qwen4" => "Qwen 3.8",
        other => other,
    }
}

/// Every `benchmarks/*/*/report.json` under `root`.
pub fn scan(root: &Path) -> Result<Vec<Placed>> {
    let mut placed = Vec::new();
    let base = root.join("benchmarks");
    let Ok(families) = std::fs::read_dir(&base) else { return Ok(placed) };
    for family in families.flatten().filter(|e| e.path().is_dir()) {
        for dir in std::fs::read_dir(family.path())?.flatten().filter(|e| e.path().is_dir()) {
            let path = dir.path().join("report.json");
            if !path.is_file() {
                continue;
            }
            let text = std::fs::read_to_string(&path)?;
            let report: Report = serde_json::from_str(&text).with_context(|| format!("{}", path.display()))?;
            let relative = dir.path().strip_prefix(root).unwrap_or(&dir.path()).to_path_buf();
            placed.push(Placed { dir: relative, report });
        }
    }
    placed.sort_by(|a, b| b.report.created.cmp(&a.report.created).then(a.dir.cmp(&b.dir)));
    Ok(placed)
}

fn short_hardware(report: &Report) -> String {
    let hw = &report.server.hardware;
    let mut out = format!("{}× RTX", hw.used_gpus());
    if !hw.sparks.is_empty() {
        out.push_str(&format!(" + {}× Spark", hw.sparks.len()));
    }
    out
}

/// `GLM-5.3-EXL3-K4-v1 (exl3)`: the checkpoint and its routed-expert format.
fn checkpoint(report: &Report) -> String {
    let name = report.server.model.rsplit('/').next().unwrap_or(&report.server.model).to_string();
    let experts: Vec<String> = report.server.configuration.quant.iter().filter(|q| q.group.contains("routed"))
        .flat_map(|q| q.formats.clone()).collect();
    if experts.is_empty() { name } else { format!("{name} ({})", experts.join("+")) }
}

fn link(dir: &Path, file: &str) -> String {
    format!("{}/{file}", dir.display()).replace('\\', "/")
}

/// The root README's results: one table per family, a row per checkpoint ×
/// reference hardware (newest basic profile of each).
pub fn results(placed: &[Placed]) -> String {
    let mut rows: BTreeMap<String, BTreeMap<(String, u8), &Placed>> = BTreeMap::new();
    for p in placed {
        let r = &p.report;
        if r.baseline.is_none() {
            continue;
        }
        let class = match r.server.hardware.class() {
            HardwareClass::Minimum => 0,
            HardwareClass::Maximum => 1,
            HardwareClass::Other => continue,
        };
        let family = r.server.family.clone().unwrap_or_else(|| "unknown".into());
        // `placed` is newest first: keep the first of each key.
        rows.entry(family).or_default().entry((checkpoint(r), class)).or_insert(p);
    }
    if rows.is_empty() {
        return "_Pending the first published run._\n".into();
    }
    let mut out = String::new();
    for (family, entries) in &rows {
        out.push_str(&format!("\n#### {}\n\n", family_title(family)));
        out.push_str("| Checkpoint | Hardware | KV / req | C1 code | prose | JSON | 8K prefill | TTFT | Quality | Run |\n");
        out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
        for ((name, class), p) in entries {
            let r = &p.report;
            let b = r.baseline.as_ref().expect("filtered");
            let decode = |content: &str| b.card.decode_of(content).map_or("—".into(), |d| rate(d.tok_s));
            let (prefill, ttft) = b.card.prefill.as_ref()
                .map_or(("—".into(), "—".into()), |p| (rate(p.tok_s), seconds(p.ttft_s)));
            let quality = match b.quality.status {
                CheckStatus::Fail => format!("⚠ **FAILED** {}", crate::render::bodies::quality_line(b)),
                CheckStatus::Pass => format!("✓ {}", b.quality.badge()),
                _ => b.quality.badge(),
            };
            let hardware = format!("{} ({})", short_hardware(r), if *class == 0 { "min" } else { "max" });
            out.push_str(&format!("| {name} | {hardware} | {} | {} | {} | {} | {prefill} | {ttft} | {quality} | [{} · {}]({}) |\n",
                r.capacity().compact(), decode("code"), decode("prose"), decode("json"), crate::render::date(&r.created),
                r.server.build.label(), link(&p.dir, "report.svg")));
        }
    }
    out.push_str("\ntok/s; C1 decode with thinking off, 8K prefill cold. Quality: logit fidelity against the \
        family golden reference, prefix-cache restore exactness, lossless speculation.\n");
    out
}

/// The benchmarks/README.md index: per family, newest first.
pub fn index(placed: &[Placed]) -> String {
    let mut by_family: BTreeMap<String, Vec<&Placed>> = BTreeMap::new();
    for p in placed {
        by_family.entry(p.report.server.family.clone().unwrap_or_else(|| "unknown".into())).or_default().push(p);
    }
    let mut out = String::new();
    for (family, reports) in by_family {
        out.push_str(&format!("\n## {}\n\n", family_title(&family)));
        for p in reports {
            let r = &p.report;
            let relative = p.dir.strip_prefix("benchmarks").unwrap_or(&p.dir);
            let mut line = format!("- {} · {} · {} · {} · build {} · [report]({})",
                crate::render::date(&r.created), crate::profiles::title_of(&r.profile), r.server.model,
                r.server.hardware.line(), r.server.build.label(), link(relative, "report.svg"));
            if r.quality_failed() {
                line.push_str(" · ⚠ quality gate failed");
            }
            out.push_str(&line);
            out.push('\n');
        }
    }
    if out.is_empty() {
        out.push_str("\n_No reports yet._\n");
    }
    out
}

/// Replaces the text between `begin` and `end` in `text`.
pub fn splice(text: &str, begin: &str, end: &str, body: &str) -> Result<String> {
    let start = text.find(begin).with_context(|| format!("marker {begin} missing"))? + begin.len();
    let stop = text[start..].find(end).with_context(|| format!("marker {end} missing"))? + start;
    Ok(format!("{}\n{}\n\n{}", &text[..start], body.trim_end_matches('\n'), &text[stop..]))
}

const INDEX_HEADER: &str = "# Benchmarks\n\nReports from `cuteafd bench` (profiles other than the basic one run \
    when asked). Each directory holds `report.svg`, `report.json` and the share card; `cuteafd bench publish` \
    rebuilds this index and the root README's table.\n\n";

/// Checks `dirs` are placed for publication and rebuilds both documents.
pub fn publish(root: &Path, dirs: &[PathBuf]) -> Result<(PathBuf, PathBuf, usize)> {
    for dir in dirs {
        let absolute = if dir.is_absolute() { dir.clone() } else { std::env::current_dir()?.join(dir) };
        let base = root.canonicalize()?.join("benchmarks");
        let canonical = absolute.canonicalize().with_context(|| format!("{}", dir.display()))?;
        let Ok(relative) = canonical.strip_prefix(&base) else {
            bail!("{} is not under {}: published reports go in benchmarks/<family>/<date>-<profile>-<hardware>/",
                dir.display(), base.display());
        };
        if relative.components().count() != 2 || !canonical.join("report.json").is_file() {
            bail!("{} must be benchmarks/<family>/<date>-<profile>-<hardware>/ with a report.json", dir.display());
        }
        let report: Report = serde_json::from_str(&std::fs::read_to_string(canonical.join("report.json"))?)?;
        let family = report.server.family.clone().unwrap_or_else(|| "unknown".into());
        let parent = relative.components().next().and_then(|c| c.as_os_str().to_str()).unwrap_or_default();
        if parent != family {
            bail!("{} holds a {family} report; move it under benchmarks/{family}/", dir.display());
        }
    }
    let placed = scan(root)?;
    let readme = root.join("README.md");
    let text = std::fs::read_to_string(&readme).with_context(|| format!("{}", readme.display()))?;
    crate::cli::write_if_changed(&readme, &splice(&text, RESULTS_BEGIN, RESULTS_END, &results(&placed))?)?;
    let index_path = root.join("benchmarks/README.md");
    std::fs::create_dir_all(root.join("benchmarks"))?;
    let current = std::fs::read_to_string(&index_path)
        .unwrap_or_else(|_| format!("{INDEX_HEADER}{INDEX_BEGIN}\n{INDEX_END}\n"));
    crate::cli::write_if_changed(&index_path, &splice(&current, INDEX_BEGIN, INDEX_END, &index(&placed))?)?;
    Ok((readme, index_path, placed.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_rebuilds_the_table_and_index() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("README.md"),
            format!("# x\n\n{RESULTS_BEGIN}\n_Pending._\n{RESULTS_END}\n\nrest\n")).unwrap();
        let mut ok = crate::sample::report(false);
        ok.created = "2026-10-02T10:00:00Z".into();
        let mut older = crate::sample::report(true);
        older.created = "2026-10-01T10:00:00Z".into();
        older.id = "older".into();
        for r in [&ok, &older] {
            let dir = crate::cli::default_dir(root.path(), r);
            let dir = if r.id == "older" { dir.with_file_name("2026-10-01-smoke-1rtx-4spark") } else { dir };
            crate::cli::write_exports(r, &dir, &["json".into(), "svg".into()]).unwrap();
        }
        let dir = root.path().join("benchmarks/deepseek_v41/2026-10-02-smoke-1rtx-4spark");
        let (readme, index, count) = publish(root.path(), &[dir]).unwrap();
        assert_eq!(count, 2);
        let readme = std::fs::read_to_string(readme).unwrap();
        assert!(readme.contains("#### DeepSeek V4.1"), "{readme}");
        // One row per checkpoint × hardware: the newest wins.
        assert_eq!(readme.matches("| DeepSeek-V4.1-Flash (mxfp4) |").count(), 1, "{readme}");
        assert!(readme.contains("benchmarks/deepseek_v41/2026-10-02-smoke-1rtx-4spark/report.svg"));
        assert!(readme.ends_with("rest\n"));
        let index = std::fs::read_to_string(index).unwrap();
        let first = index.find("2026-10-02").unwrap();
        assert!(first < index.find("2026-10-01").unwrap(), "newest first");
        assert!(index.contains("⚠ quality gate failed"));
        // Idempotent.
        let again = publish(root.path(), &[]).unwrap();
        assert_eq!(std::fs::read_to_string(again.0).unwrap(), readme);
        // Misplaced reports are refused.
        let stray = root.path().join("elsewhere");
        crate::cli::write_exports(&ok, &stray, &["json".into()]).unwrap();
        assert!(publish(root.path(), &[stray]).is_err());
    }
}
