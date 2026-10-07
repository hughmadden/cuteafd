//! Bakes the source identity (git remote, commit, dirty) into the binary for
//! report footers. A frozen WIP or release source tree has no `.git`; its
//! `BUILD_IDENTITY.json` (written by `wip.sh`) carries the same facts. At run
//! time release images override all of this with `CUTEAFD_ENGINE_COMMIT` and
//! `CUTEAFD_RELEASE_VERSION`.
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let root = Path::new(&manifest).join("../../..");
    let identity = root.join("BUILD_IDENTITY.json");
    println!("cargo:rerun-if-changed={}", identity.display());
    println!("cargo:rerun-if-env-changed=CUTEAFD_BUILD_COMMIT");
    let (mut remote, mut commit, mut dirty) = (String::new(), String::new(), String::new());
    if let Ok(text) = std::fs::read_to_string(&identity) {
        let field = |name: &str| -> String {
            // A flat JSON object of strings and booleans; no JSON crate in build deps.
            let key = format!("\"{name}\"");
            let Some(at) = text.find(&key) else { return String::new() };
            let rest = text[at + key.len()..].trim_start().trim_start_matches(':').trim_start();
            if let Some(rest) = rest.strip_prefix('"') {
                rest.split('"').next().unwrap_or_default().to_string()
            } else {
                rest.split([',', '}', '\n']).next().unwrap_or_default().trim().to_string()
            }
        };
        (remote, commit, dirty) = (field("remote"), field("commit"), field("dirty"));
    } else if let Some(head) = git(&root, &["rev-parse", "HEAD"]) {
        commit = head;
        remote = git(&root, &["config", "--get", "remote.origin.url"]).unwrap_or_default();
        dirty = git(&root, &["status", "--porcelain", "--untracked-files=no"])
            .map(|s| (!s.is_empty()).to_string()).unwrap_or_default();
        for dir in [git(&root, &["rev-parse", "--git-dir"]), git(&root, &["rev-parse", "--git-common-dir"])]
            .into_iter().flatten() {
            let dir = if Path::new(&dir).is_absolute() { Path::new(&dir).to_path_buf() } else { root.join(&dir) };
            println!("cargo:rerun-if-changed={}", dir.join("HEAD").display());
            println!("cargo:rerun-if-changed={}", dir.join("index").display());
        }
    }
    // The daemon's stamp (wip.sh) when nothing else named the commit.
    let stamp = root.join(".cuteafd-source-revision");
    println!("cargo:rerun-if-changed={}", stamp.display());
    if commit.is_empty() {
        if let Ok(text) = std::fs::read_to_string(&stamp) {
            let mut words = text.split_whitespace();
            commit = words.next().unwrap_or_default().to_string();
            dirty = (words.next() == Some("dirty")).to_string();
        }
    }
    if let Ok(value) = std::env::var("CUTEAFD_BUILD_COMMIT") {
        commit = value;
    }
    agentic_repo(Path::new(&manifest));
    println!("cargo:rustc-env=CUTEAFD_BUILD_REMOTE={remote}");
    println!("cargo:rustc-env=CUTEAFD_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=CUTEAFD_BUILD_DIRTY={dirty}");
}

/// The agentic panel's fixture repository (scripts/fixtures/agentic-repo) compiled in:
/// `$OUT_DIR/agentic_repo.rs` lists (relative path, contents).
fn agentic_repo(manifest: &Path) {
    let root = manifest.join("../../../scripts/fixtures/agentic-repo");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut files = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n != "__pycache__") {
                    stack.push(path);
                }
            } else if path.extension().is_none_or(|e| e != "pyc") {
                println!("cargo:rerun-if-changed={}", path.display());
                files.push(path);
            }
        }
    }
    files.sort();
    let mut out = String::from("pub static AGENTIC_REPO: &[(&str, &str)] = &[\n");
    for file in files {
        let relative = file.strip_prefix(&root).unwrap_or(&file).display().to_string();
        out.push_str(&format!("    ({relative:?}, include_str!({:?})),\n", file.display().to_string()));
    }
    out.push_str("];\n");
    let target = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("agentic_repo.rs");
    std::fs::write(target, out).expect("write agentic_repo.rs");
}
