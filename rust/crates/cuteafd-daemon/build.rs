//! Records the engine source revision for the live console's header:
//! `CUTEAFD_BUILD_COMMIT` and `CUTEAFD_BUILD_DIRTY` (1 when tracked files had
//! uncommitted changes). Git checkouts are asked directly; WIP slot copies
//! (no `.git`) carry `.cuteafd-source-revision` written by `wip.sh`
//! (`<commit> [dirty]`). Release images also set `CUTEAFD_ENGINE_COMMIT` at run time.
use std::path::Path;
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest).join("../../..");
    let stamp = root.join(".cuteafd-source-revision");
    println!("cargo:rerun-if-changed={}", stamp.display());
    // Any crate source edit can change the dirty state.
    println!("cargo:rerun-if-changed={}", Path::new(&manifest).join("..").display());
    let (commit, dirty) = match git(&root, &["rev-parse", "HEAD"]) {
        Some(commit) => {
            for path in ["HEAD", "index", "logs/HEAD"] {
                if let Some(path) = git(&root, &["rev-parse", "--path-format=absolute", "--git-path", path]) {
                    println!("cargo:rerun-if-changed={path}");
                }
            }
            let dirty = git(&root, &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
            (commit, dirty)
        }
        None => match std::fs::read_to_string(&stamp) {
            Ok(text) => {
                let mut words = text.split_whitespace();
                (words.next().unwrap_or_default().to_string(), words.next() == Some("dirty"))
            }
            Err(_) => (String::new(), false),
        },
    };
    let commit = if commit.trim().is_empty() { "unknown" } else { &commit };
    println!("cargo:rustc-env=CUTEAFD_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=CUTEAFD_BUILD_DIRTY={}", u8::from(dirty));
}
