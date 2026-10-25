#!/usr/bin/env python3
"""Frozen Python dependencies for the release images.

docker/release-python-lock.json names every wheel the release images install on
top of the pinned NGC base, for both architectures: exact version, file name,
HTTPS URL, SHA256 and PyPI upload time, plus the package's own requirements.
Everything else comes from the base image, pinned by digest. Installs are
--no-deps, --no-index, --require-hashes and offline; `uv pip check` afterwards
proves the base plus the lock is a consistent closure.

  lock --pins N==V ... --cutoff T --base-image REF --output LOCK   write a lock (network: PyPI JSON)
  fetch --lock LOCK --wheelhouse DIR                                download missing wheels, hash-checked
  verify --lock LOCK --wheelhouse DIR                               every wheel present with its hash
  requirements --lock LOCK --wheelhouse DIR --arch A [--only N|--exclude N]
                                                                    hash-pinned file:// requirement lines
  verify-installed --lock LOCK                                      installed versions equal the lock
  check-build-requires --pyproject FILE                             build backend satisfied by the env
  inventory                                                         this environment's distributions as JSON
  check-closure --lock LOCK --inventory JSON --arch A               base + lock satisfies every requirement

Only `lock` and `fetch` use the network; `fetch` accepts only files.pythonhosted.org
URLs and never replaces a file that is present with another hash.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

SCHEMA = "cuteafd.release-python-lock/1"
ARCHES = {"amd64": "x86_64", "arm64": "aarch64"}
PYTHON = (3, 12)
MAX_GLIBC = (2, 39)  # Ubuntu 24.04, the NGC 26.05 base
URL_PREFIX = "https://files.pythonhosted.org/packages/"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LockError(Exception):
    pass


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def wheel_rank(filename: str, arch: str) -> tuple | None:
    """Sort key of a wheel usable on CPython 3.12 Linux ARCH (lowest glibc first), else None."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        return None
    pythons, abis, platforms = (part.split(".") for part in parts[-3:])
    major, minor = PYTHON
    py_ok = any(tag in (f"cp{major}{minor}", f"py{major}", f"py{major}{minor}") for tag in pythons)
    abi3 = "abi3" in abis and any(
        re.fullmatch(rf"cp{major}(\d+)", tag) and int(tag[3:]) <= minor for tag in pythons)
    if not ((py_ok and ("none" in abis or f"cp{major}{minor}" in abis)) or abi3):
        return None
    best = None
    for platform in platforms:
        if platform == "any":
            glibc = (0, 0)
        elif platform in (f"manylinux1_{ARCHES[arch]}", f"manylinux2010_{ARCHES[arch]}", f"manylinux2014_{ARCHES[arch]}"):
            glibc = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}[platform.split("_")[0]]
        elif (match := re.fullmatch(rf"manylinux_(\d+)_(\d+)_{ARCHES[arch]}", platform)):
            glibc = (int(match[1]), int(match[2]))
        else:
            continue
        if glibc <= MAX_GLIBC and (best is None or glibc < best):
            best = glibc
    return None if best is None else (best, filename)


def load_lock(path: Path) -> dict:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LockError(f"{path}: cannot read the lock: {error}") from error
    if not isinstance(lock, dict) or lock.get("schema") != SCHEMA:
        raise LockError(f"{path}: not a {SCHEMA} lock")
    arches = lock.get("arches")
    if not isinstance(arches, list) or not arches or not set(arches) <= set(ARCHES):
        raise LockError(f"{path}: arches must be a subset of {sorted(ARCHES)}")
    cutoff = parse_time(lock["cutoff"])
    packages = lock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise LockError(f"{path}: no packages")
    seen = set()
    for package in packages:
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str) or normalize(name) in seen:
            raise LockError(f"{path}: bad or duplicate package entry {package!r:.120}")
        seen.add(normalize(name))
        wheels = package.get("wheels") or {}
        for arch in arches:
            wheel = wheels.get(arch)
            if not isinstance(wheel, dict):
                raise LockError(f"{path}: {name} {version} has no {arch} wheel")
            filename = wheel.get("filename", "")
            stem = filename.split("-")
            if (len(stem) < 5 or normalize(stem[0]) != normalize(name) or stem[1] != version
                    or wheel_rank(filename, arch) is None or "/" in filename):
                raise LockError(f"{path}: {name} {version} {arch} wheel {filename!r} does not fit")
            if not SHA256.match(wheel.get("sha256", "")):
                raise LockError(f"{path}: {filename} has no sha256")
            if not wheel.get("url", "").startswith(URL_PREFIX) or not wheel["url"].endswith("/" + filename):
                raise LockError(f"{path}: {filename} URL is not a files.pythonhosted.org wheel URL")
            if parse_time(wheel["upload_time"]) > cutoff:
                raise LockError(f"{path}: {filename} was uploaded after the lock cutoff {lock['cutoff']}")
    return lock


def lock_wheels(lock: dict, arches=None, only=(), exclude=()):
    only = {normalize(n) for n in only}
    exclude = {normalize(n) for n in exclude}
    for package in lock["packages"]:
        key = normalize(package["name"])
        if (only and key not in only) or key in exclude:
            continue
        for arch in arches or lock["arches"]:
            yield package, arch, package["wheels"][arch]


def checked_wheel(wheelhouse: Path, wheel: dict) -> Path:
    path = wheelhouse / wheel["filename"]
    if not path.is_file():
        raise LockError(f"missing wheel {path}; run release-python-lock.py fetch")
    actual = sha256_file(path)
    if actual != wheel["sha256"]:
        raise LockError(f"{path} has sha256 {actual}, the lock says {wheel['sha256']}")
    return path


def cmd_verify(args) -> None:
    lock = load_lock(args.lock)
    count = 0
    for _, _, wheel in lock_wheels(lock, args.arch or None):
        checked_wheel(args.wheelhouse, wheel)
        count += 1
    print(f"release Python lock: {count} wheels present and hash-checked in {args.wheelhouse}")


def cmd_requirements(args) -> None:
    lock = load_lock(args.lock)
    if args.arch not in lock["arches"]:
        raise LockError(f"the lock has no {args.arch} wheels")
    lines = []
    for package, _, wheel in lock_wheels(lock, [args.arch], args.only, args.exclude):
        path = checked_wheel(args.wheelhouse, wheel)
        lines.append(f"{package['name']} @ file://{path.resolve()} --hash=sha256:{wheel['sha256']}")
    if not lines:
        raise LockError("no packages selected")
    print("\n".join(lines))


def cmd_fetch(args) -> None:
    lock = load_lock(args.lock)
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    for _, _, wheel in lock_wheels(lock, args.arch or None):
        path = args.wheelhouse / wheel["filename"]
        if path.exists():
            checked_wheel(args.wheelhouse, wheel)
            continue
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(wheel["url"], timeout=120) as response, part.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if sha256_file(part) != wheel["sha256"]:
            part.unlink()
            raise LockError(f"{wheel['url']} does not match its locked sha256")
        part.rename(path)
        print(f"fetched {path.name}")


def cmd_verify_installed(args) -> None:
    import importlib.metadata as metadata
    lock = load_lock(args.lock)
    wrong = []
    for package in lock["packages"]:
        try:
            installed = metadata.version(package["name"])
        except metadata.PackageNotFoundError:
            installed = None
        if installed != package["version"]:
            wrong.append(f"{package['name']}: installed {installed}, locked {package['version']}")
    if wrong:
        raise LockError("installed packages differ from the lock: " + "; ".join(wrong))
    print(f"release Python lock: {len(lock['packages'])} installed packages match")


def _installed_satisfies(text: str) -> str | None:
    import importlib.metadata as metadata
    from packaging.requirements import Requirement
    requirement = Requirement(text)
    try:
        version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return f"{text}: not installed"
    if not requirement.specifier.contains(version, prereleases=True):
        return f"{text}: installed {version}"
    return None


def cmd_check_build_requires(args) -> None:
    import tomllib
    requires = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))["build-system"]["requires"]
    problems = [p for p in map(_installed_satisfies, requires) if p]
    if problems:
        raise LockError(f"{args.pyproject} build backend is not satisfied offline: " + "; ".join(problems))
    print(f"build requirements satisfied by the environment: {', '.join(requires)}")


def cmd_inventory(_args) -> None:
    import importlib.metadata as metadata
    out = {}
    for dist in metadata.distributions():  # sys.path order: the first one is the one imported
        key = normalize(dist.metadata["Name"])
        if key not in out:
            out[key] = {"version": dist.version, "installer": (dist.read_text("INSTALLER") or "").strip(),
                        "requires": dist.requires or []}
    print(json.dumps(out, indent=1, sort_keys=True))


def closure_problems(lock: dict, inventory: dict, arch: str, base_exclude=()) -> tuple[list, dict]:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.version import Version
    excluded = {normalize(n) for n in base_exclude}
    env = {**default_environment(), "platform_machine": ARCHES[arch], "sys_platform": "linux",
           "platform_system": "Linux", "os_name": "posix", "implementation_name": "cpython",
           "python_version": "%d.%d" % PYTHON, "python_full_version": "%d.%d.0" % PYTHON}
    dists = {k: {"version": v["version"], "requires": v["requires"]}
             for k, v in inventory.items() if k not in excluded}
    replaced = {}
    for package in lock["packages"]:
        key = normalize(package["name"])
        if key in dists and dists[key]["version"] != package["version"]:
            replaced[key] = f"{dists[key]['version']} -> {package['version']}"
        dists[key] = {"version": package["version"], "requires": package.get("requires") or []}
    problems = []
    for key, dist in sorted(dists.items()):
        for text in dist["requires"]:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({**env, "extra": ""}):
                continue
            found = dists.get(normalize(requirement.name))
            if found is None:
                problems.append(f"{key} {dist['version']} needs {text}: missing")
            elif not requirement.specifier.contains(Version(found["version"]), prereleases=True):
                problems.append(f"{key} {dist['version']} needs {text}: have {found['version']}")
    return problems, replaced


def cmd_check_closure(args) -> None:
    lock = load_lock(args.lock)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    problems, replaced = closure_problems(lock, inventory, args.arch, args.base_exclude)
    for key, change in sorted(replaced.items()):
        print(f"replaces base {key} {change}")
    if problems:
        raise LockError("incomplete closure:\n  " + "\n  ".join(problems))
    print(f"closure complete for {args.arch}: base + {len(lock['packages'])} locked packages")


def cmd_lock(args) -> None:
    cutoff = parse_time(args.cutoff)
    if cutoff > dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7):
        raise LockError("the cutoff must be at least 7 days ago")
    packages = []
    for pin in args.pins:
        name, _, version = pin.partition("==")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
            data = json.load(response)
        if data["info"]["version"] != version:
            raise LockError(f"PyPI returned {data['info']['version']} for {pin}")
        wheels = {}
        for arch in args.arches:
            files = [(wheel_rank(f["filename"], arch), f) for f in data["urls"]
                     if f["packagetype"] == "bdist_wheel" and not f.get("yanked")]
            files = sorted((rank, f) for rank, f in files if rank)
            if not files:
                raise LockError(f"{pin} has no CPython 3.12 Linux {arch} wheel")
            chosen = files[0][1]
            if parse_time(chosen["upload_time_iso_8601"]) > cutoff:
                raise LockError(f"{chosen['filename']} is newer than the cutoff {args.cutoff}")
            wheels[arch] = {"filename": chosen["filename"], "url": chosen["url"],
                            "sha256": chosen["digests"]["sha256"], "upload_time": chosen["upload_time_iso_8601"]}
        packages.append({"name": data["info"]["name"], "version": version,
                         "requires": [r for r in data["info"].get("requires_dist") or []], "wheels": wheels})
    lock = {"schema": SCHEMA, "base_image": args.base_image, "python": "%d.%d" % PYTHON,
            "arches": args.arches, "min_age_days": 7, "cutoff": args.cutoff, "packages": packages}
    args.output.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    load_lock(args.output)
    print(f"wrote {args.output}: {len(packages)} packages")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, *flags):
        command = sub.add_parser(name)
        command.set_defaults(func=func)
        for flag in flags:
            flag(command)
        return command

    lock_flag = lambda c: c.add_argument("--lock", type=Path, required=True)
    house_flag = lambda c: c.add_argument("--wheelhouse", type=Path, required=True)
    arches_flag = lambda c: c.add_argument("--arch", action="append", choices=sorted(ARCHES))
    add("verify", cmd_verify, lock_flag, house_flag, arches_flag)
    add("fetch", cmd_fetch, lock_flag, house_flag, arches_flag)
    requirements = add("requirements", cmd_requirements, lock_flag, house_flag)
    requirements.add_argument("--arch", required=True, choices=sorted(ARCHES))
    requirements.add_argument("--only", action="append", default=[])
    requirements.add_argument("--exclude", action="append", default=[])
    add("verify-installed", cmd_verify_installed, lock_flag)
    add("check-build-requires", cmd_check_build_requires,
        lambda c: c.add_argument("--pyproject", type=Path, required=True))
    add("inventory", cmd_inventory)
    closure = add("check-closure", cmd_check_closure, lock_flag)
    closure.add_argument("--inventory", type=Path, required=True)
    closure.add_argument("--arch", required=True, choices=sorted(ARCHES))
    closure.add_argument("--base-exclude", action="append", default=[],
                         help="inventory entries that are not part of the base image")
    lock = add("lock", cmd_lock)
    lock.add_argument("--pins", nargs="+", required=True)
    lock.add_argument("--cutoff", required=True)
    lock.add_argument("--base-image", required=True)
    lock.add_argument("--arches", nargs="+", default=sorted(ARCHES), choices=sorted(ARCHES))
    lock.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except LockError as error:
        print(f"release-python-lock: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
