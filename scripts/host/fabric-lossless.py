#!/usr/bin/env python3
"""Operator-only, reversible ConnectX policy; invoked by tune.sh, never sudo."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


DOCA_REPO = "https://linux.mellanox.com/public/repo/doca/3.5.0/{distro}/{arch}/"
DOCA_KEY_FINGERPRINT = "70454E6BF5D9C2D3B78709E8DC726C5E41B9CC50"
DOCA_KEYRING = Path("/usr/share/keyrings/nvidia-doca.gpg")
DOCA_SOURCES = Path("/etc/apt/sources.list.d/nvidia-doca.sources")
DOCA_PIN = Path("/etc/apt/preferences.d/nvidia-doca")
# Only mlnx-tools comes from DOCA; its rdma-core/libibverbs must never replace the distro stack.
DOCA_PIN_TEXT = ("Package: *\nPin: origin linux.mellanox.com\nPin-Priority: 1\n\n"
                 "Package: mlnx-tools\nPin: origin linux.mellanox.com\nPin-Priority: 500\n")


def command(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def doca_repo():
    """NVIDIA's signed DOCA apt repo for this host (Sparks ship their own NVIDIA repo)."""
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().split("\n") if "=" in line)
    distro = release.get("ID", "").strip('"') + release.get("VERSION_ID", "").strip('"')
    arch = {"x86_64": "x86_64", "aarch64": "arm64-sbsa"}[os.uname().machine]
    return DOCA_REPO.format(distro=distro, arch=arch)


def install_tools(mutate, dry_run):
    """Install mlnx_qos (mlnx-tools) from the pinned DOCA repo, verifying its signing key."""
    if subprocess.run(["sh", "-c", "command -v mlnx_qos"], capture_output=True).returncode == 0:
        print("mlnx_qos present; nothing to install")
        return
    repo = doca_repo()
    if not dry_run:
        key = subprocess.check_output(["curl", "-fsSL", repo + "doca_keyring.gpg"], timeout=60)
        found = subprocess.run(["gpg", "--show-keys", "--with-colons"], input=key,
                               capture_output=True, check=True).stdout.decode()
        if f"fpr:::::::::{DOCA_KEY_FINGERPRINT}:" not in found:
            raise ValueError(f"DOCA signing key at {repo} is not {DOCA_KEY_FINGERPRINT}; refusing")
        DOCA_KEYRING.write_bytes(key)
        DOCA_PIN.write_text(DOCA_PIN_TEXT)
        DOCA_SOURCES.write_text(f"Types: deb\nURIs: {repo}\nSuites: ./\nSigned-By: {DOCA_KEYRING}\n")
    else:
        print(f"DRY-RUN verify {repo}doca_keyring.gpg is {DOCA_KEY_FINGERPRINT}; write {DOCA_KEYRING}, {DOCA_SOURCES}, {DOCA_PIN} (only mlnx-tools pinned in)")
    mutate("apt-get", "update")
    mutate("apt-get", "install", "-y", "--no-install-recommends", "mlnx-tools")


def parse_qos(text):
    def match(pattern):
        found = re.search(pattern, text, re.M)
        if not found:
            raise ValueError(f"unsupported mlnx_qos output: {pattern}")
        return found.group(1)

    state = {
        "trust": match(r"^Priority trust state: (pcp|dscp)$"),
        "pfc": match(r"^\s*enabled\s+([01](?:\s+[01]){7})\s*$").split(),
        "buffers": match(r"^\s*buffer\s+([0-7](?:\s+[0-7]){7})\s*$").split(),
        "sizes": match(r"^Receive buffer size \(bytes\): ([0-9,]+)").rstrip(",").split(","),
    }
    if len(state["sizes"]) != 8 or "DCBX mode: OS controlled" not in text:
        raise ValueError("need eight receive buffers and OS-controlled DCBX; do not change DCBX implicitly")
    for tc in (3, 6):
        if not re.search(rf"^tc: {tc} .*\n\s*priority:\s+{tc}\s*$", text, re.M):
            raise ValueError(f"priority {tc} must already have its own TC {tc}")
    return state


def inventory(sysfs):
    ports = []
    for port in sorted((sysfs / "class/infiniband").glob("*/ports/*")):
        if not (port / "state").read_text().strip().endswith("ACTIVE"):
            continue
        if (port / "link_layer").read_text().strip() != "Ethernet":
            continue
        dev = port.parent.parent.name
        for entry in sorted((port / "gid_attrs/ndevs").glob("*")):
            netdev = entry.read_text().strip()
            if not re.fullmatch(r"[a-zA-Z0-9_.:-]+", netdev):
                raise ValueError("invalid netdev")
            addresses = json.loads((sysfs / "class/net" / netdev / "ip.json").read_text()
                                   if sysfs != Path("/sys") else
                                   command("ip", "-j", "-4", "addr", "show", "dev", netdev))
            if any(a.get("local", "").startswith(("10.55.0.", "10.55.1."))
                   for link in addresses for a in link.get("addr_info", [])):
                ports.append((netdev, dev, port.name))
                break
    ports = sorted(set(ports))
    if not ports:
        raise ValueError("no active Ethernet RDMA ports with cluster fabric IPv4 addresses")
    if len({n for n, _, _ in ports}) != len(ports):
        raise ValueError("multiple RDMA ports share one netdev; review bond topology manually")
    return ports


def pending_backups(state_dir):
    return sorted(p for p in state_dir.glob("*.json") if not re.search(r"\.restored\.[0-9]+\.json$", p.name))


def rollback_plan(sysfs, state_dir):
    planned = []
    for backup in pending_backups(state_dir):
        if backup.is_symlink():
            raise ValueError(f"refusing symlink backup {backup}")
        saved = json.loads(backup.read_text())
        try:
            netdev, rdma = saved["netdev"], saved["rdma"]
            valid_name = lambda value: isinstance(value, str) and bool(re.fullmatch(r"[a-zA-Z0-9_:-][a-zA-Z0-9_.:-]*", value))
            if not valid_name(netdev) or backup.name != netdev + ".json":
                raise ValueError("invalid saved netdev")
            if not isinstance(rdma, list) or len(rdma) != 2 or not valid_name(rdma[0]) or not re.fullmatch(r"[1-9][0-9]*", rdma[1]):
                raise ValueError("invalid saved RDMA identity")
            for key, limit in (("pfc", 1), ("buffers", 7), ("sizes", 2**32 - 1)):
                values = saved[key]
                if not isinstance(values, list) or len(values) != 8 or any(
                        not isinstance(v, str) or not v.isascii() or not v.isdecimal() or int(v) > limit for v in values):
                    raise ValueError(f"invalid saved {key}")
            if saved["trust"] not in ("pcp", "dscp"):
                raise ValueError("invalid saved trust")
            if any(saved["pause"][key] not in ("on", "off") for key in ("autoneg", "rx", "tx")):
                raise ValueError("invalid saved pause")
            pairs = saved["app"].get("dscp_prio", [])
            if not isinstance(pairs, list) or any(
                    not isinstance(pair, list) or len(pair) != 2 or
                    any(type(v) is not int for v in pair) or
                    not (0 <= pair[0] <= 63 and 0 <= pair[1] <= 7) for pair in pairs):
                raise ValueError("invalid saved DSCP mappings")
            dev, port = rdma
            device = sysfs / "class/infiniband" / dev
            # device/net survives link-down and address removal; no active GID required.
            if not (device / "device/net" / netdev).exists() or not (device / "ports" / port).is_dir():
                raise ValueError("saved NIC identity does not match sysfs device/net")
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"malformed backup {backup}: {error}") from error
        planned.append((netdev, saved, backup))
    return planned


def restore_backups(planned, mutate, probe, dry_run):
    for netdev, saved, backup in planned:
        mutate("mlnx_qos", "-i", netdev, "--pfc", "0,0,0,0,0,0,0,0")
        mutate("mlnx_qos", "-i", netdev, "--prio2buffer", ",".join(saved["buffers"]))
        mutate("mlnx_qos", "-i", netdev, "--buffer_size", ",".join(saved["sizes"]))
        mutate("dcb", "app", "flush", "dev", netdev, "dscp-prio")
        pairs = saved["app"].get("dscp_prio", [])
        if pairs:
            mutate("dcb", "app", "add", "dev", netdev, "dscp-prio", *[f"{d}:{p}" for d, p in pairs])
        mutate("mlnx_qos", "-i", netdev, "--trust", saved["trust"])
        mutate("mlnx_qos", "-i", netdev, "--pfc", ",".join(saved["pfc"]))
        pause = saved["pause"]
        mutate("ethtool", "-A", netdev, "autoneg", pause["autoneg"], "rx", pause["rx"], "tx", pause["tx"])
        if not dry_run:
            print(probe("mlnx_qos", "-i", netdev))
            receipt = backup.with_name(f"{netdev}.restored.{time.time_ns()}.json")
            backup.rename(receipt)
            print(f"restored-state receipt: {receipt}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tools", "apply", "check", "rollback"))
    parser.add_argument("--role", required=True, choices=("coordinator", "spark"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sysfs = Path(os.environ.get("TUNE_SYSFS_ROOT", "/sys"))
    fixture = sysfs != Path("/sys")
    state_dir = sysfs / "var/lib/cuteafd/fabric-lossless" if fixture else Path("/var/lib/cuteafd/fabric-lossless")
    if args.action != "check" and not args.dry_run and os.geteuid() != 0:
        parser.error("apply/rollback requires root; TJ runs agent-sudo after review")
    if fixture and not args.dry_run and args.action != "check":
        parser.error("fixture writes are forbidden; use --dry-run")
    if args.dry_run and args.action == "check":
        parser.error("check is already read-only")

    def mutate(*argv):
        print(("DRY-RUN " if args.dry_run else "") + "command " + " ".join(argv), flush=True)
        if not args.dry_run:
            command(*argv)

    if args.action == "tools" or (args.action == "apply" and not fixture):
        install_tools(mutate, args.dry_run)
        if args.action == "tools":
            return
    if args.action == "rollback":
        planned = rollback_plan(sysfs, state_dir)
        if not planned:
            print("no pending NIC backups; nothing to restore")
        restore_backups(planned, mutate, command, args.dry_run)
        return
    if args.action == "apply" and pending_backups(state_dir):
        raise ValueError("pending NIC backups exist; complete rollback before another apply")
    planned = []
    for netdev, dev, port in inventory(sysfs):
        nic = sysfs / "class/net" / netdev
        text = (nic / "mlnx_qos.txt").read_text() if fixture else command("mlnx_qos", "-i", netdev)
        app = json.loads((nic / "app.json").read_text() if fixture else
                         command("dcb", "-j", "-N", "app", "show", "dev", netdev))
        print(f"role={args.role} netdev={netdev} rdma={dev}/{port}\n{text}\nAPP={json.dumps(app, sort_keys=True)}")
        tc = sysfs / "class/infiniband" / dev / "tc" / port / "traffic_class"
        print(f"{tc}=" + (tc.read_text().strip() if tc.exists() else "absent") +
              "; engine RDMA_TRAFFIC_CLASS=104 is required (no global sysfs override)")
        pause_text = (nic / "pause.txt").read_text() if fixture else command("ethtool", "-a", netdev)
        print(pause_text)
        if args.action == "check":
            continue
        current = parse_qos(text)
        if (sysfs / "class/net" / netdev / "qos/dscp2prio").exists():
            raise ValueError("legacy sysfs DSCP mapping needs a separate reviewed restore; refusing")
        backup = state_dir / (netdev + ".json")
        if args.action == "apply":
            if backup.exists():
                raise ValueError(f"{backup} exists; check or rollback first (never overwrite original state)")
            if any(v != "0" for v in current["pfc"]):
                raise ValueError("existing PFC policy needs manual review")
            if "dscp_prio" in app and not isinstance(app["dscp_prio"], list):
                raise ValueError("unsupported dcb app JSON")
            # Split existing headroom, not shared device memory. Driver calculates
            # speed/MTU/cable-length XOFF and rejects an undersized lossless half.
            half = sum(map(int, current["sizes"])) // 2048 * 1024
            if half < 131072:
                raise ValueError("insufficient existing receive headroom; review NIC buffer budget")
            pause = {}
            for label, key in [("Autonegotiate", "autoneg"), ("RX", "rx"), ("TX", "tx")]:
                found = re.search(rf"^{label}:\s+(on|off)\s*$", pause_text, re.M)
                if not found:
                    raise ValueError("cannot snapshot global pause policy")
                pause[key] = found.group(1)
            current.update(netdev=netdev, rdma=[dev, port], app=app, half=half, pause=pause)
            planned.append((netdev, current, backup))
    # Preflight every NIC before the first change or backup; failures remain visible.
    if args.action == "apply" and not args.dry_run:
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for _, saved, backup in planned:
            with backup.open("x") as f:
                json.dump(saved, f, indent=2)
                f.write("\n")
    for netdev, saved, backup in planned:
        if args.action == "apply":
            print(f"original-state backup: {backup}")
            half = str(saved["half"])
            mutate("ethtool", "-A", netdev, "autoneg", "off", "rx", "off", "tx", "off")
            mutate("mlnx_qos", "-i", netdev, "--buffer_size", ",".join([half, half] + ["0"] * 6))
            mutate("mlnx_qos", "-i", netdev, "--prio2buffer", "0,0,0,1,0,0,0,0")
            # Remove conflicting DSCP assignments; leave EtherType/port APPs alone.
            mutate("dcb", "app", "flush", "dev", netdev, "dscp-prio")
            mappings = ["--dscp2prio=set,%d,%d" % (dscp, 3 if dscp == 26 else 6 if dscp == 48 else 0)
                        for dscp in range(64)]
            mutate("mlnx_qos", "-i", netdev, *mappings)
            # mlnx_qos explicitly forbids combining trust with dscp2prio.
            mutate("mlnx_qos", "-i", netdev, "--trust", "dscp")
            mutate("mlnx_qos", "-i", netdev, "--pfc", "0,0,0,1,0,0,0,0")
        if not args.dry_run:
            print(command("mlnx_qos", "-i", netdev))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"fabric-lossless: {error}; stop and inspect, use saved rollback if apply was partial")
