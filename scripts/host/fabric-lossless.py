#!/usr/bin/env python3
"""Operator-only, reversible ConnectX policy; invoked by tune.sh, never sudo."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "check", "rollback"))
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
        if args.action == "rollback":
            saved = json.loads(backup.read_text())
            if saved["netdev"] != netdev or saved["rdma"] != [dev, port]:
                raise ValueError("saved NIC identity does not match")
            planned.append((netdev, saved, backup))
        else:
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
        else:
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
            if not args.dry_run:
                backup.unlink()
        if not args.dry_run:
            print(command("mlnx_qos", "-i", netdev))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"fabric-lossless: {error}; stop and inspect, use saved rollback if apply was partial")
