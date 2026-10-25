#!/usr/bin/env python3
"""Address of a build seed that is not on the RoCE fabric.

  seed-transfer.py lan-addr HOST

Prints HOST's IPv4 address on a subnet this host is directly attached to,
ignoring loopback, tailscale, docker and other virtual bridges. A LAN-only seed
(kakapo, cassowary: 10 GbE only) resolves by name to its tailscale address, a slower route
than the shared LAN, so bulk transfers to it use this address instead. Nothing
here is cluster-specific: both sides report their own interfaces (`ip -4 -o
addr show scope global`). Exit 1 when no shared subnet exists.
"""
from __future__ import annotations

import ipaddress
import re
import subprocess
import sys

VIRTUAL = re.compile(r'^(lo|docker|br-|veth|tailscale|kube|virbr|cni|flannel|wg|zt|tun|tap)')
CGNAT = ipaddress.ip_network('100.64.0.0/10')


def interfaces(text: str) -> list[ipaddress.IPv4Interface]:
    """Physical global IPv4 interfaces from `ip -4 -o addr show scope global`."""
    found = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != 'inet' or VIRTUAL.match(fields[1]):
            continue
        address = ipaddress.ip_interface(fields[3])
        if address.ip not in CGNAT:
            found.append(address)
    return found


def lan_addr(local: str, remote: str) -> str | None:
    networks = [address.network for address in interfaces(local)]
    for address in interfaces(remote):
        if any(address.ip in network for network in networks):
            return str(address.ip)
    return None


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != 'lan-addr':
        print(__doc__, file=sys.stderr)
        return 2
    host = sys.argv[2]
    command = ['ip', '-4', '-o', 'addr', 'show', 'scope', 'global']
    local = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, ' '.join(command)],
                            check=True, capture_output=True, text=True, timeout=60).stdout
    address = lan_addr(local, remote)
    if address is None:
        print(f'{host} shares no directly attached IPv4 subnet with this host', file=sys.stderr)
        return 1
    print(address)
    return 0


if __name__ == '__main__':
    sys.exit(main())
