#!/usr/bin/env python3
"""Stop the owned container before the enclosing hardware lock is released."""
import argparse
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    process = subprocess.Popen(command)
    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        result = process.wait()
    finally:
        stopped = subprocess.run(["docker", "stop", "--time", "30", args.container],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if stopped.returncode:
            subprocess.run(["docker", "kill", args.container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
    return result


if __name__ == "__main__":
    sys.exit(main())
