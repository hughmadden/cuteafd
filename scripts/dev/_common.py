"""Shared GitHub invocation and public-text guards for dev commands."""
import json
import os
import re
import subprocess
from pathlib import Path

OWNER = os.environ.get('CUTEAFD_PLAN_OWNER', 'tpurtell')
SECTIONS = ('Goal', 'Base/HEAD', 'Gates', 'In flight', 'Next step', 'Decision needed')


def run(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        # gh diagnostics can quote a rejected body; don't print them publicly.
        raise ValueError(f'{args[0]} {args[1]} failed (exit {result.returncode})')
    return result.stdout


def gh(*args):
    return json.loads(run('gh', *args))


def repo():
    return gh('repo', 'view', '--json', 'nameWithOwner')['nameWithOwner']


def branch():
    name = run('git', 'symbolic-ref', '--quiet', '--short', 'HEAD').strip()
    if not name:
        raise ValueError('a checked-out branch is required')
    return name


def task_dir(task=None):
    task = task or branch().removeprefix('work/')
    path = Path(task)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('task must be a relative name without ..')
    root = Path.home() / '.cache/cuteafd/builds'
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('task path escapes the build directory')
    return target


def validate_public(text, directory):
    patterns = (
        r'sk-', r'ghp_', r'github_pat_', r'-----BEGIN', r'AKIA',
        r'xox[bp]-', r'Bearer\s+',
        r'(?i)\b(?:[\w-]*(?:key|token|secret)[\w-]*)\b[\s\"\']*[:=][\s\"\']*[A-Za-z0-9+/=_-]{32,}',
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError('refusing text matching a secret pattern')
    if directory.exists():
        for path in directory.rglob('*'):
            if path.is_file() and ('api-key' in path.name.lower() or path.name.lower() == 'api.key'):
                secret = path.read_text().strip()
                if secret and secret in text:
                    raise ValueError('refusing text containing a task API key')


def validate_body(text, directory):
    if len(text.encode('utf-8')) > 4096:
        raise ValueError('PR body exceeds 4096 bytes')
    # Fenced examples do not count as actual state sections.
    headings = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            fenced = not fenced
        elif not fenced:
            match = re.match(r'^#{1,6}\s+(.+?)\s*#*\s*$', line)
            if match:
                headings.append(match[1])
    missing = [section for section in SECTIONS if section not in headings]
    if missing:
        raise ValueError('missing required sections: ' + ', '.join(missing))
    validate_public(text, directory)
