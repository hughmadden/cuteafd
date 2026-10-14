#!/usr/bin/env python3
"""Interactive external-agent control via tmux; Python stdlib only."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ANSI = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
POLL = 0.5
IDLE_SETTLE = 2.0


def clean(text):
    return ANSI.sub('', text).replace('\r', '').replace('\x00', '')


def message_marker(text):
    return next((line.strip()[:100] for line in text.splitlines() if line.strip()), '')


def tmux(*args, check=True, data=None):
    result = subprocess.run(['tmux', *args], input=data, capture_output=True,
                            text=True, timeout=10)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or 'tmux command failed')
    return result.stdout


def runs():
    return Path(os.environ.get('EXT_AGENT_RUNS', '~/.cache/ext-agent')).expanduser().resolve()


def directory(name):
    if not NAME.fullmatch(name):
        raise ValueError('name must be 1-64 letters, digits, underscores or hyphens')
    return runs() / name


def read_meta(path):
    return json.loads((path / 'meta.json').read_text())


@contextlib.contextmanager
def locked(path):
    with (path / 'control.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def save(path, **updates):
    with locked(path):
        meta = read_meta(path)
        meta.update(updates)
        temporary = path / ('meta.' + uuid.uuid4().hex)
        temporary.write_text(json.dumps(meta, separators=(',', ':')))
        temporary.replace(path / 'meta.json')
    return meta


def event(path, kind, **fields):
    with (path / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.time(), 'event': kind, **fields}) + '\n')


class Adapter:
    prompt = r'^\s*[>❯]\s*$'
    busy = r'(?i)esc to cancel|escape to interrupt|esc to interrupt|ctrl\+c to interrupt'
    error = r'(?i)eligibility check failed|\bUNAVAILABLE\b.*503|authentication failed|not signed in|rate limit|connection failed|\bError:'
    transient = r'(?i)Eligibility check failed:\s*UNAVAILABLE|UNAVAILABLE\s*\(code 503\)'
    asking = r'(?i)\b(?:allow|approve|permission required|confirm|choose an option|waiting for (?:your|user))\b'
    tools = r'^\s*(?:[◆○●]\s*(?:Ran|Running|Read|Edit|Write|Search)|\$\s|(?:Bash|Read|Edit|Write|Search)\()'
    trust_key = 'Enter'

    def state(self, text, meta):
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        body = '\n'.join(lines)
        if (re.search(r'^[\s]*[⠁-⣿]', '\n'.join(lines[-5:]), re.M) or
                re.search(self.busy, '\n'.join(lines[-2:])) or
                re.search(r'\b[1-9][0-9]* queued\b|edit queued messages', '\n'.join(lines[-2:])) or
                re.search(r'^[\s⠀-⣿]*(?:Waiting for response|Thinking|Working|Responding|Running command|Loading)[^\n]*', '\n'.join(lines[-3:]), re.M)):
            return 'busy'
        # Only inspect the current turn, not old questions/errors in scrollback.
        marker = message_marker(meta.get('last_message', ''))
        if marker:
            indices = [i for i, line in enumerate(lines) if marker in line]
            if indices:
                body = '\n'.join(lines[indices[-1] + 1:])
        if re.search(self.error, body):
            return 'error'
        prompt = any(re.search(self.prompt, line) for line in lines[-5:])
        if prompt:
            content = [line.strip() for line in body.splitlines()
                       if not re.search(self.prompt, line) and not self.footer(line)]
            if content and (content[-1].endswith('?') or re.search(self.asking, '\n'.join(content[-4:]))):
                return 'asking'
            return 'idle'
        if re.search(self.asking, '\n'.join(lines[-8:])):
            return 'asking'
        return 'busy' if meta.get('submitted_at') else 'starting'

    def footer(self, line):
        return bool(re.search(r'^[\s─-╿]+$|Worked for |minimal .* /help|ctrl\+o transcript|for shortcuts|Gemini .* (?:high|medium|low)$|Grok .*always-approve', line))


class Grok(Adapter):
    trust_key = 'y'

    def command(self, meta):
        cmd = [meta['executable'], '--always-approve', '--minimal', '--no-alt-screen', '--cwd', meta['cwd']]
        if meta.get('model'):
            cmd += ['--model', meta['model']]
        if meta.get('effort'):
            cmd += ['--reasoning-effort', meta['effort']]
        return cmd


class Agy(Adapter):
    def command(self, meta):
        cmd = [meta['executable'], '--dangerously-skip-permissions', '--add-dir', meta['cwd']]
        if meta.get('model'):
            cmd += ['--model', meta['model']]
        elif meta.get('effort'):
            cmd += ['--effort', meta['effort']]
        return cmd


ADAPTERS = {'grok': Grok(), 'agy': Agy()}


def pane(meta):
    info = tmux('display-message', '-p', '-t', meta['pane'], '#{pane_dead}:#{pane_pid}', check=False).strip()
    if not info or ':' not in info:
        return None
    dead, pid = info.split(':')
    if not pid.isdigit():
        return None
    return {'dead': dead == '1', 'pid': int(pid)}


def screen(meta, lines=80):
    return clean(tmux('capture-pane', '-p', '-t', meta['pane'], '-S', '-' + str(lines), check=False)).rstrip()


def snapshot(path, lines=30):
    meta = read_meta(path)
    info = pane(meta) if meta.get('pane') else None
    text = screen(meta, max(200, lines)) if info else ''
    state = meta.get('state', 'starting')
    if info and not info['dead'] and state not in ('starting', 'error', 'exited'):
        state = ADAPTERS[meta['tool']].state(text, meta)
    if not info and state not in ('starting', 'error'):
        state = 'exited'
    quiet = max(0, time.time() - meta.get('last_output_at', meta['created_at']))
    if state in ('idle', 'asking') and meta.get('submitted_at'):
        raw = path / 'transcript.raw'
        size = raw.stat().st_size if raw.exists() else 0
        if quiet < IDLE_SETTLE or size != meta.get('raw_size'):
            state = 'busy'
    context = re.findall(r'\b([0-9.]+[KM]?)\s*/\s*([0-9.]+[KM]?)\s*\(([0-9]+)%\)', text, re.I)
    usage = {'used': context[-1][0], 'limit': context[-1][1], 'percent': int(context[-1][2])} if context else None
    return {'context_usage': usage, 'name': meta['name'], 'tool': meta['tool'], 'state': state,
            'seconds_since_output': round(quiet, 1),
            'needs_attention': state in ('asking', 'error', 'exited') or
                (state == 'idle' and bool(meta.get('submitted_at'))) or quiet > meta['quiet_seconds'],
            'lines': text.splitlines()[-lines:],
            'recent_tools': [line.strip() for line in text.splitlines()
                             if re.search(ADAPTERS[meta['tool']].tools, line)][-10:],
            'retry_count': meta.get('retry_count', 0), 'retry_at': meta.get('retry_at'),
            'error': meta.get('error'), 'session': meta['session'], 'run_dir': str(path)}


def submit(path, text):
    if not text.strip():
        raise ValueError('message must not be empty')
    if re.search(r'[\x00-\x08\x0b-\x1f\x7f]', text):
        raise ValueError('message must not contain terminal control characters')
    with locked(path):
        meta = read_meta(path)
        info = pane(meta)
        if not info or info['dead']:
            raise RuntimeError('agent has exited')
        before = screen(meta, 200)
        queued_before = re.search(r'\b([0-9]+) queued\b', before)
        queued_before = int(queued_before[1]) if queued_before else 0
        buffer = 'ext-agent-' + uuid.uuid4().hex
        tmux('load-buffer', '-b', buffer, '-', data=text)
        try:
            tmux('paste-buffer', '-p', '-d', '-b', buffer, '-t', meta['pane'])
            # Both TUIs treat a bracketed paste as one multiline input. Let their
            # event loops consume it before sending the separate submit key.
            time.sleep(0.15)
            pasted = screen(meta, 200)
            tmux('send-keys', '-t', meta['pane'], 'Enter')
        finally:
            tmux('delete-buffer', '-b', buffer, check=False)
        meta.update(last_message=text, submitted_at=time.time(), state='busy', error=None)
        temporary = path / ('meta.' + uuid.uuid4().hex)
        temporary.write_text(json.dumps(meta, separators=(',', ':')))
        temporary.replace(path / 'meta.json')
    event(path, 'submit', characters=len(text), paste_visible=message_marker(text) in pasted)
    # Verify echo, a busy state, or output after submission; never silently retry
    # here, since repeating a command could duplicate the worker's side effects.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = screen(meta, 200)
        queued = re.search(r'\b([0-9]+) queued\b', current)
        if message_marker(text) in current or (queued and int(queued[1]) > queued_before):
            return {'submitted': True, 'characters': len(text), 'queued': bool(queued)}
        time.sleep(POLL)
    raise RuntimeError('submission unconfirmed; inspect screen/transcript before sending again')


def spawn(path):
    meta = read_meta(path)
    adapter = ADAPTERS[meta['tool']]
    token = 'ext-agent-' + uuid.uuid4().hex
    command = 'tmux wait-for ' + token + '; exec ' + shlex.join(adapter.command(meta))
    # PWD must agree with -c: a stale inherited PWD makes Grok trust the wrong tree.
    output = tmux('new-session', '-d', '-P', '-F', '#{pane_id}', '-s', meta['session'],
                  '-x', '180', '-y', '50', '-c', meta['cwd'],
                  *[arg for key, value in os.environ.items() if key not in ('TMUX', 'TMUX_PANE', 'PWD')
                    for arg in ('-e', key + '=' + value)], '-e', 'PWD=' + meta['cwd'], command)
    pane_id = output.strip()
    tmux('set-option', '-t', pane_id, 'remain-on-exit', 'on')
    tmux('set-option', '-t', pane_id, 'history-limit', '50000')
    tmux('pipe-pane', '-t', pane_id, 'cat >> ' + shlex.quote(str(path / 'transcript.raw')))
    save(path, pane=pane_id, state='starting', screen_hash='', retry_at=None, error=None)
    tmux('wait-for', '-S', token)
    event(path, 'launch', command=adapter.command(meta))


def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def descendants(pid):
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                pass
    found = {pid}
    while True:
        extended = found | {child for child, parent in parents.items() if parent in found}
        if extended == found:
            return [(child, identity(child)) for child in found]
        found = extended


def terminate(meta):
    info = pane(meta) if meta.get('pane') else None
    processes = descendants(info['pid']) if info and not info['dead'] else []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, birth in reversed(processes):
            if birth and identity(pid) == birth:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
        if sig == signal.SIGTERM and processes:
            time.sleep(0.3)
    if info:
        tmux('kill-session', '-t', meta['pane'], check=False)


def write_final(path, text, meta):
    lines = [line.rstrip() for line in text.splitlines()]
    headers = [i for i, line in enumerate(lines) if re.match(r'^\s*[>❯]\s+\S', line)]
    if headers:
        lines = lines[headers[-1] + 1:]
    adapter = ADAPTERS[meta['tool']]
    result = []
    for line in lines:
        if re.search(r'^\s*Worked for ', line):
            break
        if (re.search(adapter.prompt, line) or adapter.footer(line) or
                re.search(r'^\s*[┃◆○●]|^\s*\$ ', line)):
            continue
        result.append(line)
    message = '\n'.join(result).strip()
    with locked(path):
        destination = path / 'final.md'
        if destination.exists():
            return
        destination.write_text('# Worker final snapshot\n\nState: idle\n'
                               f"Created: {meta['created_at']}\nSubmitted: {meta['submitted_at']}\n"
                               f'Idle: {time.time()}\n\nLast assistant message (screen-extracted):\n\n' + message + '\n')
    event(path, 'final_snapshot', file=str(destination))


def supervise(path):
    meta = read_meta(path)
    save(path, supervisor_pid=os.getpid(), supervisor_birth=identity(os.getpid()))
    try:
        spawn(path)
        trust_sent = False
        while True:
            meta = read_meta(path)
            if meta.get('stop_requested'):
                terminate(meta)
                save(path, state='exited', exited_at=time.time(), retry_at=None)
                return
            info = pane(meta)
            text = screen(meta, 200) if info else ''
            adapter = ADAPTERS[meta['tool']]
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            raw = path / 'transcript.raw'
            size = raw.stat().st_size if raw.exists() else 0
            changed = fingerprint != meta.get('screen_hash') or size != meta.get('raw_size')
            if changed:
                meta = save(path, screen_hash=fingerprint, raw_size=size, last_output_at=time.time())
            if 'Do you trust the contents' in text and not trust_sent:
                tmux('send-keys', '-t', meta['pane'], adapter.trust_key)
                trust_sent = True
                event(path, 'trust', cwd=meta['cwd'])
                time.sleep(POLL)
                continue
            state = adapter.state(text, meta) if info and not info['dead'] else 'exited'
            if state in ('idle', 'asking') and meta.get('submitted_at') and time.time() - meta['last_output_at'] < IDLE_SETTLE:
                state = 'busy'
            # Old 503 output must not trigger retries after a later successful turn.
            current = text
            marker = message_marker(meta.get('last_message', ''))
            if marker and marker in current:
                current = current.rsplit(marker, 1)[-1]
            if re.search(adapter.transient, current) and meta['tool'] == 'agy':
                count = meta.get('retry_count', 0)
                if count < meta['max_retries']:
                    retry_at = meta.get('retry_at') or time.time() + meta['retry_delay'] * (2 ** count)
                    save(path, state='error', retry_at=retry_at, error='transient agy 503; retry scheduled')
                    if time.time() >= retry_at:
                        event(path, 'retry', attempt=count + 1)
                        message = meta.get('last_message')
                        if info and not info['dead'] and message and re.search(adapter.prompt, text, re.M):
                            save(path, retry_count=count + 1, retry_at=None)
                            submit(path, message)
                        else:
                            terminate(meta)
                            save(path, retry_count=count + 1, submitted_at=None, replay=message)
                            spawn(path)
                            trust_sent = False
                    time.sleep(POLL)
                    continue
                state = 'error'
                save(path, error='agy 503 retry budget exhausted', retry_at=None)
            if state == 'exited':
                save(path, state=state, exited_at=time.time())
                return
            if state == 'idle' and meta.get('submitted_at'):
                write_final(path, text, meta)
            if state != meta.get('state'):
                save(path, state=state)
                event(path, 'state', state=state)
            if state == 'idle' and (meta.get('replay') or (not meta.get('submitted_at') and meta.get('brief'))):
                message = meta.get('replay') or meta['brief']
                save(path, replay=None)
                submit(path, message)
            time.sleep(POLL)
    except Exception as exc:
        event(path, 'supervisor_error', error=str(exc))
        save(path, state='error', error=str(exc))


def start(args):
    path = directory(args.name)
    runs().mkdir(parents=True, exist_ok=True)
    # Existing runs are never overwritten, including their evidence and cursor.
    cwd = str(Path(args.cwd).expanduser().resolve())
    if not Path(cwd).is_dir():
        raise ValueError('cwd is not a directory')
    pieces = [Path(f).expanduser().read_text() for f in (args.preamble, args.brief) if f]
    if args.max_retries < 0 or min(args.retry_delay, args.ready_timeout, args.quiet_seconds) < 0:
        raise ValueError('timeouts and retry budget must be nonnegative')
    session = 'ext-' + args.name
    exists = subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True, timeout=10)
    if exists.returncode == 0:
        raise ValueError('tmux session already exists; never reuse another launcher\'s session')
    path.mkdir(mode=0o700)
    now = time.time()
    meta = {'name': args.name, 'tool': args.tool, 'session': 'ext-' + args.name,
            'cwd': cwd, 'model': args.model, 'effort': args.effort,
            'executable': os.environ.get('EXT_AGENT_' + args.tool.upper(), args.tool),
            'brief': '\n\n'.join(pieces), 'created_at': now, 'last_output_at': now,
            'state': 'starting', 'quiet_seconds': args.quiet_seconds,
            'max_retries': args.max_retries, 'retry_delay': args.retry_delay}
    (path / 'meta.json').write_text(json.dumps(meta, separators=(',', ':')))
    with (path / 'supervisor.log').open('a') as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_supervise', str(path)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                         cwd=cwd)
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        meta = read_meta(path)
        if meta['state'] in ('idle', 'asking', 'error', 'exited') and (not pieces or meta.get('submitted_at') or meta['state'] in ('error', 'exited')):
            return snapshot(path)
        if meta.get('submitted_at'):
            return snapshot(path)
        time.sleep(POLL)
    result = snapshot(path)
    result['ready_timeout'] = True
    return result


def wait(args):
    path = directory(args.name)
    first = snapshot(path)
    initial = first['state']
    # Edge-triggered: a worker already past the threshold when the wait starts
    # was reported by an earlier wait, so only a fresh crossing wakes the driver.
    start_usage = first['context_usage']
    context_armed = not (start_usage and args.until_context and start_usage['percent'] >= args.until_context)
    if args.progress_interval < 0:
        raise ValueError('progress interval must be nonnegative')
    deadline = time.monotonic() + args.timeout
    progress_at = time.monotonic() + args.progress_interval
    raw = path / 'transcript.raw'
    offset = raw.stat().st_size if raw.exists() else 0
    while True:
        result = snapshot(path, args.lines)
        if args.progress_interval and time.monotonic() >= progress_at:
            size = raw.stat().st_size if raw.exists() else 0
            if size > offset:
                with raw.open('rb') as handle:
                    handle.seek(offset)
                    # Do not advance the driver's incremental transcript cursor.
                    delta = clean(handle.read().decode('utf-8', errors='replace'))
                    offset = handle.tell()
                notable = ADAPTERS[read_meta(path)['tool']].tools + r'|(?i:error|warning:|panicked|test result)'
                markers = list(dict.fromkeys(line.strip()[:160] for line in delta.splitlines()
                                            if re.search(notable, line)))
                usage = result['context_usage']
                digest = {'event': 'progress', 'state': result['state'],
                          'seconds_since_output': result['seconds_since_output'],
                          'context_pct': usage['percent'] if usage else None,
                          'recent_tools': [line[:160] for line in result['recent_tools']],
                          'new_markers': markers[-5:]}
                print(json.dumps(digest, separators=(',', ':')), flush=True)
                if args.return_on_progress:
                    # One driver turn per progress period while the worker is producing.
                    result['woke'] = 'progress'
                    return result
            progress_at = time.monotonic() + args.progress_interval
        meta = read_meta(path)
        retryable = (meta['tool'] == 'agy' and result['retry_count'] < meta['max_retries'] and
                     re.search(ADAPTERS['agy'].transient, '\n'.join(result['lines'])))
        if ((args.until == 'any' and result['state'] != initial) or result['state'] == args.until or
                (args.until == 'idle' and (result['state'] in ('asking', 'exited') or
                                          (result['state'] == 'error' and not result['retry_at'] and not retryable)))):
            return result
        # Wake the driver when it has something to act on, not only on a timer.
        usage = result['context_usage']
        if usage and args.until_context and usage['percent'] < args.until_context:
            context_armed = True
        if context_armed and args.until_context and usage and usage['percent'] >= args.until_context:
            result['woke'] = 'context'
            return result
        if args.until_quiet and result['state'] == 'busy' and result['seconds_since_output'] >= args.until_quiet:
            result['woke'] = 'quiet'
            return result
        if time.monotonic() >= deadline:
            result['timed_out'] = True
            return result
        time.sleep(POLL)


def transcript(path, incremental):
    with locked(path):
        file = path / 'transcript.raw'
        cursor = path / 'transcript.cursor'
        offset = int(cursor.read_text()) if incremental and cursor.exists() else 0
        with file.open('rb') if file.exists() else contextlib.nullcontext(None) as handle:
            if handle:
                handle.seek(offset)
                text = clean(handle.read().decode('utf-8', errors='replace'))
                end = handle.tell()
            else:
                text, end = '', 0
        if incremental:
            cursor.write_text(str(end))
    return {'text': text, 'offset': offset, 'next_offset': end}


def prune(hours):
    if hours < 0:
        raise ValueError('older-than must be nonnegative')
    archived = []
    now = time.time()
    for file in sorted(runs().glob('*/meta.json')):
        path = file.parent
        meta = read_meta(path)
        if snapshot(path)['state'] != 'exited':
            continue
        if now - meta.get('exited_at', now) < hours * 3600:
            continue
        # A stop can race the supervisor's last write. Do not move live control
        # files until both processes have actually gone away.
        pid = meta.get('supervisor_pid')
        if pid and meta.get('supervisor_birth') == identity(pid):
            continue
        info = pane(meta) if meta.get('pane') else None
        if info and not info['dead']:
            continue
        archive = runs() / 'archive'
        archive.mkdir(mode=0o700, exist_ok=True)
        destination = archive / (meta['name'] + '-' + str(int(now)) + '-' + uuid.uuid4().hex[:8])
        with locked(path):
            path.rename(destination)
        archived.append({'name': meta['name'], 'archive_dir': str(destination)})
    return {'archived': archived}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', action='store_true', help='human-readable output (also accepted after subcommands)')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('start', 'status', 'screen', 'send', 'wait', 'interrupt', 'stop', 'list', 'prune', 'transcript'):
        c = sub.add_parser(command)
        c.add_argument('--text', action='store_true', default=argparse.SUPPRESS)
        if command not in ('list', 'prune'):
            c.add_argument('name')
        if command == 'prune':
            c.add_argument('--older-than', type=float, default=0, metavar='H', help='minimum hours since exit')
        if command == 'start':
            c.add_argument('--tool', choices=ADAPTERS, required=True)
            c.add_argument('--model')
            c.add_argument('--effort')
            c.add_argument('--cwd', default=os.getcwd())
            c.add_argument('--brief')
            c.add_argument('--preamble')
            c.add_argument('--ready-timeout', type=float, default=60)
            c.add_argument('--quiet-seconds', type=float, default=300)
            c.add_argument('--max-retries', type=int, default=4)
            c.add_argument('--retry-delay', type=float, default=60)
        if command in ('status', 'screen', 'wait'):
            c.add_argument('--lines', type=int, default=30 if command != 'screen' else 80)
        if command == 'send':
            c.add_argument('message', nargs='?')
            c.add_argument('--file')
        if command == 'wait':
            c.add_argument('--until', choices=('idle', 'asking', 'error', 'exited', 'any'), default='idle')
            c.add_argument('--timeout', type=float, default=600)
            c.add_argument('--progress-interval', type=float, default=0, metavar='S', help='emit progress JSON lines on new output every S seconds')
            c.add_argument('--until-context', type=float, default=0, metavar='PCT', help='also return when the worker context fill crosses PCT percent during this wait')
            c.add_argument('--return-on-progress', action='store_true', help='return after the first progress event (one driver turn per period)')
            c.add_argument('--until-quiet', type=float, default=0, metavar='S', help='also return if a busy worker writes nothing for S seconds')
        if command == 'transcript':
            c.add_argument('--since-last', action='store_true')
    c = sub.add_parser('_supervise', help=argparse.SUPPRESS)
    c.add_argument('path')
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == '_supervise':
            supervise(Path(args.path))
            return 0
        if args.command == 'start':
            result = start(args)
        elif args.command == 'list':
            result = [snapshot(file.parent) for file in sorted(runs().glob('*/meta.json'))]
        elif args.command == 'prune':
            result = prune(args.older_than)
        else:
            path = directory(args.name)
            meta = read_meta(path)
            if args.command == 'status':
                result = snapshot(path, args.lines)
            elif args.command == 'screen':
                result = {'text': screen(meta, args.lines)}
            elif args.command == 'send':
                if bool(args.file) == (args.message is not None):
                    raise ValueError('provide either TEXT or --file, not both')
                result = submit(path, Path(args.file).read_text() if args.file else args.message)
            elif args.command == 'wait':
                result = wait(args)
            elif args.command == 'transcript':
                result = transcript(path, args.since_last)
            elif args.command == 'interrupt':
                tmux('send-keys', '-t', meta['pane'], 'Escape')
                event(path, 'interrupt')
                result = {'interrupted': True}
            elif args.command == 'stop':
                save(path, stop_requested=True)
                pid = meta.get('supervisor_pid')
                if pid and meta.get('supervisor_birth') == identity(pid):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGTERM)
                terminate(read_meta(path))
                save(path, state='exited', exited_at=time.time(), retry_at=None)
                event(path, 'stop')
                result = {'stopped': True}
        if args.text:
            if isinstance(result, dict) and 'text' in result:
                print(result['text'])
            else:
                print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, separators=(',', ':')))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'error': str(exc)}, separators=(',', ':')))
        return 1


if __name__ == '__main__':
    sys.exit(main())
