import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'agents' / 'ext-agent.py'
spec = importlib.util.spec_from_file_location('ext_agent', SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

# A raw-terminal fake, not a mock of tmux: bracketed paste must remain one
# message and Enter must arrive outside it. Busy footer retains a > prompt.
FAKE = r'''#!/usr/bin/env python3
import os, select, signal, subprocess, sys, termios, time, tty
from pathlib import Path
if os.environ.get('FAKE_STARTUP_503'):
    marker = Path(os.environ['FAKE_STARTUP_503'])
    if not marker.exists():
        marker.write_text('once')
        print('Eligibility check failed: UNAVAILABLE (code 503)', flush=True)
        sys.exit(1)
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setraw(fd)
sys.stdout.write('\x1b[?2004h')
sys.stdout.flush()
tool = 'grok' if '--always-approve' in sys.argv else 'agy'
prompt = '❯' if tool == 'grok' else '>'

def show(text):
    sys.stdout.write('\x1b[2J\x1b[H' + text.replace('\n', '\r\n'))
    sys.stdout.flush()

def idle(answer=''):
    show(answer + '\n' + prompt + '\n? for shortcuts')

signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
idle('READY')
message = ''
inside = False
pending = b''
try:
    while True:
        if not select.select([fd], [], [], 0.1)[0]:
            continue
        pending += os.read(fd, 4096)
        while pending:
            if pending.startswith(b'\x1b[200~'):
                inside = True
                pending = pending[6:]
                continue
            if pending.startswith(b'\x1b[201~'):
                inside = False
                pending = pending[6:]
                continue
            if pending.startswith(b'\x1b') and len(pending) < 6 and inside:
                break
            char, pending = pending[:1], pending[1:]
            if char == b'\x1b':
                message = ''
                idle('Interrupted')
            elif char in (b'\r', b'\n') and not inside:
                if not message:
                    continue
                text, message = message.replace('\r', '\n'), ''
                with Path(os.environ['FAKE_MESSAGES']).open('a') as log:
                    import json
                    log.write(json.dumps(text) + '\n')
                show(prompt + ' ' + text + '\nThinking...\n' + prompt + '\nesc to cancel')
                if 'spawn-child' in text:
                    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
                    Path(os.environ['FAKE_CHILD']).write_text(str(child.pid))
                deadline = time.monotonic() + float(os.environ.get('FAKE_DELAY', '0.8'))
                interrupted = False
                while time.monotonic() < deadline:
                    if select.select([fd], [], [], 0.05)[0]:
                        incoming = os.read(fd, 4096)
                        if incoming == b'\x1b':
                            idle('Interrupted')
                            interrupted = True
                            break
                        pending += incoming
                if interrupted:
                    continue
                once = Path(os.environ['FAKE_ONCE'])
                if '503' in text and not once.exists():
                    once.write_text('seen')
                    idle(prompt + ' ' + text + '\nEligibility check failed: UNAVAILABLE (code 503)')
                elif 'question' in text:
                    idle(prompt + ' ' + text + '\nWhich color should I choose?')
                else:
                    idle(prompt + ' ' + text + '\nDONE ' + text)
            else:
                message += char.decode('utf-8', errors='replace')
                show(prompt + ' ' + message + '\n? for shortcuts')
finally:
    termios.tcsetattr(fd, termios.TCSANOW, old)
'''


@pytest.fixture
def cli(tmp_path, monkeypatch):
    if not shutil.which('tmux'):
        pytest.skip('tmux is required for the external driver integration tests')
    fake = tmp_path / 'fake-cli'
    fake.write_text(FAKE)
    fake.chmod(0o755)
    run_root = tmp_path / 'runs'
    monkeypatch.setenv('EXT_AGENT_RUNS', str(run_root))
    monkeypatch.setenv('EXT_AGENT_GROK', str(fake))
    monkeypatch.setenv('EXT_AGENT_AGY', str(fake))
    monkeypatch.setenv('FAKE_MESSAGES', str(tmp_path / 'messages'))
    monkeypatch.setenv('FAKE_CHILD', str(tmp_path / 'child'))
    monkeypatch.setenv('FAKE_ONCE', str(tmp_path / 'once'))
    name = 'test-' + uuid.uuid4().hex[:12]

    def call(*args, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPT), *args],
                                capture_output=True, text=True, timeout=25)
        assert not ok or result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    yield name, call, tmp_path
    if (run_root / name / 'meta.json').exists():
        call('stop', name)


@pytest.mark.parametrize('tool', ['grok', 'agy'])
def test_start_send_wait_and_transcript(cli, tool):
    name, call, root = cli
    brief = root / 'brief'
    brief.write_text('FIRST\nSECOND')
    result = call('start', name, '--tool', tool, '--cwd', str(root), '--brief', str(brief))
    assert result['state'] == 'busy'
    result = call('wait', name, '--until', 'idle', '--timeout', '8')
    assert not result.get('timed_out')
    messages = [json.loads(line) for line in (root / 'messages').read_text().splitlines()]
    assert messages == ['FIRST\nSECOND']
    assert result['needs_attention']
    assert call('send', name, 'THIRD\nFOURTH')['submitted']
    call('wait', name, '--until', 'idle', '--timeout', '8')
    messages = [json.loads(line) for line in (root / 'messages').read_text().splitlines()]
    assert messages == ['FIRST\nSECOND', 'THIRD\nFOURTH']
    first = call('transcript', name, '--since-last')
    assert 'FIRST' in first['text'] and 'THIRD' in first['text']
    assert '\x1b' not in first['text']
    second = call('transcript', name, '--since-last')
    assert second['text'] == '' and second['offset'] == first['next_offset']
    assert call('list')[0]['name'] == name
    assert 'FOURTH' in call('screen', name)['text']
    assert call('wait', name, '--until', 'error', '--timeout', '0.1')['timed_out']
    call('stop', name)
    assert call('status', name)['state'] == 'exited'
    assert subprocess.run(['tmux', 'has-session', '-t', 'ext-' + name], capture_output=True).returncode
    assert 'File exists' in call('start', name, '--tool', tool, ok=False)['error']


def test_asking_then_answer(cli):
    name, call, root = cli
    call('start', name, '--tool', 'agy', '--cwd', str(root))
    call('send', name, 'question')
    result = call('wait', name, '--until', 'asking', '--timeout', '8')
    assert result['state'] == 'asking' and result['needs_attention']
    call('send', name, 'blue')
    assert call('wait', name, '--until', 'idle', '--timeout', '8')['state'] == 'idle'


def test_503_retry(cli):
    name, call, root = cli
    brief = root / 'brief'
    brief.write_text('503 please')
    call('start', name, '--tool', 'agy', '--cwd', str(root), '--brief', str(brief), '--retry-delay', '0.2')
    result = call('wait', name, '--until', 'error', '--timeout', '8')
    assert result['state'] == 'error'
    deadline = time.monotonic() + 12
    while True:
        result = call('wait', name, '--until', 'idle', '--timeout', '12')
        if result['state'] == 'idle':
            break
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert not result.get('timed_out'), result
    assert result['retry_count'] == 1
    messages = [json.loads(line) for line in (root / 'messages').read_text().splitlines()]
    assert messages == ['503 please', '503 please']
    events = (root / 'runs' / name / 'events.jsonl').read_text()
    assert '"event": "retry"' in events
    assert '503' in call('transcript', name)['text']
    # A mid-run error must retry the latest steering message, not the brief.
    (root / 'once').unlink()
    call('send', name, '503 later')
    result = call('wait', name, '--until', 'idle', '--timeout', '12')
    assert result['retry_count'] == 2 and not result.get('timed_out')
    messages = [json.loads(line) for line in (root / 'messages').read_text().splitlines()]
    assert messages[-2:] == ['503 later', '503 later']


def test_stop_kills_child(cli):
    name, call, root = cli
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    call('send', name, 'spawn-child')
    call('wait', name, '--until', 'idle', '--timeout', '8')
    pid = int((root / 'child').read_text())
    assert driver.identity(pid)
    call('stop', name)
    deadline = time.monotonic() + 5
    while driver.identity(pid) and time.monotonic() < deadline:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            break
        time.sleep(0.1)
    else:
        assert driver.identity(pid) is None


@pytest.mark.parametrize('text,state', [
    ('>\n? for shortcuts', 'idle'),
    ('Thinking...\n>\nesc to cancel', 'busy'),
    ('Which color should I choose?\n>\n? for shortcuts', 'asking'),
    ('Eligibility check failed: UNAVAILABLE (code 503)\n>', 'error'),
    ('Signing in...', 'starting'),
    ('⠸ Waiting for response...\n❯\nGrok 4.7 high', 'busy'),
    ('❯\nGrok 4.7 high 1 queued', 'busy'),
    ('>\nPress up to edit queued messages', 'busy'),
    ('esc to cancel\n> question\nWhich color?\n>\n? for shortcuts', 'asking'),
])
def test_adapter_states(text, state):
    assert driver.Agy().state(text, {}) == state


def test_names_and_terminal_controls(cli):
    name, call, root = cli
    assert 'name must' in call('start', '../escape', '--tool', 'grok', ok=False)['error']
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    assert 'control characters' in call('send', name, '\x1b[31munsafe', ok=False)['error']
    assert 'either TEXT' in call('send', name, 'text', '--file', '/no/file', ok=False)['error']


def test_since_last_cursor_is_independent_of_full_read(cli):
    name, call, root = cli
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    first = call('transcript', name, '--since-last')
    call('transcript', name)
    assert call('transcript', name, '--since-last')['offset'] == first['next_offset']


def test_interrupt_and_midrun_send(cli, monkeypatch):
    name, call, root = cli
    monkeypatch.setenv('FAKE_DELAY', '2')
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    call('send', name, 'first turn')
    call('send', name, 'queued turn')
    call('wait', name, '--until', 'idle', '--timeout', '8')
    messages = [json.loads(line) for line in (root / 'messages').read_text().splitlines()]
    assert messages == ['first turn', 'queued turn']
    call('send', name, 'interrupt me')
    assert call('interrupt', name)['interrupted']
    result = call('wait', name, '--until', 'idle', '--timeout', '8')
    assert 'Interrupted' in '\n'.join(result['lines'])


def test_preexisting_session_is_not_stopped(cli):
    name, call, root = cli
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'ext-' + name, 'sleep 60'], check=True)
    try:
        result = call('start', name, '--tool', 'agy', '--cwd', str(root), ok=False)
        assert 'already exists' in result['error']
        assert not (root / 'runs' / name).exists()
        assert subprocess.run(['tmux', 'has-session', '-t', 'ext-' + name]).returncode == 0
    finally:
        subprocess.run(['tmux', 'kill-session', '-t', 'ext-' + name], check=True)


def test_retry_budget_exhausted(cli):
    name, call, root = cli
    call('start', name, '--tool', 'agy', '--cwd', str(root), '--max-retries', '0')
    call('send', name, '503 budget')
    result = call('wait', name, '--until', 'error', '--timeout', '8')
    assert result['state'] == 'error' and result['retry_count'] == 0
    deadline = time.monotonic() + 2
    while 'budget exhausted' not in (result.get('error') or '') and time.monotonic() < deadline:
        time.sleep(0.1)
        result = call('status', name)
    assert 'budget exhausted' in result['error']


def test_prompt_is_not_ready_until_busy_footer_disappears():
    text = '> task\nThinking...\n>\nesc to cancel'
    assert driver.Agy().state(text, {'submitted_at': time.time(), 'last_message': 'task'}) == 'busy'
    text = '> old question\nWhich color?\n> answer\nDONE\n>\n? for shortcuts'
    assert driver.Agy().state(text, {'last_message': 'answer'}) == 'idle'


def test_marker_handles_blank_and_wrapped_first_line():
    message = '\n' + 'x' * 240 + '\nnext'
    assert driver.message_marker(message) == 'x' * 100
    assert driver.Agy().state('> ' + 'x' * 170 + '\nWhich color?\n>\n? for shortcuts', {'last_message': message}) == 'asking'


def test_prune_archives_only_old_exited_runs_and_restart_adopts(cli):
    name, call, root = cli
    # Every call is a new process: start exits while its supervisor/worker live,
    # then another process adopts the same run and cursor without restarting it.
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    meta_path = root / 'runs' / name / 'meta.json'
    before = json.loads(meta_path.read_text())
    call('transcript', name, '--since-last')
    call('send', name, 'adopted by new driver')
    call('wait', name, '--until', 'idle', '--timeout', '8')
    after = json.loads(meta_path.read_text())
    assert before['pane'] == after['pane']
    assert before['supervisor_pid'] == after['supervisor_pid']
    assert 'adopted by new driver' in call('transcript', name, '--since-last')['text']
    assert call('prune')['archived'] == []
    call('stop', name)
    assert call('prune', '--older-than', '1')['archived'] == []
    # Give the terminated supervisor its scheduling quantum; zombie PIDs are
    # not active control writers and must not prevent pruning.
    deadline = time.monotonic() + 5
    while True:
        result = call('prune')
        if result['archived']:
            break
        assert time.monotonic() < deadline
        time.sleep(0.1)
    archive = Path(result['archived'][0]['archive_dir'])
    assert archive.parent == root / 'runs' / 'archive'
    assert (archive / 'transcript.raw').exists()
    assert (archive / 'meta.json').exists()
    assert call('list') == []
    assert not meta_path.exists()


def test_dead_startup_503_relaunches_and_clears_error(cli, monkeypatch):
    name, call, root = cli
    monkeypatch.setenv('FAKE_STARTUP_503', str(root / 'startup503'))
    call('start', name, '--tool', 'agy', '--cwd', str(root), '--retry-delay', '0.2')
    result = call('wait', name, '--until', 'idle', '--timeout', '12')
    assert not result.get('timed_out')
    assert result['retry_count'] == 1 and result['error'] is None
    events = [json.loads(line) for line in (root / 'runs' / name / 'events.jsonl').read_text().splitlines()]
    assert len([event for event in events if event['event'] == 'launch']) == 2
    assert 'Eligibility check failed' in call('transcript', name)['text']


def test_context_usage_and_wait_progress(cli):
    name, call, root = cli
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    raw = root / 'runs' / name / 'transcript.raw'
    cursor = call('transcript', name, '--since-last')['next_offset']
    process = subprocess.Popen([sys.executable, str(SCRIPT), 'wait', name,
                                '--until', 'error', '--timeout', '2', '--progress-interval', '0.2'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.35)
    with raw.open('a') as handle:
        handle.write('\n$ ls -1\n$ ls -1\nwarning: example\nplain uninteresting line\n')
    output, stderr = process.communicate(timeout=8)
    assert process.returncode == 0, stderr
    events = [json.loads(line) for line in output.splitlines()]
    progress = [event for event in events if event.get('event') == 'progress']
    assert len(progress) == 1  # No new output: no repeat digest.
    assert progress[0]['new_markers'] == ['$ ls -1', 'warning: example']
    assert progress[0]['context_pct'] is None
    assert events[-1]['timed_out']
    assert int((root / 'runs' / name / 'transcript.cursor').read_text()) == cursor
    meta = json.loads((root / 'runs' / name / 'meta.json').read_text())
    subprocess.run(['tmux', 'send-keys', '-t', meta['pane'], '-l',
                    '124K / 256K (49%)'], check=True)
    deadline = time.monotonic() + 3
    while True:
        result = call('status', name)
        if result['context_usage']:
            break
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert result['context_usage'] == {'used': '124K', 'limit': '256K', 'percent': 49}
    # The context wake is edge-triggered: already past the threshold when the wait
    # starts does not return at once (a driver would spin), a fresh crossing does.
    started = time.monotonic()
    result = call('wait', name, '--until', 'error', '--timeout', '1', '--until-context', '40')
    assert result['timed_out'] and 'woke' not in result and time.monotonic() - started >= 0.9
    process = subprocess.Popen([sys.executable, str(SCRIPT), 'wait', name,
                                '--until', 'error', '--timeout', '5', '--until-context', '60'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    subprocess.run(['tmux', 'send-keys', '-t', meta['pane'], '-l', ' 170K / 256K (66%)'], check=True)
    output, stderr = process.communicate(timeout=8)
    assert process.returncode == 0, stderr
    assert json.loads(output.splitlines()[-1])['woke'] == 'context'


def test_wait_idle_yields_for_question(cli):
    name, call, root = cli
    call('start', name, '--tool', 'agy', '--cwd', str(root))
    call('send', name, 'question')
    result = call('wait', name, '--until', 'idle', '--timeout', '8')
    assert result['state'] == 'asking' and not result.get('timed_out')


def test_final_snapshot_survives_driver_and_is_first_idle_only(cli):
    name, call, root = cli
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    final = root / 'runs' / name / 'final.md'
    assert not final.exists()
    call('send', name, 'first assistant result')
    call('wait', name, '--until', 'idle', '--timeout', '8')
    deadline = time.monotonic() + 3
    while not final.exists():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    first = final.read_text()
    assert 'State: idle' in first and 'Submitted:' in first and 'Idle:' in first
    assert 'DONE first assistant result' in first
    call('send', name, 'later result')
    call('wait', name, '--until', 'idle', '--timeout', '8')
    assert final.read_text() == first


def test_visible_prompt_stream_growth_is_not_idle(cli):
    name, call, root = cli
    call('start', name, '--tool', 'grok', '--cwd', str(root))
    call('send', name, 'streaming turn')
    assert call('status', name)['state'] == 'busy'
    # The fake clears its busy footer and leaves a visible prompt after 0.8 s;
    # the two-second quiet window must still keep it busy.
    time.sleep(1)
    assert call('wait', name, '--until', 'idle', '--timeout', '0.1')['timed_out']
    result = call('wait', name, '--until', 'idle', '--timeout', '8')
    assert result['state'] == 'idle' and result['seconds_since_output'] >= 2


def test_running_background_command_with_prompt_is_busy():
    text = '◆ Task started: Test cuteafd-core\n◎ 1 command still running · send a message to interrupt\n❯\nGrok 4.7 (high) · always-approve · 159K / 256K (62%) · ctrl+o transcript\n'
    assert driver.ADAPTERS['grok'].state(text, {'submitted_at': 1}) == 'busy'


def test_bare_spinner_with_prompt_is_busy():
    assert driver.Grok().state('⠸ Thinking...\n❯\nGrok 4.7 high', {}) == 'busy'
    assert driver.Agy().state('⠸ Streaming response...\n>\n? for shortcuts', {}) == 'busy'


def test_wait_wake_flags_are_wired():
    source = (Path(__file__).resolve().parents[2] / 'scripts/agents/ext-agent.py').read_text()
    for flag in ('--return-on-progress', '--until-context', '--until-quiet'):
        assert flag in source
    assert "result['woke'] = 'progress'" in source
    assert "result['woke'] = 'context'" in source
    assert "result['woke'] = 'quiet'" in source
