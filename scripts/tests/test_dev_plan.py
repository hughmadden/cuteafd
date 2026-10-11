"""Planning trust boundary, through a fake gh executable (no network writes)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'dev' / 'plan'


@pytest.fixture
def fake_gh(tmp_path):
    executable = tmp_path / 'gh'
    executable.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
data = json.loads(Path(os.environ['GH_FIXTURE']).read_text())
if args[:2] == ['repo', 'view']:
    reply = {'nameWithOwner': 'test/repo'}
elif args[0] in ('pr', 'issue') and args[1] == 'list':
    reply = data.get(args[0] + 's', [])
    if '--label' in args:
        label = args[args.index('--label') + 1]
        reply = [x for x in reply if any(l['name'] == label for l in x['labels'])]
elif args[:2] == ['api', 'graphql']:
    field = 'pullRequest' if 'pullRequest(number' in ' '.join(args) else 'issue'
    reply = {'data': {'repository': {field: data['item']}}}
elif args[0] == 'api' and args[1].endswith('/timeline'):
    assert '--paginate' in args and '--slurp' in args
    reply = data.get('timeline', [[]])
elif args[0] == 'api' and args[1].endswith('/comments'):
    assert '--paginate' in args and '--slurp' in args
    reply = data.get('comments', [[]])
else:
    reply = {'pull_request': {}} if data.get('prs') else {}
print(json.dumps(reply))
''')
    executable.chmod(0o755)
    fixture = tmp_path / 'fixture.json'
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'GH_FIXTURE': str(fixture), 'CUTEAFD_PLAN_OWNER': 'tpurtell'}

    def call(data, *args):
        fixture.write_text(json.dumps(data))
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=env,
                              capture_output=True, text=True)
    return call


def data(author='outside', accepted=True, kind='issue', edited=None, pushed=None):
    labels = [{'name': 'accepted'}] if accepted else []
    item = dict(number=17, title='A plan', body='State', author={'login': author},
                labels={'nodes': labels}, milestone={'title': 'v3'}, lastEditedAt=edited,
                updatedAt='2026-10-01T00:00:00Z')
    if kind == 'pr':
        item.update(headRefOid='abc', headRef={'target': {'pushedDate': pushed}})
    summary = {**item, 'labels': labels}
    return {'item': item, kind + 's': [summary], 'timeline': [[{
        'event': 'labeled', 'label': {'name': 'accepted'}, 'actor': {'login': 'tpurtell'},
        'created_at': '2026-10-02T00:00:00Z'}]]}


def test_author_trusted_without_label(fake_gh):
    result = fake_gh(data(author='tpurtell', accepted=False), 'list')
    assert result.returncode == 0, result.stderr
    assert '#17 issue A plan [] v3' in result.stdout


def test_accepted_by_owner_is_trusted_and_timeline_paginated(fake_gh):
    fixture = data()
    fixture['timeline'] = [[], fixture['timeline'][0]]
    result = fake_gh(fixture, 'list', '--issues', '--milestone', 'v3')
    assert result.returncode == 0, result.stderr
    assert '#17 issue' in result.stdout and 're-review' not in result.stdout


def test_accepted_by_other_is_untrusted(fake_gh):
    fixture = data()
    fixture['timeline'][0][0]['actor']['login'] = 'outside'
    assert fake_gh(fixture, 'list').stdout == ''
    shown = fake_gh(fixture, 'show', '17')
    assert shown.returncode == 1 and 'untrusted' in shown.stderr
    assert 'State' not in shown.stdout


def test_unaccepted_outsider_is_untrusted(fake_gh):
    assert fake_gh(data(accepted=False), 'list').stdout == ''


@pytest.mark.parametrize('kind,edited,pushed', [
    ('issue', '2026-10-03T00:00:00Z', None),
    ('pr', None, '2026-10-03T00:00:00Z'),
])
def test_edits_or_pushes_after_accept_need_review(fake_gh, kind, edited, pushed):
    result = fake_gh(data(kind=kind, edited=edited, pushed=pushed), 'list')
    assert result.returncode == 0, result.stderr
    assert '(re-review)' in result.stdout


def test_owner_authored_accepted_item_still_flags_edit(fake_gh):
    result = fake_gh(data(author='tpurtell', edited='2026-10-03T00:00:00Z'), 'list')
    assert result.returncode == 0, result.stderr
    assert '(re-review)' in result.stdout


def test_missing_push_time_uses_conservative_updated_time(fake_gh):
    fixture = data(kind='pr')
    fixture['item']['updatedAt'] = '2026-10-03T00:00:00Z'
    result = fake_gh(fixture, 'list', '--prs')
    assert result.returncode == 0, result.stderr
    assert '(re-review)' in result.stdout


def test_removed_then_reaccepted_by_other_does_not_keep_trust(fake_gh):
    fixture = data()
    event = fixture['timeline'][0][0]
    fixture['timeline'] += [[{**event, 'event': 'unlabeled'},
                              {**event, 'actor': {'login': 'outside'}}]]
    assert fake_gh(fixture, 'list').stdout == ''


def test_show_filters_nonowner_comments(fake_gh):
    fixture = data(author='tpurtell')
    fixture['comments'] = [[{'id': 1, 'user': {'login': 'outside'}, 'body': 'UNTRUSTED'}],
                           [{'id': 2, 'user': {'login': 'tpurtell'}, 'body': 'VERDICT'}]]
    result = fake_gh(fixture, 'show', '17')
    assert result.returncode == 0, result.stderr
    assert 'State' in result.stdout and 'VERDICT' in result.stdout
    assert 'UNTRUSTED' not in result.stdout


def test_decisions_and_pr_milestone_filter(fake_gh):
    fixture = data(author='tpurtell', kind='pr')
    fixture['prs'][0]['labels'].append({'name': 'decision-needed'})
    fixture['item']['labels']['nodes'].append({'name': 'decision-needed'})
    assert '#17 pr' in fake_gh(fixture, 'decisions').stdout
    assert fake_gh(fixture, 'list', '--prs', '--milestone', 'v3.1').stdout == ''
