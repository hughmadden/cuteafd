#!/usr/bin/env python3
"""Produce a fresh qualified family tree from measured repeat evidence; never upload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'python/reference'))
from fidelity_windows import validate_public_metadata, validate_public_text
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
for name in ("source", "out", "validation", "arms", "tokenizer"):
    parser.add_argument("--" + name, required=True, type=Path)
parser.add_argument("--config", required=True)
parser.add_argument("--comparison-policy-commit", required=True)
parser.add_argument("--validator-source-commit", required=True)
parser.add_argument("--omit-checkpoint-license", action="store_true",
                    help="Omit CHECKPOINT_LICENSE for this config; preserve other licence files")
args = parser.parse_args()
SOURCE, OUT, NAME, VALIDATION = args.source, args.out, args.config, args.validation
assert Path(NAME).name == NAME and NAME not in (".", "..")
assert OUT.resolve() != SOURCE.resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def omit_checkpoint_license(config, manifest):
    """Remove only the optional checkpoint notice from a fresh output config."""
    (config / 'CHECKPOINT_LICENSE').unlink(missing_ok=True)
    manifest['licence_files'] = [entry for entry in manifest.get('licence_files', [])
                                 if entry['path'] != 'CHECKPOINT_LICENSE']


report = json.loads((VALIDATION / 'report.json').read_text())
assert report['qualifies'] and report['repeatability_pass'], 'Measured qualification and baseline repeatability must both pass'
assert all(report['shapes'][s]['qualifies'] and report['shapes'][s]['original_pass']
           and report['shapes'][s]['pass_verdict'] for s in ('decode', 'prefill'))
assert report['qualification_scope'] == 'reference repeatability, not a precision-default verdict'
manifest = json.loads((SOURCE / NAME / 'manifest.json').read_text())
assert report['set_sha256'] == manifest['set_sha256']
for shape in ('decode', 'prefill'):
    for i in (0, 1):
        path = args.arms / f'baseline-{i}' / f'full-{shape}.json'
        assert digest(path) == report['source_report_sha256'][f'{i}-{shape}']
for entry in manifest['files']:
    assert digest(SOURCE / NAME / entry['path']) == entry['sha256']
assert not OUT.exists(), 'Preserve immutable output attempts'
# Only the config is handed off; the coordinator owns the shared root index/card.
OUT.mkdir()
config = OUT / NAME
shutil.copytree(SOURCE / NAME, config)
if args.omit_checkpoint_license:
    omit_checkpoint_license(config, manifest)
preparation = json.loads((config / 'qualification.json').read_text())
report['prefix_qualification'] = preparation['prefix_qualification']
report['source_audit'] = preparation['source_audit']
report['storage_verified'] = preparation['storage_verified']
report['privacy_verified'] = preparation['privacy_verified']
(config / 'qualification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
manifest['expect'] = report['family_expect']
manifest['calibration'] = dict(label="calibrated from this config's baselines",
    expect_by_path=report['expect_by_path'], baseline_metrics=report['baseline_metrics'],
    common_floor=report['common_floor'], paired_bar=report['paired_bar'],
    source_report_sha256=report['source_report_sha256'],
    legacy_v41_tripwire_context=report['legacy_v41_tripwire_context'],
    daemon_identity=report['daemon_identity'], coordinator_sha256=report['coordinator_sha256'],
    comparison_sha256=report['comparison_sha256'],
    qualification_scope=report['qualification_scope'])
revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
manifest['packaging_source_commit'] = revision
manifest['calibration']['coordinator_feature_commit'] = report['coordinator_feature_commit']
manifest['calibration']['coordinator_source_diff_sha256'] = report['coordinator_source_diff_sha256']
manifest['calibration']['comparison_policy_commit'] = args.comparison_policy_commit
manifest['calibration']['validator_source_commit'] = args.validator_source_commit
for entry in manifest.get('licence_files', []):
    if entry['path'] == '../LICENSE':
        licence = SOURCE / 'LICENSE'
        assert digest(licence) == entry['sha256']
        shutil.copyfile(licence, config / 'LICENSE')
        entry['path'] = 'LICENSE'
    elif entry['path'] == 'CHECKPOINT_LICENSE':
        assert digest(config / entry['path']) == entry['sha256'], 'Licence checksum differs'
        entry['provenance'] = 'Official checkpoint licence text, copied verbatim and unmodified from the pinned official snapshot'
manifest['qualification_sha256'] = digest(config / 'qualification.json')
manifest['publication_status'] = 'QUALIFIED REFERENCE - coordinator review and upload required; no precision default promoted'
manifest.pop('reference_sha256')
manifest['reference_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
(config / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
# The per-config card is outside the sealed manifest; shared root files are untouched.
readme = (SOURCE / NAME / 'README.md').read_text()
readme = readme.replace('Fidelity Draft', 'Fidelity Reference')
start = readme.index('This is a numerical-fidelity panel') if 'This is a numerical-fidelity panel' in readme else readme.index('DRAFT:')
end = readme.index('## Config')
readme = readme[:start] + ('This is a qualified numerical-fidelity reference, not a training corpus or a general\n'
    'quality ranking. Both full scoring shapes preserve paired full-vocabulary versus\n'
    'top-1024 deltas and bounds within 1e-4 nat and preserve the repeatability verdict.\n'
    'This is reference qualification, not a precision-default decision. Coordinator\n'
    'review and upload remain required; no publication revision is claimed.\n\n') + readme[end:]
readme = readme.replace('`qualification.json` currently records preparation checks, not a paired PASS.',
                       '`qualification.json` records actual paired repeatability and compact/full equivalence evidence.')
start = readme.index('The first checkpoint-precision baseline') if 'The first checkpoint-precision baseline' in readme else readme.index('## Pending Calibration')
end = readme.index('## Public-source and privacy policy') if '## Public-source and privacy policy' in readme else readme.index('## Provenance And Privacy')
body = ['Two default-precision baseline repeats completed for each full scoring shape, with',
        f"`FULL_PREFILL_LOGITS=on`; primary count is {report['shapes']['decode']['positions']:,}",
        'generated positions per run (see qualification.json for this config). Quick remains decode-only.', '',
        '| Shape | Repeat | Top-1 | Full-Vocabulary KL |',
        '|---|---:|---:|---:|']
for shape in ('decode', 'prefill'):
    for i, metrics in enumerate(report['baseline_metrics'][shape]):
        body.append(f"| {shape} | {i} | {metrics['top1'] * 100:.4f}% | {metrics['kl']:.8f} |")
body += ['', f"Common floor: top-1 >=90% / KL <=0.06 nat. This config's calibrated `expect`:",
         f"top-1 >={report['family_expect']['top1_min'] * 100:.0f}% / KL <={report['family_expect']['kl_max']:.2f} nat.",
         f"Calibrated confident-top-1 >={report['family_expect']['tripwires']['confident_top1_min'] * 100:.0f}% / "
         f"top-3 containment >={report['family_expect']['tripwires']['top3_min'] * 100:.0f}%.",
         'Paired gross tripwires: lower95(baseline-candidate) >0.01 confident-top-1',
         'or >0.005 top-3. Raw values and both bounds are retained; V4.1 98%/99% is legacy context.',
         'Per-path gates and the measured repeated baselines are retained in the manifest.',
         report['daemon_identity'] + '; binary SHA256 `' + report['coordinator_sha256'] + '`.',
         'Paired precision bar: top-1 loss <0.005 / KL increase <0.005 nat, both as',
         'one-sided 95% bounds on both full shapes. These three policies are separate.',
         'The two compared arms here are repeated default-precision baselines, not',
         'a proposed precision change. No precision-default verdict or promotion is implied.', '']
readme = readme[:start] + '\n'.join(body) + '\n' + readme[end:]
(config / 'README.md').write_text(readme)
for entry in manifest['files']:
    assert digest(config / entry['path']) == entry['sha256']
assert digest(config / 'windows.json') == manifest['windows_sha256']
assert digest(config / 'qualification.json') == manifest['qualification_sha256']
tokenizer = Tokenizer.from_file(str(args.tokenizer))
for window in json.loads((config / 'windows.json').read_text())['windows']:
    validate_public_text(tokenizer.decode(window['tokens'], skip_special_tokens=False), scored_text=True)
def validate_output_file(path, config, manifest):
    # Preserve exact upstream licence notices, including public business contacts.
    licences = [entry for entry in manifest.get('licence_files', [])
                if entry['path'] == 'CHECKPOINT_LICENSE'
                and path.resolve() == (config / 'CHECKPOINT_LICENSE').resolve()]
    if licences:
        assert len(licences) == 1 and digest(path) == licences[0]['sha256'], 'Licence checksum differs'
    elif path.suffix == '.json':
        validate_public_metadata(json.loads(path.read_text()))
    elif path.name == 'README.md':
        validate_public_text(path.read_text(), scored_text=True)
    else:
        validate_public_text(path.read_text())


for path in OUT.rglob('*'):
    if path.is_file() and path.suffix != '.safetensors':
        validate_output_file(path, config, manifest)
print('Qualified config folder', config, 'manifestSHA', digest(config / 'manifest.json'))
print('No upload performed; production loader and coordinator audit still required')
