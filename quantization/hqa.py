"""HQA checkpoints: native attention plus compact-attention bundles.

CPU/IO-only. Derives `wrldsuksgo2mars/<parent>-HQA` from a pinned EXL3 Core-K6
parent: the main attention query up-projection and output projection (family
specific; see FAMILIES) are restored byte-for-byte from the pinned original
release and written to a few balanced `hqa-attention-*` shards. Every other
file is inherited: hard links inside the SparkNest HF cache locally, and
server-side `CommitOperationCopy` from the pinned parent on the Hub.

Two complete, mutually exclusive views are published:

  HQA (default)        model.safetensors.index.json, config.json, quantize_config.json
  --compact-attention  model.compact-attention.safetensors.index.json,
                       config.compact-attention.json, quantize_config.compact-attention.json

The compact index is the parent index byte-for-byte; the compact config and
quantization metadata are the parent's with only sibling-file references
redirected to the compact filenames. `open_view` is the explicit selection
harness implementing the loader contract until the CLI option exists.

Stages: build (inventory, shards, metadata, links), validate, card, publish.
Run state, receipts and logs live in ~/.cache/cuteafd/hqa/<destination>/.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_shard import read_header  # noqa: E402
from write_export import _fingerprint, _sync_dir  # noqa: E402

OWNER = 'wrldsuksgo2mars'
HUB = Path('/mnt/sparknest/hf-home/hub')
STATE = Path.home() / '.cache/cuteafd/hqa'
CHUNK = 64 << 20
SHARD_TARGET = 2_000_000_000
SCHEMA = 'cuteafd-hqa-v1'
VIEWS = {
    'hqa': dict(index='model.safetensors.index.json', config='config.json',
                quantize='quantize_config.json'),
    'compact': dict(index='model.compact-attention.safetensors.index.json',
                    config='config.compact-attention.json',
                    quantize='quantize_config.compact-attention.json'),
}
PARENT_CARD = 'README.parent.md'
PROVENANCE = 'hqa-provenance.json'
ELEMENT = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1, 'I16': 2, 'I32': 4, 'U8': 1, 'I8': 1}


@dataclasses.dataclass(frozen=True)
class Family:
    key: str
    name: str
    parent_rev: str
    source_repo: str
    source_rev: str
    source_cache_rev: str
    pattern: str            # restored module stems, group 1 = layer, group 2 = leaf
    expected: dict          # leaf -> module count
    native_dtype: str
    scales: tuple = ()      # required native scale suffixes beside `weight`
    block: int = 0          # FP8 scale block size

    @property
    def parent(self):
        return f'{OWNER}/{self.name}'

    @property
    def destination(self):
        return f'{OWNER}/{self.name}-HQA'


FAMILIES = {f.key: f for f in [
    Family('glm53-flash', 'GLM-5.3-Flash-EXL3-K4-Core-K6', '5d3fffe1df9c4a109bbbdafb60dc19df9fac019d',
           'zai-org/GLM-5.3-Flash-BF16', 'f9fed3bbc82291073373d67452e9f25317ef7577',
           'a5b45eb41df6402735dedc900be14a42e8d5e538',
           r'^model\.language_model\.layers\.(\d+)\.self_attn\.(q_b_proj|o_proj)$',
           {'q_b_proj': 12, 'o_proj': 46}, 'BF16'),
    Family('glm53', 'GLM-5.3-EXL3-K4-Core-K6', '5a4e83a09a11f3145332ed632d28a9894ea88225',
           'zai-org/GLM-5.3-BF16', '9d2398f478cab2de883137db3a36ad2c96205e24',
           '9d2398f478cab2de883137db3a36ad2c96205e24',
           r'^model\.layers\.(\d+)\.self_attn\.(q_b_proj|o_proj)$', {'q_b_proj': 79, 'o_proj': 79}, 'BF16'),
    Family('kimi-k3', 'Kimi-K3-EXL3-K2-Core-K6', '1a28050d2954426a43a3db705e53ca120e5f2b47',
           'moonshotai/Kimi-K3', 'f831ab66814297da540d832a5235f8e904f29d06',
           'f831ab66814297da540d832a5235f8e904f29d06',
           r'^language_model\.model\.layers\.(\d+)\.self_attn\.(q_b_proj|o_proj)$',
           {'q_b_proj': 24, 'o_proj': 93}, 'BF16'),
    Family('step5', 'Step-5-Preview-EXL3-K4-Core-K6', '83d47a384eabc07ea6fcfaac4779b0e5f013ce89',
           'stepfun-ai/Step-5-Preview-BF16', '7734b7ffac118d70d8398bfa3eb0fab007eb45b1',
           '7734b7ffac118d70d8398bfa3eb0fab007eb45b1',
           r'^model\.layers\.(\d+)\.self_attn\.(o_proj)$', {'o_proj': 95}, 'BF16'),
    Family('muse', 'Muse-Spark-Open-EXL3-K4-Core-K6', '89998fb177acd9a862ac8bd4fb79bfd50e940c97',
           'meta-models/Muse-Spark-Open-816B-A42B', '0fb02478b02565b2ce7fc57e6051c91172774ac9',
           '0fb02478b02565b2ce7fc57e6051c91172774ac9',
           r'^model\.language_model\.layers\.(\d+)\.self_attn\.(o_proj)$', {'o_proj': 62}, 'BF16'),
    Family('mistral-large4', 'Mistral-Large-4-EXL3-K4-Core-K6', '03c80aebaa483d55760703ae5bc0c9734c9d67d6',
           'mistralai/Mistral-Large-4-1T-A52B', 'f12dabdc0c1e99e7f7924461a6f308859b05af4d',
           'f12dabdc0c1e99e7f7924461a6f308859b05af4d',
           r'^layers\.(\d+)\.attention\.(wq|wo)$', {'wq': 72, 'wo': 72}, 'F8_E4M3', ('qscale_weight',), 128),
]}


class HqaError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def tensor_bytes(item):
    return item['data_offsets'][1] - item['data_offsets'][0]


class Run:
    """Paths, event log and the pinned inputs of one family assembly."""

    def __init__(self, family, *, hub=HUB, state=STATE, api=None, remote=None):
        self.family, self.hub, self.api = family, Path(hub), api
        self.state = Path(state) / (family.name + '-HQA')
        self.parent = self.hub / f'models--{OWNER}--{family.name}' / 'snapshots' / family.parent_rev
        self.source = (self.hub / ('models--' + family.source_repo.replace('/', '--'))
                       / 'snapshots' / family.source_cache_rev)
        self.root = self.hub / f'models--{OWNER}--{family.name}-HQA'
        self._remote = remote  # injected for tests: {(repo, rev): {path: {bytes, blob}}}
        self.state.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **fields):
        item = dict(event=kind, time=round(time.time(), 3), **fields)
        print(json.dumps(item), flush=True)
        with (self.state / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')

    def remote(self, repo, rev):
        """Remote tree of a pinned revision, cached once in run state."""
        if self._remote is not None:
            return self._remote[(repo, rev)]
        path = self.state / 'remote' / f'{repo.replace("/", "--")}@{rev}.json'
        if path.exists():
            return json.loads(path.read_text())
        from upload_model import _remote
        from huggingface_hub import HfApi
        tree = _remote(self.api or HfApi(), repo, rev)
        path.parent.mkdir(exist_ok=True)
        write_new(path, (canonical(tree) + '\n').encode())
        return tree


# ---------------------------------------------------------------- file writes

def write_new(path, data):
    """Publish `data` at a new directory entry; an existing entry must be identical.

    Never opens an existing (possibly hard-linked) file for writing.
    """
    path = Path(path)
    if path.is_symlink():
        raise HqaError(f'refuse symlink at {path}')
    if path.exists():
        if path.stat().st_size != len(data) or path.read_bytes() != data:
            raise HqaError(f'existing file differs: {path}')
        return False
    fd, temporary = tempfile.mkstemp(prefix='.hqa-', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), 0o644)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_dir(path.parent)
    finally:
        os.unlink(temporary)
    return True


def replace_generated(path, data):
    """Rewrite a file this tool generated (producer identity may change between runs).

    Only an unpublished, single-link staging file can be replaced; an inherited
    hard link (link count > 1) is never unlinked or rewritten here.
    """
    path = Path(path)
    if path.exists() and not path.is_symlink() and path.read_bytes() != data:
        if path.stat().st_nlink != 1:
            raise HqaError(f'refuse to replace a linked file: {path}')
        path.unlink()
        _sync_dir(path.parent)
    return write_new(path, data)


def json_style(raw):
    """Serialization settings that reproduce the parent's bytes, if any."""
    value = json.loads(raw)
    for indent, separators in ((None, (',', ':')), (2, (',', ': ')), (4, (',', ': ')), (None, (', ', ': '))):
        for ascii_only in (True, False):
            for newline in ('\n', ''):
                text = json.dumps(value, indent=indent, separators=separators, ensure_ascii=ascii_only) + newline
                if text.encode() == raw:
                    return dict(indent=indent, separators=separators, ensure_ascii=ascii_only, newline=newline)
    return dict(indent=None, separators=(',', ':'), ensure_ascii=False, newline='\n')


def ordered(value, like):
    """Keep a parent's sorted top-level key order when keys are added."""
    return dict(sorted(value.items())) if list(like) == sorted(like) else value


def dumps(value, style):
    style = dict(style)
    newline = style.pop('newline')
    return (json.dumps(value, **style) + newline).encode()


def redirect(raw, mapping):
    """Rewrite sibling-file reference strings in JSON bytes, nothing else.

    A reference is a whole string value equal to a mapped filename, optionally
    followed by `:<key>` (e.g. `quantize_config.json:tensor_storage`).
    """
    value = json.loads(raw)
    found = []
    names = '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    reference = re.compile(rf'^({names})((?::.*)?)$')

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            match = reference.match(node)
            if match and match[1] in mapping:
                found.append(node)
                return mapping[match[1]] + match[2]
        return node

    expected = walk(value)
    out = raw
    for old in sorted(set(found)):
        new = reference.sub(lambda m: mapping[m[1]] + m[2], old)
        token, replacement = json.dumps(old).encode(), json.dumps(new).encode()
        if out.count(token) != found.count(old):
            raise HqaError(f'reference {old!r} also appears outside JSON string values')
        out = out.replace(token, replacement)
    if json.loads(out) != expected:
        raise HqaError('reference redirect changed more than filename references')
    return out, sorted(set(found))


# ------------------------------------------------------------------ inventory

def headers_for(snapshot, files):
    out = {}
    for name in sorted(set(files)):
        header, base = read_header(snapshot / name)
        out[name] = (header, base)
    return out


def file_inventory(snapshot):
    files = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_dir() and not path.is_symlink():
            continue
        name = path.relative_to(snapshot).as_posix()
        real = Path(os.path.realpath(path))
        if not real.is_file():
            raise HqaError(f'parent entry does not resolve to a file: {name}')
        files[name] = dict(path=str(real), bytes=real.stat().st_size)
    return files


def main_layers(config):
    text = config.get('text_config', config)
    for key in ('num_hidden_layers', 'n_layers'):
        if key in text:
            return int(text[key])
    raise HqaError('config does not declare the main layer count')


def module_rewrite(patterns, universe, restored, leaves):
    """Remove restored modules from quantizer selection regexes, verified exactly."""
    out, changes = [], []
    for pattern in patterns:
        regex = re.compile(pattern)
        matched = {s for s in universe if regex.search(s)}
        if not matched & restored:
            out.append(pattern)
            continue
        if matched <= restored:
            changes.append(dict(removed=pattern))
            continue
        edited = pattern
        for leaf in leaves:
            edited = re.sub(r'\|' + re.escape(leaf) + r'(?=[|)])', '', edited)
            edited = re.sub(r'(?<=\(\?:)' + re.escape(leaf) + r'\|', '', edited)
        if {s for s in universe if re.search(edited, s)} != matched - restored:
            raise HqaError(f'cannot exclude restored modules from selection rule {pattern}')
        out.append(edited)
        changes.append(dict(rewrote=pattern, to=edited))
    selected = {s for s in universe if any(re.search(p, s) for p in out)}
    if selected & restored:
        raise HqaError('rewritten module selection still matches restored modules')
    return out, changes


def inventory(run):
    """Derive the replacement inventory from parent metadata and both headers."""
    fam = run.family
    parent_files = file_inventory(run.parent)
    raw = {name: Path(parent_files[name]['path']).read_bytes()
           for name in ('config.json', 'quantize_config.json', 'model.safetensors.index.json')}
    config, quantize, index = (json.loads(raw[n]) for n in
                               ('config.json', 'quantize_config.json', 'model.safetensors.index.json'))
    if 'transformers_weights' in config and config['transformers_weights'] != 'model.safetensors.index.json':
        raise HqaError('parent declares another authoritative index; family needs explicit handling')
    remote = run.remote(fam.parent, fam.parent_rev)
    if set(remote) - {'.gitattributes'} != set(parent_files) - {'.gitattributes'}:
        raise HqaError('local parent snapshot inventory differs from the pinned Hub revision')
    for name, record in parent_files.items():
        if name in remote and remote[name]['bytes'] != record['bytes']:
            raise HqaError(f'local parent file size differs from Hub: {name}')
        if not name.endswith('.safetensors') and name in remote:
            data = Path(record['path']).read_bytes()
            if remote[name]['blob'] not in (digest(data), git_blob(data)):
                raise HqaError(f'local parent metadata differs from Hub: {name}')
    weight_map, storage = index['weight_map'], quantize['tensor_storage']
    regex = re.compile(fam.pattern)
    layers = main_layers(config)
    stems = sorted(s for s in storage if regex.match(s))
    counts = {}
    for stem in stems:
        counts[regex.match(stem)[2]] = counts.get(regex.match(stem)[2], 0) + 1
    if counts != fam.expected:
        raise HqaError(f'restored module counts {counts} differ from {fam.expected}')
    # Every candidate module name in the parent index must be EXL3 storage.
    for name in weight_map:
        stem = name.rsplit('.', 1)[0]
        if regex.match(stem) and stem not in storage:
            raise HqaError(f'parent stores a matching projection outside EXL3: {name}')
    parent_shards = sorted(set(weight_map.values()))
    parent_headers = headers_for(run.parent, parent_shards)
    source_index = json.loads((run.source / 'model.safetensors.index.json').read_bytes())
    source_map = source_index['weight_map']
    source_names = {}
    for name in source_map:
        source_names.setdefault(name.rsplit('.', 1)[0], []).append(name)
    needed_source = sorted({source_map[f'{s}.weight'] for s in stems if f'{s}.weight' in source_map})
    source_headers = headers_for(run.source, needed_source)
    source_identity = verify_source(run, needed_source)
    modules = []
    for stem in stems:
        match = regex.match(stem)
        entry = storage[stem]
        old = {}
        for name, declared in sorted(entry['stored_tensors'].items()):
            shard = weight_map.get(name)
            if shard is None:
                raise HqaError(f'EXL3 buffer missing from parent index: {name}')
            item = parent_headers[shard][0].get(name)
            if item is None or item['shape'] != declared['shape']:
                raise HqaError(f'EXL3 buffer header differs: {name}')
            old[name] = dict(shard=shard, dtype=item['dtype'], shape=item['shape'], bytes=tensor_bytes(item))
        extra = {n for n in weight_map if n.rsplit('.', 1)[0] == stem} - set(old)
        if extra:
            raise HqaError(f'parent index has undeclared tensors for {stem}: {sorted(extra)}')
        wanted = {f'{stem}.weight', *(f'{stem}.{s}' for s in fam.scales)}
        present = set(source_names.get(stem, []))
        if present != wanted:
            raise HqaError(f'source package for {stem} is {sorted(present)}, expected {sorted(wanted)}')
        native = {}
        for name in sorted(wanted):
            shard = source_map[name]
            if shard not in source_headers:
                source_headers.update(headers_for(run.source, [shard]))
            header, base = source_headers[shard]
            item = header[name]
            start, end = item['data_offsets']
            native[name] = dict(dtype=item['dtype'], shape=item['shape'], bytes=end - start,
                                source_file=shard, source_extent=[base + start, base + end])
        weight = native[f'{stem}.weight']
        shape = [entry['out_features'], entry['in_features']]
        if weight['dtype'] != fam.native_dtype or weight['shape'] != shape:
            raise HqaError(f'{stem}.weight is {weight["dtype"]}{weight["shape"]}, expected {fam.native_dtype}{shape}')
        if weight['bytes'] != math.prod(shape) * ELEMENT[weight['dtype']]:
            raise HqaError(f'{stem}.weight byte extent differs from its shape')
        for suffix in fam.scales:
            scale = native[f'{stem}.{suffix}']
            blocks = [math.ceil(d / fam.block) for d in shape]
            if scale['dtype'] != 'F32' or scale['shape'] != blocks:
                raise HqaError(f'{stem}.{suffix} is {scale["dtype"]}{scale["shape"]}, expected F32{blocks}')
        layer = int(match[1])
        modules.append(dict(module=stem, leaf=match[2], layer=layer, mtp=layer >= layers,
                            compact_bits=entry['bits_per_weight'], compact=old, native=native,
                            native_bytes=sum(t['bytes'] for t in native.values()),
                            compact_bytes=sum(t['bytes'] for t in old.values())))
    return dict(schema=SCHEMA, family=fam.key, destination=fam.destination, parent=fam.parent,
                parent_rev=fam.parent_rev, source=fam.source_repo, source_rev=fam.source_rev,
                source_identity=source_identity, main_layers=layers, modules=modules,
                parent_index_sha256=digest(raw['model.safetensors.index.json']),
                parent_quantize_sha256=digest(raw['quantize_config.json']),
                parent_config_sha256=digest(raw['config.json']),
                parent_total_size=index['metadata']['total_size'])


def verify_source(run, files):
    """Bind cached source shards to the pinned revision through Hub LFS IDs."""
    fam = run.family
    pinned = run.remote(fam.source_repo, fam.source_rev)
    evidence = dict(pinned_revision=fam.source_rev, cached_revision=fam.source_cache_rev)
    if fam.source_cache_rev != fam.source_rev:
        cached = run.remote(fam.source_repo, fam.source_cache_rev)
        weights = {n for n in pinned if n.endswith('.safetensors')}
        if ({n for n in cached if n.endswith('.safetensors')} != weights
                or any(cached[n] != pinned[n] for n in weights)
                or cached['model.safetensors.index.json'] != pinned['model.safetensors.index.json']):
            raise HqaError('cached source revision is not tensor-equivalent to the pinned revision')
        evidence['equivalence'] = f'index and all {len(weights)} safetensors LFS SHA256 values equal'
    index = (run.source / 'model.safetensors.index.json').read_bytes()
    if pinned['model.safetensors.index.json']['blob'] not in (digest(index), git_blob(index)):
        raise HqaError('cached source index differs from the pinned revision')
    for name in files:
        path = run.source / name
        if path.stat().st_size != pinned[name]['bytes']:
            raise HqaError(f'cached source shard size differs: {name}')
        if path.is_symlink() and Path(os.readlink(path)).name != pinned[name]['blob']:
            raise HqaError(f'cached source shard object differs from pinned LFS SHA256: {name}')
    evidence['shards_bound_by_cache_object_id'] = len(files)
    return evidence


# ------------------------------------------------------------------- sharding

def shard_plan(modules):
    total = sum(m['native_bytes'] for m in modules)
    count = min(8, max(2, math.floor(total / SHARD_TARGET + 0.5)))
    if len(modules) < count:
        raise HqaError('fewer projection groups than shards')
    sizes, members = [0] * count, [[] for _ in range(count)]
    for module in sorted(modules, key=lambda m: (-m['native_bytes'], m['module'])):
        target = min(range(count), key=lambda i: (sizes[i], i))
        sizes[target] += module['native_bytes']
        members[target].append(module['module'])
    shards = []
    for i in range(count):
        if not members[i]:
            raise HqaError('empty attention shard')
        shards.append(dict(file=f'hqa-attention-{i + 1:05d}-of-{count:05d}.safetensors',
                           payload_bytes=sizes[i], modules=sorted(members[i])))
    return dict(count=count, total_payload_bytes=total, shards=shards,
                largest=max(sizes), smallest=min(sizes), spread=max(sizes) - min(sizes))


def shard_header(tensors):
    header, cursor = {'__metadata__': {'format': 'pt'}}, 0
    for name, item in tensors:
        header[name] = dict(dtype=item['dtype'], shape=item['shape'],
                            data_offsets=[cursor, cursor + item['bytes']])
        cursor += item['bytes']
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, 'little') + encoded, cursor


def shard_tensors(inv, shard):
    by_module = {m['module']: m for m in inv['modules']}
    return sorted((name, item) for stem in shard['modules']
                  for name, item in by_module[stem]['native'].items())


def _reclaim(stream, start, length):
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(stream.fileno(), start, length, os.POSIX_FADV_DONTNEED)


def write_shard(run, inv, shard, staging, records, plan_digest, *, fault_after=None):
    """Bounded raw-byte writer: header, streamed extents, hash, fsync, atomic link."""
    final, partial = staging / shard['file'], staging / (shard['file'] + '.partial')
    record_path = records / (shard['file'] + '.json')
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    if record is not None and record.get('plan') != plan_digest:
        raise HqaError(f'shard record belongs to another plan: {shard["file"]}')
    if final.exists():
        if record is None:
            raise HqaError(f'unexplained existing shard: {final}')
        if record.get('local') == _fingerprint(final):
            return record
        # Linked but not acknowledged: diagnose by re-hashing before accepting.
        with final.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256'] or final.stat().st_size != record['bytes']:
            raise HqaError(f'existing shard differs from its record: {final}')
        record['local'] = _fingerprint(final)
        record_path.unlink()
        write_new(record_path, (canonical(record) + '\n').encode())
        return record
    if partial.exists():
        run.event('partial_discarded', file=shard['file'], bytes=partial.stat().st_size)
        partial.unlink()
    if record is not None:
        record_path.unlink()
    tensors = shard_tensors(inv, shard)
    prefix, payload = shard_header(tensors)
    hasher, written = hashlib.sha256(prefix), 0
    fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, 'wb') as out:
        out.write(prefix)
        for name, item in tensors:
            start, end = item['source_extent']
            with (run.source / item['source_file']).open('rb') as source:
                source.seek(start)
                remaining = end - start
                while remaining:
                    chunk = source.read(min(CHUNK, remaining))
                    if not chunk:
                        raise HqaError(f'truncated source extent: {name}')
                    hasher.update(chunk)
                    out.write(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
                    if fault_after is not None and written >= fault_after:
                        raise KeyboardInterrupt('injected interruption')
                _reclaim(source, start, end - start)
        out.flush()
        os.fsync(out.fileno())
        _reclaim(out, 0, 0)
    if written != payload:
        raise HqaError('written payload size differs from header')
    record = dict(file=shard['file'], plan=plan_digest, sha256=hasher.hexdigest(),
                  bytes=len(prefix) + payload, header_bytes=len(prefix), payload_bytes=payload)
    write_new(record_path, (canonical(record) + '\n').encode())
    os.link(partial, final)
    partial.unlink()
    _sync_dir(staging)
    record['local'] = _fingerprint(final)
    record_path.unlink()
    write_new(record_path, (canonical(record) + '\n').encode())
    run.event('shard_written', file=shard['file'], bytes=record['bytes'], sha256=record['sha256'])
    return record


# ------------------------------------------------------------------- metadata

def attention_view():
    return dict(schema='cuteafd-attention-views-v1', mode='hqa',
                views={'hqa': dict(VIEWS['hqa']), 'compact-attention': dict(VIEWS['compact'])},
                native_modules='quantize_config.json:native_attention')


def native_description(fam, module, shard):
    tensors = {n: dict(dtype=t['dtype'], shape=t['shape']) for n, t in module['native'].items()}
    out = dict(format='bf16' if fam.native_dtype == 'BF16' else 'fp8_e4m3_block_scaled',
               tensors=tensors, shard=shard, compact_attention_bits=module['compact_bits'])
    if fam.scales:
        out['scale'] = dict(tensor=f'{module["module"]}.{fam.scales[0]}', dtype='F32',
                            block_size=[fam.block, fam.block],
                            semantics='checkpoint-native: original FP8 weight with its original per-block FP32 scale')
    return out


def build_metadata(run, inv, plan, shard_records):
    """All new small metadata, as {filename: bytes}; nothing is written here."""
    fam = run.family
    raw = {n: (run.parent / n).read_bytes() for n in
           ('config.json', 'quantize_config.json', 'model.safetensors.index.json')}
    index, quantize, config = (json.loads(raw[n]) for n in
                               ('model.safetensors.index.json', 'quantize_config.json', 'config.json'))
    removed = {n for m in inv['modules'] for n in m['compact']}
    shard_of = {stem: s['file'] for s in plan['shards'] for stem in s['modules']}
    added = {n: shard_of[m['module']] for m in inv['modules'] for n in m['native']}
    weight_map = {k: v for k, v in index['weight_map'].items() if k not in removed}
    if set(weight_map) & set(added):
        raise HqaError('native tensor names collide with parent entries')
    weight_map.update(added)
    if list(index['weight_map']) == sorted(index['weight_map']):
        weight_map = dict(sorted(weight_map.items()))
    total = (inv['parent_total_size'] - sum(m['compact_bytes'] for m in inv['modules'])
             + plan['total_payload_bytes'])
    hqa_index = dict(index)
    hqa_index['metadata'] = dict(index.get('metadata', {}), total_size=total)
    hqa_index['weight_map'] = weight_map
    restored = {m['module'] for m in inv['modules']}
    hqa_quant = dict(quantize)
    hqa_quant['tensor_storage'] = {k: v for k, v in quantize['tensor_storage'].items() if k not in restored}
    leaves = sorted({m['leaf'] for m in inv['modules']})
    rewrite = []
    if 'module_include' in quantize:
        hqa_quant['module_include'], rewrite = module_rewrite(
            quantize['module_include'], set(quantize['tensor_storage']), restored, leaves)
    for key in ('kept_tensors', 'retained_native_tensors'):
        if key not in quantize:
            continue
        declared = quantize[key]
        exl3 = {n for e in quantize['tensor_storage'].values() for n in e['stored_tensors']}
        if set(declared) != set(index['weight_map']) - exl3:
            raise HqaError(f'parent {key} is not an exhaustive native inventory')
        sample = next(iter(declared.values()))
        additions = {}
        for module in inv['modules']:
            for name, tensor in module['native'].items():
                if 'torch_dtype' in sample:
                    additions[name] = dict(shape=tensor['shape'], torch_dtype={
                        'BF16': 'bfloat16', 'F32': 'float32', 'F8_E4M3': 'float8_e4m3fn'}[tensor['dtype']])
                else:
                    additions[name] = dict(dtype=tensor['dtype'], shape=tensor['shape'])
        merged = dict(declared, **additions)
        hqa_quant[key] = dict(sorted(merged.items())) if list(declared) == sorted(declared) else merged
    view = attention_view()
    hqa_quant['attention_view'] = view
    hqa_quant['native_attention'] = {m['module']: native_description(fam, m, shard_of[m['module']])
                                     for m in inv['modules']}
    hqa_config = json.loads(raw['config.json'])
    if not isinstance(hqa_config.get('quantization_config'), dict):
        raise HqaError('parent config lacks inline quantization metadata')
    hqa_config['quantization_config']['attention_view'] = view
    hqa_quant, hqa_config = ordered(hqa_quant, quantize), ordered(hqa_config, config)
    hqa_config['quantization_config'] = ordered(hqa_config['quantization_config'], config['quantization_config'])
    files = {
        VIEWS['hqa']['index']: dumps(hqa_index, json_style(raw['model.safetensors.index.json'])),
        VIEWS['hqa']['quantize']: dumps(hqa_quant, json_style(raw['quantize_config.json'])),
        VIEWS['hqa']['config']: dumps(hqa_config, json_style(raw['config.json'])),
    }
    mapping = {VIEWS['hqa'][k]: VIEWS['compact'][k] for k in ('index', 'config', 'quantize')}
    compact_config, config_refs = redirect(raw['config.json'], mapping)
    compact_quant, quant_refs = redirect(raw['quantize_config.json'], mapping)
    compact = dict(config=dict(redirected=config_refs), quantize=dict(redirected=quant_refs))
    files[VIEWS['compact']['config']] = compact_config
    if compact_quant != raw['quantize_config.json']:
        files[VIEWS['compact']['quantize']] = compact_quant
    attributes = (run.parent / '.gitattributes').read_text()
    lines = attributes.rstrip('\n').split('\n')
    for role in ('index', 'config', 'quantize'):
        parent_name, compact_name = VIEWS['hqa'][role], VIEWS['compact'][role]
        if any(line.split()[:1] == [parent_name] and 'filter=lfs' in line for line in lines):
            entry = f'{compact_name} filter=lfs diff=lfs merge=lfs -text'
            if entry not in lines:
                lines.append(entry)
    files['.gitattributes'] = ('\n'.join(lines) + '\n').encode()
    return files, dict(hqa_total_size=total, module_include=rewrite, compact=compact)


def provenance(run, inv, plan, shard_records, summary):
    code = Path(__file__).resolve()
    try:
        commit = subprocess.run(['git', '-C', str(code.parent), 'rev-parse', 'HEAD'], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(code.parent), 'status', '--porcelain', '--', '.'],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    fam = run.family
    return dict(
        schema='cuteafd-hqa-provenance-v1', destination=fam.destination,
        meaning='HQA = High Quality Attention: restored native attention projections; '
                'compact-attention view = the pinned parent representation, unchanged.',
        parent=dict(repo=fam.parent, commit=fam.parent_rev, index_sha256=inv['parent_index_sha256'],
                    config_sha256=inv['parent_config_sha256'], quantize_config_sha256=inv['parent_quantize_sha256'],
                    payload_bytes=inv['parent_total_size'],
                    provenance_scope='quantization-provenance.json, quantization-errors/ and README.parent.md '
                                     'describe the parent; they apply to the compact-attention view and '
                                     'to every unchanged module, not to the restored native modules.'),
        source=dict(repo=fam.source_repo, commit=fam.source_rev, identity=inv['source_identity']),
        views={'hqa': dict(VIEWS['hqa'], payload_bytes=summary['hqa_total_size']),
               'compact-attention': dict(VIEWS['compact'], payload_bytes=inv['parent_total_size'],
                                         redirected_references=summary['compact'])},
        module_selection_rewrite=summary['module_include'],
        restored=[dict(module=m['module'], leaf=m['leaf'], layer=m['layer'], mtp=m['mtp'],
                       compact_attention=dict(bits=m['compact_bits'], tensors=m['compact']),
                       native=m['native']) for m in inv['modules']],
        scale_semantics=(None if not fam.scales else
                         f'{fam.native_dtype} weight with F32 {fam.scales[0]} per {fam.block}x{fam.block} block, '
                         'copied unchanged from the source'),
        shards=[dict(s, sha256=shard_records[s['file']]['sha256'], bytes=shard_records[s['file']]['bytes'])
                for s in plan['shards']],
        shard_balance=dict(count=plan['count'], largest=plan['largest'], smallest=plan['smallest'],
                           spread=plan['spread']),
        producer=dict(tool='quantization/hqa.py', code_sha256=digest(code.read_bytes()), git_commit=commit,
                      dirty=dirty, host=socket.gethostname(), device='cpu'))


def plan_files(run, inv, plan, metadata):
    """Published inventory: inherited {dest: parent name} and new names."""
    parent = file_inventory(run.parent)
    replaced = {'config.json', 'quantize_config.json', 'model.safetensors.index.json', 'README.md',
                '.gitattributes'}
    inherited = {n: n for n in parent if n not in replaced}
    inherited[PARENT_CARD] = 'README.md'
    inherited[VIEWS['compact']['index']] = 'model.safetensors.index.json'
    if VIEWS['compact']['quantize'] not in metadata:
        inherited[VIEWS['compact']['quantize']] = 'quantize_config.json'
    new = sorted({*metadata, PROVENANCE, 'README.md', *(s['file'] for s in plan['shards'])})
    if set(inherited) & set(new):
        raise HqaError('inherited and new inventories overlap')
    return inherited, new


# ------------------------------------------------------------------- assembly

def run_id(inv, plan):
    return 'run-' + digest(canonical(dict(inv=inv, plan=plan)).encode())[:16]


def link_inherited(run, staging, inherited):
    linked = 0
    for dest, name in sorted(inherited.items()):
        source = Path(os.path.realpath(run.parent / name))
        target = staging / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise HqaError(f'staging entry is a symlink: {dest}')
        if not target.exists():
            os.link(source, target)
        if not os.path.samefile(source, target):
            raise HqaError(f'inherited staging entry is not the parent object: {dest}')
        linked += target.stat().st_size
    _sync_dir(staging)
    return linked


def staging_inventory(staging):
    return {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file() or p.is_symlink()}


def build(run, *, fault_after=None):
    fam = run.family
    inv_path = run.state / 'inventory.json'
    inv = inventory(run)
    if inv_path.exists() and json.loads(inv_path.read_text()) != inv:
        raise HqaError('inventory differs from the recorded run; pinned inputs changed')
    write_new(inv_path, (canonical(inv) + '\n').encode())
    plan = shard_plan(inv['modules'])
    plan_digest = digest(canonical(dict(inv=inv, plan=plan)).encode())
    rid = run_id(inv, plan)
    staging = run.root / '.hqa-staging' / rid
    staging.mkdir(parents=True, exist_ok=True)
    records = run.state / 'shards'
    records.mkdir(exist_ok=True)
    write_new(run.state / 'plan.json', (canonical(dict(plan, run_id=rid, digest=plan_digest)) + '\n').encode())
    run.event('plan', run_id=rid, shards=plan['count'], total=plan['total_payload_bytes'],
              sizes=[s['payload_bytes'] for s in plan['shards']], spread=plan['spread'])
    shard_records = {s['file']: write_shard(run, inv, s, staging, records, plan_digest, fault_after=fault_after)
                     for s in plan['shards']}
    metadata, summary = build_metadata(run, inv, plan, shard_records)
    for name, data in metadata.items():
        write_new(staging / name, data)
    replace_generated(staging / PROVENANCE, (json.dumps(provenance(run, inv, plan, shard_records, summary),
                                                        indent=1, sort_keys=False) + '\n').encode())
    inherited, new = plan_files(run, inv, plan, metadata)
    linked = link_inherited(run, staging, inherited)
    present = staging_inventory(staging)
    expected = set(inherited) | set(new)
    unexpected = present - expected
    if unexpected:
        raise HqaError(f'unexplained staging entries: {sorted(unexpected)[:5]}')
    state = dict(run_id=rid, staging=str(staging), inherited=inherited, new=new, linked_bytes=linked,
                 new_bytes=sum((staging / n).stat().st_size for n in new if (staging / n).exists()),
                 summary=summary, shards=shard_records)
    run.event('assembled', run_id=rid, inherited=len(inherited), new=len(new), linked_bytes=linked)
    return inv, plan, state


# ---------------------------------------------------------- selection harness

class SelectionError(HqaError):
    pass


@dataclasses.dataclass
class View:
    mode: str
    identity: str
    index: dict
    quantize: dict
    config: dict
    opened: set
    dispatch: dict      # module -> 'exl3-K<bits>' or native format
    active_bytes: int


def open_view(root, mode, *, read=read_header):
    """Reference loader selection: one bundle, authoritative index, no fallbacks.

    Opens only the selected bundle's three files and the shards its index
    names; a shard header may hold other tensors, which are ignored.
    """
    root = Path(root)
    if mode not in VIEWS:
        raise SelectionError(f'unknown attention view {mode}')
    files = VIEWS[mode]
    default = root / VIEWS['hqa']['config']
    if not default.is_file():
        raise SelectionError('checkpoint has no default config.json')
    marker = json.loads(default.read_bytes()).get('quantization_config', {}).get('attention_view')
    if marker is None:
        raise SelectionError('not an HQA checkpoint: config.json declares no attention_view')
    if marker.get('views', {}).get('compact-attention') != VIEWS['compact'] or marker['views'].get('hqa') != VIEWS['hqa']:
        raise SelectionError('attention_view names an unexpected bundle')
    missing = [name for name in files.values() if not (root / name).is_file()]
    if missing:
        raise SelectionError(f'{mode} bundle incomplete: missing {missing}')
    raw = {role: (root / name).read_bytes() for role, name in files.items()}
    config, quantize, index = (json.loads(raw[r]) for r in ('config', 'quantize', 'index'))
    pointer = f'{files["quantize"]}:tensor_storage'
    for where, value in (('config', config.get('quantization_config', {}).get('per_module_metadata')),
                         ('quantize', quantize.get('per_module_metadata'))):
        if value is not None and value != pointer:
            raise SelectionError(f'{mode} bundle mixes views: {where} per_module_metadata is {value}')
    if mode == 'hqa':
        if config['quantization_config'].get('attention_view') != quantize.get('attention_view'):
            raise SelectionError('hqa config and quantization metadata disagree on attention_view')
        native = quantize.get('native_attention')
        if not isinstance(native, dict) or not native:
            raise SelectionError('hqa quantization metadata lacks native_attention')
    else:
        if 'attention_view' in config.get('quantization_config', {}) or 'attention_view' in quantize:
            raise SelectionError('compact bundle carries hqa metadata (mixed view)')
        if 'native_attention' in quantize:
            raise SelectionError('compact bundle carries native attention modules (mixed view)')
        native = {}
    weight_map, storage = index['weight_map'], quantize['tensor_storage']
    dispatch = {}
    for stem, entry in storage.items():
        for name in entry['stored_tensors']:
            if name not in weight_map:
                raise SelectionError(f'{mode}: EXL3 buffer {name} absent from the selected index')
        if f'{stem}.weight' in weight_map:
            raise SelectionError(f'{mode}: {stem} exposes both native and EXL3 storage')
        dispatch[stem] = f'exl3-K{entry["bits_per_weight"]}'
    for stem, entry in native.items():
        if stem in storage:
            raise SelectionError(f'{mode}: {stem} is both native and EXL3')
        for name in entry['tensors']:
            if name not in weight_map:
                raise SelectionError(f'{mode}: native tensor {name} absent from the selected index')
        exl3 = [n for n in weight_map if n.startswith(stem + '.') and n.rsplit('.', 1)[1] in
                ('trellis', 'suh', 'svh', 'mcg')]
        if exl3:
            raise SelectionError(f'{mode}: {stem} exposes both native and EXL3 storage')
        dispatch[stem] = entry['format']
    opened, active = set(), 0
    by_shard = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in sorted(by_shard.items()):
        path = root / shard
        if not path.is_file():
            raise SelectionError(f'{mode}: selected shard {shard} unavailable')
        header, _ = read(path)
        opened.add(shard)
        for name in names:
            item = header.get(name)
            if item is None:
                raise SelectionError(f'{mode}: {name} missing from {shard} header')
            expected = native.get(name.rsplit('.', 1)[0], {}).get('tensors', {}).get(name)
            if expected and (item['dtype'], item['shape']) != (expected['dtype'], expected['shape']):
                raise SelectionError(f'{mode}: {name} header differs from native_attention')
            active += tensor_bytes(item)
    if active != index['metadata']['total_size']:
        raise SelectionError(f'{mode}: indexed payload {active} != total_size {index["metadata"]["total_size"]}')
    identity = digest(b'cuteafd-attention-view-v1\0' + mode.encode() + b'\0'
                      + b''.join(digest(raw[r]).encode() for r in ('config', 'index', 'quantize')))
    return View(mode, identity, index, quantize, config, opened, dispatch, active)


def materialize_view(root, mode, target, *, canonical_names=True):
    """A directory exposing only one selected bundle under canonical role names.

    Used to drive loaders that have no selection option yet. Sibling-file
    references in the compact bundle are mapped back to the canonical names,
    which is exactly the selection layer's documented filename mapping.
    """
    root, target = Path(root), Path(target)
    view = open_view(root, mode)
    target.mkdir(parents=True, exist_ok=False)
    others = {n for v in VIEWS.values() for n in v.values()}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_dir() or name in others or name.endswith('.safetensors'):
            continue
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).symlink_to(path)
    for shard in sorted(view.opened):
        (target / shard).symlink_to(root / shard)
    reverse = {VIEWS['compact'][k]: VIEWS['hqa'][k] for k in ('index', 'config', 'quantize')}
    for role in ('index', 'config', 'quantize'):
        data = (root / VIEWS[mode][role]).read_bytes()
        if mode == 'compact' and canonical_names:
            data = redirect(data, reverse)[0]
        (target / VIEWS['hqa'][role]).write_bytes(data)
    return view


def _unredirect(value, mapping):
    if isinstance(value, dict):
        return {k: _unredirect(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_unredirect(v, mapping) for v in value]
    if isinstance(value, str):
        for old, new in mapping.items():
            if value == old or value.startswith(old + ':'):
                return new + value[len(old):]
    return value


# ----------------------------------------------------------------- validation

def compare_extents(run, inv, staging, plan):
    """Independent bounded comparison of every replacement tensor with its source."""
    compared = 0
    for shard in plan['shards']:
        path = staging / shard['file']
        header, base = read_header(path)
        names = set(header) - {'__metadata__'}
        tensors = dict(shard_tensors(inv, shard))
        if names != set(tensors):
            raise HqaError(f'{shard["file"]} tensor inventory differs')
        with path.open('rb') as out:
            for name, item in tensors.items():
                h = header[name]
                if h['dtype'] != item['dtype'] or h['shape'] != item['shape'] or tensor_bytes(h) != item['bytes']:
                    raise HqaError(f'{name} header differs from inventory')
                start, end = item['source_extent']
                out.seek(base + h['data_offsets'][0])
                with (run.source / item['source_file']).open('rb') as source:
                    source.seek(start)
                    remaining = end - start
                    while remaining:
                        n = min(CHUNK, remaining)
                        a, b = out.read(n), source.read(n)
                        if len(a) != n or a != b:
                            raise HqaError(f'{name} bytes differ from the pinned source')
                        remaining -= n
                    _reclaim(source, start, end - start)
                compared += item['bytes']
    return compared


def validate(run, inv, plan, state, *, planner=None):
    fam = run.family
    staging = Path(state['staging'])
    results = dict(family=fam.key, destination=fam.destination, run_id=state['run_id'])
    # Shard hashes, independent of the write pass.
    for name, record in state['shards'].items():
        with (staging / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                raise HqaError(f'{name} SHA256 differs from its write record')
            _reclaim(stream, 0, 0)
    results['shards'] = {n: dict(bytes=r['bytes'], sha256=r['sha256']) for n, r in state['shards'].items()}
    results['replacement_bytes_compared'] = compare_extents(run, inv, staging, plan)
    # Inherited identity and SparkNest residency.
    mount = os.stat(run.hub).st_dev
    for dest, name in state['inherited'].items():
        path = staging / dest
        if not os.path.samefile(path, os.path.realpath(run.parent / name)) or path.stat().st_dev != mount:
            raise HqaError(f'inherited {dest} is not a SparkNest link to the parent object')
    results['inherited_links'] = len(state['inherited'])
    # Both views through the selection harness.
    parent_index_raw = (run.parent / 'model.safetensors.index.json').read_bytes()
    if (staging / VIEWS['compact']['index']).read_bytes() != parent_index_raw:
        raise HqaError('compact index is not the parent index byte-for-byte')
    views = {mode: open_view(staging, mode) for mode in VIEWS}
    compact, hqa = views['compact'], views['hqa']
    parent_quant = json.loads((run.parent / 'quantize_config.json').read_bytes())
    parent_config = json.loads((run.parent / 'config.json').read_bytes())
    mapping = {VIEWS['compact'][k]: VIEWS['hqa'][k] for k in ('index', 'config', 'quantize')}
    if _unredirect(compact.quantize, mapping) != parent_quant or _unredirect(compact.config, mapping) != parent_config:
        raise HqaError('compact metadata differs from the parent beyond filename references')
    for role in ('config', 'quantize'):
        name = VIEWS['hqa'][role]
        if redirect((staging / VIEWS['compact'][role]).read_bytes(), mapping)[0] != (run.parent / name).read_bytes():
            raise HqaError(f'compact {role} does not map back to the parent bytes exactly')
    if compact.index['weight_map'] != json.loads(parent_index_raw)['weight_map'] or \
            compact.active_bytes != inv['parent_total_size']:
        raise HqaError('compact selection differs from the pinned parent')
    new_shards = {s['file'] for s in plan['shards']}
    if compact.opened & new_shards:
        raise HqaError('compact view opened native attention shards')
    if not new_shards <= hqa.opened:
        raise HqaError('hqa view does not open every native shard')
    restored = {m['module'] for m in inv['modules']}
    parent_map = compact.index['weight_map']
    removed = {n for m in inv['modules'] for n in m['compact']}
    added = {n for m in inv['modules'] for n in m['native']}
    shared = {k: v for k, v in parent_map.items() if k not in removed}
    if {k: v for k, v in hqa.index['weight_map'].items() if k not in added} != shared:
        raise HqaError('hqa index differs from the parent beyond the listed replacements')
    if set(hqa.index['weight_map']) - set(shared) != added:
        raise HqaError('hqa index native entries differ from the replacement inventory')
    for stem in restored:
        if not hqa.dispatch[stem].startswith(('bf16', 'fp8')) or not compact.dispatch[stem].startswith('exl3-K'):
            raise HqaError(f'{stem} dispatch differs from the contract')
    changed = {k for k in set(compact.dispatch) | set(hqa.dispatch)
               if compact.dispatch.get(k) != hqa.dispatch.get(k)}
    if changed != restored:
        raise HqaError('views dispatch differently outside the restored modules')
    if compact.identity == hqa.identity:
        raise HqaError('views share one identity')
    hqa_storage = {k: v for k, v in parent_quant['tensor_storage'].items() if k not in restored}
    if hqa.quantize['tensor_storage'] != hqa_storage:
        raise HqaError('hqa tensor_storage differs beyond the restored modules')
    results['views'] = {m: dict(active_bytes=v.active_bytes, identity=v.identity, shards_opened=len(v.opened),
                                native_shards_opened=len(v.opened & new_shards),
                                dispatch=_dispatch_summary(v, restored))
                        for m, v in views.items()}
    results['compact_equals_parent'] = dict(index='byte-identical', metadata='identical except redirected filenames',
                                            payload_bytes=compact.active_bytes)
    # Compact must load with native shards unavailable: drive the harness on a
    # materialized compact view that physically lacks them.
    views_dir = run.state / 'views' / state['run_id']
    if views_dir.exists():
        _remove_tree(views_dir)
    for mode in VIEWS:
        materialize_view(staging, mode, views_dir / mode)
    absent = open_view_canonical(views_dir / 'compact')
    if absent & new_shards:
        raise HqaError('compact canonical view references native shards')
    results['compact_without_native_shards'] = 'loads; native shards absent from the view directory'
    if planner:
        results['planner'] = run_planner(run, planner, views_dir)
    downloads = sum((staging / n).stat().st_size for n in staging_inventory(staging))
    safetensors = sum((staging / n).stat().st_size for n in staging_inventory(staging) if n.endswith('.safetensors'))
    results['sizes'] = dict(hqa_active=hqa.active_bytes, compact_active=compact.active_bytes,
                            download_total=downloads, download_safetensors=safetensors,
                            new_written=state['new_bytes'], inherited_linked=state['linked_bytes'],
                            native_payload=plan['total_payload_bytes'],
                            shard_sizes=[state['shards'][s['file']]['bytes'] for s in plan['shards']],
                            shard_spread=plan['spread'])
    results['status'] = 'passed'
    return results


def _dispatch_summary(view, restored):
    counts = {}
    for stem in restored:
        counts[view.dispatch[stem]] = counts.get(view.dispatch[stem], 0) + 1
    return counts


def open_view_canonical(directory):
    """Header-check a materialized single-view directory without selection."""
    index = json.loads((directory / 'model.safetensors.index.json').read_bytes())
    shards = set(index['weight_map'].values())
    for shard in shards:
        if not (directory / shard).exists():
            raise HqaError(f'materialized view lacks {shard}')
    present = {p.name for p in directory.iterdir() if p.name.endswith('.safetensors')}
    if present != shards:
        raise HqaError('materialized view holds unselected shards')
    return shards


def run_planner(run, planner, views_dir):
    """The real cuteafd planner on the parent and on each materialized view."""
    out = {}
    def plan(path):
        proc = subprocess.run([planner, 'plan', str(path)], capture_output=True, text=True, timeout=600)
        text = (proc.stdout + proc.stderr).replace(str(path), '<snapshot>')
        return proc.returncode, text
    base_rc, base = plan(run.parent)
    (run.state / 'plan-parent.txt').write_text(base)
    for mode in VIEWS:
        rc, text = plan(views_dir / mode)
        (run.state / f'plan-{mode}.txt').write_text(text)
        # New planner diagnostics relative to the parent, minus the per-count summary.
        blockers = sorted({line[2:] for line in text.splitlines()
                           if line.startswith('* ') and line not in base.splitlines()
                           and not line.startswith('* Spark count') and ': ' in line})
        out[mode] = dict(exit=rc, identical_to_parent=(rc == base_rc and text == base), blockers=blockers,
                         tail=text.strip().splitlines()[-3:] if rc else None)
    out['parent'] = dict(exit=base_rc)
    return out


def _remove_tree(path):
    for child in sorted(Path(path).rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink()
    Path(path).rmdir()


# ----------------------------------------------------------------------- card

def gb(n):
    return f'{n / 1e9:.2f} GB'


def front_matter(text):
    if not text.startswith('---\n') or '\n---\n' not in text[4:]:
        return []
    return text[4:text.index('\n---\n', 4)].splitlines()


def bits_text(modules):
    parts = []
    for mtp, label in ((False, 'main'), (True, 'MTP')):
        bits = sorted({m['compact_bits'] for m in modules if m['mtp'] == mtp})
        if bits:
            parts.append('/'.join(f'K{b}' for b in bits) + ' ' + label)
    return ' EXL3, '.join(parts) + ' EXL3'


def planner_text(results):
    planner = results.get('planner')
    if not planner:
        return ''
    compact = planner['compact']
    hqa_ok = planner['hqa']['exit'] == 0 and not planner['hqa'].get('blockers')
    text = (f'- The cuteafd planner (`cuteafd plan` at {results.get("planner_revision", "this branch")}) on a '
            'directory exposing only the compact bundle under the canonical filenames, with the native shards '
            f'absent, {"reproduced the pinned parent plan exactly" if compact["identical_to_parent"] else "did not reproduce the parent plan"}. ')
    if hqa_ok:
        text += 'The same planner accepted the HQA bundle.\n'
    else:
        text += ('The same planner does **not** yet execute the HQA bundle: '
                 + '; '.join(f'`{b}`' for b in planner['hqa'].get('blockers') or ['see provenance'])
                 + '. That is the separate family loader work; the artifact contract does not depend on it.\n')
    return text


def card(run, inv, plan, state, results):
    fam = run.family
    meta = front_matter((run.parent / 'README.md').read_text())
    if 'tags:' in meta:
        at = meta.index('tags:') + 1
        while at < len(meta) and meta[at].startswith('- '):
            at += 1
        meta.insert(at, '- hqa')
    else:
        meta += ['tags:', '- hqa']
    leafs = {}
    for m in inv['modules']:
        leafs.setdefault(m['leaf'], []).append(m)
    s = results['sizes']
    rows = '\n'.join(
        f'| `{leaf}` | {len(ms)} | {sum(not m["mtp"] for m in ms)} main + {sum(m["mtp"] for m in ms)} MTP | '
        f'{bits_text(ms)} | {fam.native_dtype}'
        f'{" + F32 " + fam.scales[0] + " (block " + str(fam.block) + ")" if fam.scales else ""} |'
        for leaf, ms in sorted(leafs.items()))
    shard_rows = '\n'.join(f'| `{x["file"]}` | {x["bytes"]:,} | `{x["sha256"]}` |'
                           for x in results['shard_table'])
    views = results['views']
    ptext = planner_text(results)
    lines = ['---', *meta, '---']
    body = f'''
# {fam.name}-HQA

**HQA = High Quality Attention.** This is [{fam.parent}](https://huggingface.co/{fam.parent}/tree/{fam.parent_rev})
(pinned at `{fam.parent_rev}`) with its main attention projections restored to the original
release's own weights: native {fam.native_dtype} from [{fam.source_repo}](https://huggingface.co/{fam.source_repo}/tree/{fam.source_rev})
at `{fam.source_rev}`, copied byte-for-byte{" together with their original FP32 block-" + str(fam.block) + " scales" if fam.scales else ""}
(no casting, requantization, padding or transposition). The expert quantization and every other tensor are the
parent's, unchanged. HQA describes the format and fidelity of these tensors; it is not a measured benchmark claim.

## Restored tensors

| Projection | Modules | Layers | Compact-attention format | HQA format |
| --- | ---: | --- | --- | --- |
{rows}

{len(inv['modules'])} projections in total. The compact view keeps the parent's quantization of these
modules exactly, including the lower-bit MTP attention where the parent used it. Indexer projections,
draft weights that were already native, and all other modules are untouched.

## Two attention views

This repository holds two complete, mutually exclusive views of the same files:

| View | Tensor index | Model config | Quantization metadata | Active payload |
| --- | --- | --- | --- | ---: |
| HQA (default) | `{VIEWS['hqa']['index']}` | `{VIEWS['hqa']['config']}` | `{VIEWS['hqa']['quantize']}` | {gb(s['hqa_active'])} |
| compact-attention | `{VIEWS['compact']['index']}` | `{VIEWS['compact']['config']}` | `{VIEWS['compact']['quantize']}` | {gb(s['compact_active'])} |

The default view maps each restored projection to its native tensors in the `hqa-attention-*` shards and
omits that projection's EXL3 buffers. The compact-attention index is the parent's index byte-for-byte: it maps
the original EXL3 attention buffers (which remain in the inherited shards, deliberately, for this view) and
never references the `hqa-attention-*` shards, so a compact download can skip them. Its config and quantization
metadata are the parent's, with only sibling-file references redirected to the compact filenames.
These alternate filenames are a cuteafd convention, not a Hugging Face mode switch: generic loaders read the
default files only, and a valid safetensors index does not imply generic Transformers/EXL3 support.

**Loader support.** A loader must select one bundle (index, config and quantization metadata together) for the
lifetime of the loaded model, treat the selected index as authoritative, and refuse a missing or mixed bundle.
The cuteafd `--compact-attention` option that selects the compact bundle is **future work and is not implemented
yet**; per-family cuteafd loader support for the HQA view lands separately. Until then this checkpoint is
published for that work; do not assume a current cuteafd release serves either view of it.

## Sizes

| | Bytes | GB |
| --- | ---: | ---: |
| HQA active payload | {s['hqa_active']:,} | {gb(s['hqa_active'])} |
| compact-attention active payload | {s['compact_active']:,} | {gb(s['compact_active'])} |
| Full download (all files) | {s['download_total']:,} | {gb(s['download_total'])} |
| Full download, safetensors only | {s['download_safetensors']:,} | {gb(s['download_safetensors'])} |
| New native attention shards | {sum(s['shard_sizes']):,} | {gb(sum(s['shard_sizes']))} |

The views share almost all bytes; do not add them. New shards (balanced across {plan['count']}, spread
{plan['spread']:,} bytes):

| Shard | Bytes | SHA256 |
| --- | ---: | --- |
{shard_rows}

## Validation performed

- Every replacement tensor was compared byte-for-byte with its pinned source extent
  ({results['replacement_bytes_compared']:,} bytes); shapes, dtypes{", scale pairing" if fam.scales else ""} and shard hashes checked.
- Both views were opened through an explicit selection harness implementing the contract above: every indexed
  tensor resolves in its file header; each view's payload equals its `total_size`; no projection exposes both
  native and EXL3 storage; the compact view equals the pinned parent's tensor selection and payload exactly and
  opens none of the native shards (and loads with them physically absent); the views have distinct identities.
- Inherited files are server-side copies of the pinned parent. Publication completes only after the remote tree
  matches the planned inventory exactly: inherited LFS SHA256 values equal the pinned parent's, and new files
  equal the local write records.
{ptext}- Not performed: whole-model generation, quality or performance evaluation of the HQA view.

## Provenance

`{PROVENANCE}` binds both views, the parent and source revisions, every restored tensor's source byte extent,
shard membership and hashes, and the producing tool. `{PARENT_CARD}`, `quantization-provenance.json` and
`quantization-errors/` are the parent's records: they describe the compact-attention view and every unchanged
module, not the restored native modules. License and usage files are inherited unchanged from the parent.

Produced by `quantization/hqa.py` in [cuteafd](https://github.com/tpurtell/cuteafd) (CPU and I/O only).
'''
    return ('\n'.join(lines) + '\n' + body).encode()


# -------------------------------------------------------------------- publish

def publish(run, inv, plan, state, results):
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from upload_model import _remote, _restored_operation
    from materialize_cache import materialize_cache
    fam = run.family
    api = run.api or HfApi()
    staging = Path(state['staging'])
    parent_remote = run.remote(fam.parent, fam.parent_rev)
    if _remote(api, fam.parent, fam.parent_rev) != parent_remote:
        raise HqaError('pinned parent revision on the Hub differs from the recorded tree')
    repo = fam.destination
    parent_path = run.state / 'upload-parent.json'
    if parent_path.exists():
        guard = json.loads(parent_path.read_text())
    else:
        try:
            info = api.repo_info(repo, repo_type='model')
        except RepositoryNotFoundError:
            api.create_repo(repo, repo_type='model', private=False)
            info = api.repo_info(repo, repo_type='model')
        if info.private or set(_remote(api, repo, info.sha)) - {'.gitattributes'}:
            raise HqaError('destination must be a new, empty, public repository')
        guard = dict(commit=info.sha)
        write_new(parent_path, (canonical(guard) + '\n').encode())
    operations, prepared = [], {}
    for dest, name in sorted(state['inherited'].items()):
        operations.append(CommitOperationCopy(src_path_in_repo=name, path_in_repo=dest, src_revision=fam.parent_rev,
                                              src_repo_id=fam.parent, src_repo_type='model'))
        prepared[dest] = dict(parent_remote[name], local=_fingerprint(staging / dest))
    uploads = run.state / 'preuploaded'
    uploads.mkdir(exist_ok=True)
    for name in state['new']:
        path = staging / name
        local = _fingerprint(path)
        record_path = uploads / (name.replace('/', '--') + '.json')
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        if record is None:
            run.event('upload_started', file=name, bytes=local['bytes'])
            op = CommitOperationAdd(name, str(path))
            api.preupload_lfs_files(repo, [op], repo_type='model', revision='main', num_threads=1,
                                    free_memory=False, gitignore_content='')
            if op._should_ignore or op._upload_mode not in ('lfs', 'regular'):
                raise HqaError('unexpected upload classification')
            if op._upload_mode == 'lfs':
                if not op._is_uploaded or not op.upload_info.is_hashed:
                    raise HqaError('new payload upload not acknowledged')
                blob = op.upload_info.sha256.hex()
                if name.endswith('.safetensors') and blob != state['shards'][name]['sha256']:
                    raise HqaError('uploaded shard hash differs from the write record')
            else:
                if local['bytes'] > 64 << 20:
                    raise HqaError('unexpected large regular upload')
                blob = git_blob(path.read_bytes())
            record = dict(mode=op._upload_mode, bytes=local['bytes'], blob=blob, local=local)
            write_new(record_path, (canonical(record) + '\n').encode())
            run.event('upload_prepared', file=name, mode=record['mode'], bytes=local['bytes'])
        else:
            if record['local'] != local:
                raise HqaError(f'prepared upload changed locally: {name}')
            op = _restored_operation(name, str(path), record)
        prepared[name] = {k: record[k] for k in ('bytes', 'blob', 'local')}
        operations.append(op)
    expected = {n: {k: r[k] for k in ('bytes', 'blob')} for n, r in prepared.items()}
    info = api.repo_info(repo, repo_type='model')
    if info.private:
        raise HqaError('destination became private')
    if info.sha != guard['commit']:
        if _remote(api, repo, info.sha) != expected:
            raise HqaError('destination head moved and differs from the planned inventory')
        commit = info.sha
        run.event('commit_recovered', commit=commit)
    else:
        commit = api.create_commit(repo, operations, repo_type='model', revision='main',
                                   parent_commit=guard['commit'],
                                   commit_message='Publish HQA checkpoint: native attention default view and '
                                                  'compact-attention bundle; parent files copied server-side').oid
        run.event('committed', commit=commit)
    remote = _remote(api, repo, commit)
    if remote != expected:
        diff = sorted(n for n in set(remote) | set(expected) if remote.get(n) != expected.get(n))
        raise HqaError(f'published tree differs from the planned inventory: {diff[:5]}')
    anonymous = HfApi(token=False)
    if anonymous.model_info(repo, revision=commit).sha != commit:
        raise HqaError('published revision is not anonymously readable')
    paths = [p.path for p in anonymous.get_paths_info(repo, [*VIEWS['hqa'].values(), *VIEWS['compact'].values(),
                                                            'README.md', PROVENANCE], revision=commit)]
    if len(paths) != 8:
        raise HqaError('anonymous metadata access is incomplete')
    receipt = dict(schema='cuteafd-upload-receipt-v1', status='uploaded', repo_id=repo, commit=commit,
                   files=prepared)
    write_new(run.state / 'upload-complete.json', (canonical(receipt) + '\n').encode())
    cached = materialize_cache(staging, run.hub, receipt)
    snapshot = Path(cached['snapshot'])
    # Retire staging links; blobs keep every object alive.
    blobs = run.root / 'blobs'
    for name, record in prepared.items():
        if not os.path.samefile(staging / name, blobs / record['blob']):
            raise HqaError(f'cache blob is not the staged object: {name}')
    new_blobs = sorted({prepared[n]['blob'] for n in state['new']})
    _remove_tree(staging)
    stage_parent = run.root / '.hqa-staging'
    if stage_parent.exists() and not any(stage_parent.iterdir()):
        stage_parent.rmdir()
    sealed = seal(blobs, new_blobs)
    result = dict(status='complete', repo_id=repo, commit=commit, url=f'https://huggingface.co/{repo}',
                  snapshot=str(snapshot), inherited_copied_server_side=len(state['inherited']),
                  new_uploaded_bytes=sum(prepared[n]['bytes'] for n in state['new']), sealed_new_objects=sealed,
                  remote_lfs_sha256={n: r['blob'] for n, r in expected.items() if len(r['blob']) == 64})
    write_new(run.state / 'publication-complete.json', (canonical(result) + '\n').encode())
    run.event('published', repo=repo, commit=commit)
    return result


def seal(blobs, names):
    sealed = 0
    for name in names:
        proc = subprocess.run(['nest', 'seal', str(blobs / name)], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise HqaError(f'nest seal failed for {name}: {proc.stderr.strip()[:200]}')
        sealed += 1
    return sealed


# ----------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', required=True, choices=sorted(FAMILIES))
    parser.add_argument('--stage', choices=['build', 'validate', 'card', 'publish'], default='validate',
                        help='run every stage up to and including this one')
    parser.add_argument('--planner', help='cuteafd binary for planner checks of both materialized views')
    args = parser.parse_args(argv)
    run = Run(FAMILIES[args.family])
    with (run.state / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run.event('started', stage=args.stage, family=args.family)
        inv, plan, state = build(run)
        if args.stage == 'build':
            return
        results = validate(run, inv, plan, state, planner=args.planner)
        results['shard_table'] = [dict(file=s['file'], bytes=state['shards'][s['file']]['bytes'],
                                       sha256=state['shards'][s['file']]['sha256']) for s in plan['shards']]
        if args.planner:
            results['planner_revision'] = 'cuteafd ' + subprocess.run(
                ['git', '-C', str(Path(__file__).parent), 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True).stdout.strip()
        path = run.state / f'validation-{state["run_id"]}.json'
        if path.exists():
            path.unlink()
        write_new(path, (json.dumps(results, indent=1) + '\n').encode())
        run.event('validated', sizes=results['sizes'])
        if args.stage == 'validate':
            return
        staging = Path(state['staging'])
        if (run.state / 'upload-parent.json').exists() and (staging / 'README.md').exists():
            write_new(staging / 'README.md', card(run, inv, plan, state, results))
        else:
            replace_generated(staging / 'README.md', card(run, inv, plan, state, results))
        state['new_bytes'] = sum((staging / n).stat().st_size for n in state['new'])
        if staging_inventory(staging) != set(state['inherited']) | set(state['new']):
            raise HqaError('staging inventory differs from the planned publication')
        open_view(staging, 'hqa'), open_view(staging, 'compact')
        if args.stage == 'card':
            return
        result = publish(run, inv, plan, state, results)
        print(json.dumps(dict(result, sizes=results['sizes']), indent=1))


if __name__ == '__main__':
    main()
