"""Startup argument rewrites cannot silently fall back to report launch defaults."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DAEMON = ROOT / 'rust/crates/cuteafd-daemon/src'


def production(source):
    """Mask comments and test items, retaining offsets and later production items."""
    tokens = re.compile(r'''//[^\n]*|/\*[\s\S]*?\*/|\b(?:br|r)(\#*)"[\s\S]*?"\1|"(?:\\[\s\S]|[^"\\])*"|'(?:\\.|[^'\\])' '''.rstrip())
    masked = tokens.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
    # Attributes contain no strings here; keep them available for cfg discovery.
    for match in reversed(list(re.finditer(r'#\[cfg\(test\)\]', masked))):
        item = masked[match.end():].lstrip()
        if not re.match(r'(?:(?:pub(?:\([^)]*\))?|async|unsafe)\s+)*(?:mod|fn)\b', item):
            continue
        start = masked.find('{', match.end())
        semicolon = masked.find(';', match.end())
        if semicolon >= 0 and (start < 0 or semicolon < start):
            end = semicolon + 1
            source = source[:match.start()] + ' ' * (end - match.start()) + source[end:]
            masked = masked[:match.start()] + ' ' * (end - match.start()) + masked[end:]
            continue
        if start < 0:
            raise AssertionError('test item has no body')
        depth, end = 1, start + 1
        while depth:
            depth += (masked[end] == '{') - (masked[end] == '}')
            end += 1
        source = source[:match.start()] + ' ' * (end - match.start()) + source[end:]
        masked = masked[:match.start()] + ' ' * (end - match.start()) + masked[end:]
    return source, masked


def argument_fields(source):
    fields = {}
    for match in re.finditer(r'#\[arg\((.*?)\)\]\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:', source, re.S):
        attributes, field = match.groups()
        if re.search(r'\bskip\b', attributes) or not re.search(r'\blong\b', attributes):
            continue
        explicit = re.search(r'\blong\s*=\s*"([^"]+)"', attributes)
        fields[field] = explicit[1] if explicit else field.replace('_', '-')
    return fields


def registry():
    source = (DAEMON / 'commands/bench.rs').read_text()
    block = source.split('const DERIVED_ARGUMENTS:', 1)[1].split('\n];', 1)[0]
    rules = {}
    for family, names in re.findall(r'family: (None|Some\("[^"]+"\)), names: &\[(.*?)\]', block, re.S):
        rules[None if family == 'None' else family.split('"')[1]] = set(re.findall(r'"([^"]+)"', names))
    assert rules and None in rules
    return rules


def test_every_rewritten_serve_argument_is_marked_derived():
    rules = registry()
    common = {}
    for path in (DAEMON / 'shared').rglob('*.rs'):
        common.update(argument_fields(production(path.read_text())[0]))
    missing = []
    for directory in sorted((DAEMON / 'families').iterdir()):
        if not directory.is_dir():
            continue
        sources = {path: production(path.read_text()) for path in directory.rglob('*.rs')}
        fields = dict(common)
        for source, _ in sources.values():
            fields.update(argument_fields(source))
        derived = rules[None] | rules.get(directory.name, set())
        for path, (_, code) in sources.items():
            # Only parsed argument objects: engine state and request fields are not launch options.
            rewrite = r'\b(?:args|engine_args|resolved|selected_args|context_args)\.(?:\w+\.)*(\w+)\s*(?:=(?!=)|\.(?:clear|take|get_or_insert|push)\()'
            for match in re.finditer(rewrite, code):
                field = match[1]
                if field not in fields:
                    continue
                # These are diagnostic-only commands, never dispatched by serve.
                functions = list(re.finditer(r'\bfn\s+(\w+)\s*(?:<[^\n]*>)?\(', code[:match.start()]))
                function = functions[-1][1] if functions else None
                if path.name in {'packed_check.rs', 'golden.rs'} or function in {'golden', 'run_golden'} or field == 'nll':
                    continue
                if fields[field] not in derived:
                    line = code.count('\n', 0, match.start()) + 1
                    missing.append(f'{path.relative_to(ROOT)}:{line}: {fields[field]}')
    assert not missing, 'mark rewritten serve arguments in DERIVED_ARGUMENTS:\n' + '\n'.join(missing)


def test_rewrite_scan_keeps_production_after_tests_and_ignores_test_mutations():
    source = '''#[cfg(test)] mod tests { fn fixture() { args.slots = 1; } }
fn normalize() { args.rings = 16; }
#[cfg(test)] fn test() { args.rings = 2; }
fn later() { args.max_context = 512; }'''
    _, code = production(source)
    assert 'args.slots' not in code
    assert 'args.rings = 16' in code
    assert 'args.rings = 2' not in code
    assert 'args.max_context = 512' in code


def test_registry_names_have_runtime_publication_and_family_seam_coverage():
    rules = registry()
    sources = [production(path.read_text())[0] for path in DAEMON.rglob('*.rs')]
    sources.append((ROOT / 'rust/crates/cuteafd-bench/src/context.rs').read_text())
    # Publication records may be emitted centrally (KV, placement, prefix and prefill).
    published = set()
    for source in sources:
        published.update(re.findall(r'(?:set_resolved|\.value)\(\s*"([^"]+)"', source))
        # Compact loops publish one (name, value) record per element.
        for loop in re.finditer(r'for \(name, value(?:, default)?\) in \[([\s\S]*?)\]\s*\{', source):
            published.update(re.findall(r'\("([^"]+)"\s*,', loop[1]))
    # Native flag records are collected by native::resolved_settings before publication.
    native = (DAEMON / 'families/glm5_flash/native.rs').read_text()
    published.update(re.findall(r'\("([^"]+)"\s*,', native))
    missing = set.union(*rules.values()) - published
    assert not missing, f'derived settings without runtime publication: {sorted(missing)}'
    for directory in (DAEMON / 'families').iterdir():
        if directory.is_dir() and (directory / 'serve.rs').exists() or directory.name == 'deepseek_v41':
            source = '\n'.join(path.read_text() for path in directory.rglob('*.rs'))
            assert 'assert_resolved_family_seam(' in source, f'{directory.name}: exercise a CPU startup seam'
