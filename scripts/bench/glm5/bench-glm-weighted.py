#!/usr/bin/env python3
"""glmrt v9's weighted decode benchmark against any OpenAI chat endpoint.

The corpus, request and aggregate are glmrt-release v9's
python/tools/bench_real_full_mtp_acceptance.py (weighted suite): eight cases,
temperature 0, thinking off, one request at a time, each prompt behind a
token-zero nonce (a CJK marker that is the prompt's first token) so no
request reuses a cached prefix. Per request, timed tokens are
completion_tokens - 1 (the first token belongs to the prefill) over the decode
time after the first token; the aggregate pools them:

    weighted tok/s = sum(w * (completion_tokens - 1)) / sum(w * decode_seconds)

--timing stream (default) streams the response and times the first output
delta to the finish on the client, which any server supports.
--timing server posts without streaming and uses the server's
metrics.decode_ms, which is exactly glmrt's number (glmrt only). glmrt v9
published 33.42 with --repeats 5 --nonce-seed 2026090801.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

# (id, max_tokens, weight, json_schema, prompt): glmrt v9 WEIGHTED_CASE_IDS in order.
CASES = [
    ('code', 320, 1.0, False,
     'Write a Python function merge_intervals(intervals) that merges overlapping integer intervals. Include type '
     'hints, a short docstring, and three assert-based examples. Return only one Python code block.'),
    ('math', 128, 1.0, False,
     'A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the discounted price. What is the final '
     'price? Show the calculation briefly.'),
    ('fable', 256, 1.0, False,
     'Write a self-contained fable of exactly 150 words about two parrots who disagree about sharing credit. Output '
     'no title or preamble. Include the final one-sentence moral in the 150-word total. Before responding, silently '
     'revise the draft until the entire response is between 140 and 170 words.'),
    ('hello', 32, 1.0, False, 'hi'),
    ('topic', 384, 1.0, False,
     'Explain virtual memory to a junior programmer in five concise bullet points, including paging, page faults, '
     'and the role of the TLB.'),
    ('structured-json', 128, 0.5, False,
     'Return only a JSON object describing a file edit with keys path, operation, line_start, line_end, and '
     'rationale. Use path src/cache.rs, operation replace, lines 41 through 47, and a one-sentence rationale about '
     'removing a redundant copy.'),
    ('structured-json-schema', 128, 0.5, True,
     'Return only a JSON object describing a file edit with keys path, operation, line_start, line_end, and '
     'rationale. Use path src/cache.rs, operation replace, lines 41 through 47, and a one-sentence rationale about '
     'removing a redundant copy.'),
    ('multilingual', 384, 1.0, False,
     '請用繁體中文，以四個簡短條列解釋什麼是寫入時複製（copy-on-write），並包含一個行程 fork 後修改記憶體頁面的例子。'),
]
WEIGHTS = {case[0]: case[2] for case in CASES}
SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'file_edit', 'strict': True, 'schema': {
    'type': 'object',
    'properties': {'path': {'type': 'string'}, 'operation': {'type': 'string'}, 'line_start': {'type': 'integer'},
                   'line_end': {'type': 'integer'}, 'rationale': {'type': 'string'}},
    'required': ['path', 'operation', 'line_start', 'line_end', 'rationale'], 'additionalProperties': False}}}
MORAL_TERMS = ('credit', 'share', 'sharing', 'together', 'cooperat', 'team', 'recognition', 'praise', 'glory',
               'harmony', 'humility', 'fair', 'both')


def token_zero_nonces(count, seed, tokenizer_path):
    """glmrt's token-zero nonces: a CJK marker that encodes to one token and
    is the first token of its prefix, distinct for every request."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    first, candidates = 0x4E00, 0x9FFF - 0x4E00 + 1
    nonces, seen = [], set()
    for offset in range(candidates):
        marker = chr(first + (seed % candidates + offset) % candidates)
        prefix = f'{marker} request nonce {seed}-{len(nonces)}.\n'
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        if not encoded or encoded[0] in seen:
            continue
        marker_ids = tokenizer.encode(marker, add_special_tokens=False).ids
        if len(marker_ids) != 1 or encoded[0] != marker_ids[0]:
            continue
        seen.add(encoded[0])
        nonces.append(prefix)
        if len(nonces) == count:
            return nonces
    raise SystemExit(f'only {len(nonces)} token-zero nonces exist; {count} needed')


def _python_block(content):
    match = re.fullmatch(r'\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*', content, flags=re.DOTALL | re.IGNORECASE)
    return (None, ['response is not exactly one Python code block']) if match is None else (match.group('code'), [])


def validate(case_id, content):
    """glmrt v9's prompt-visible output contract (glmrt-semantic-decode-contract-v3)."""
    issues, stripped = [], content.strip()
    if not stripped:
        return ['response is empty']
    if case_id == 'code':
        code, issues = _python_block(content)
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return issues + ['Python code does not parse']
            functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == 'merge_intervals']
            if len(functions) != 1:
                issues.append('merge_intervals function is missing or duplicated')
            else:
                f = functions[0]
                if not f.args.args or f.args.args[0].annotation is None or f.returns is None:
                    issues.append('merge_intervals lacks requested type hints')
                if ast.get_docstring(f) is None:
                    issues.append('merge_intervals lacks a docstring')
            if sum(isinstance(n, ast.Assert) for n in ast.walk(tree)) < 3:
                issues.append('fewer than three assert examples were provided')
    elif case_id == 'math':
        normalized = stripped.replace(',', '')
        if re.search(r'(?<![0-9])(?:\$\s*)?194\.4(?:0)?(?![0-9])', normalized) is None:
            issues.append('response does not contain the correct final price 194.40')
        if ('240' not in normalized or not any(t in normalized for t in ('25', '75', '0.75', '.75'))
                or not any(t in normalized for t in ('8', '1.08'))):
            issues.append('response does not show the requested calculation inputs')
    elif case_id == 'fable':
        words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
        if not 140 <= len(words) <= 170:
            issues.append(f'fable has {len(words)} words, outside 140..170')
        sentences = list(re.finditer(r'(?:^|(?<=[.!?]))\s*([^.!?]+[.!?])', stripped))
        final = sentences[-1].group(1).strip() if sentences else ''
        moral_words = re.findall(r"\b[\w'-]+\b", final, flags=re.UNICODE)
        if not 3 <= len(moral_words) <= 32 or not any(t in final.casefold() for t in MORAL_TERMS):
            issues.append('response does not end with a concise moral about sharing credit')
    elif case_id == 'hello':
        if len(stripped) > 512:
            issues.append('short greeting response is unexpectedly long')
    elif case_id == 'topic':
        bullets = [line for line in stripped.splitlines() if re.match(r'^\s*(?:[-*•]|[1-5][.)])\s+', line)]
        if len(bullets) != 5:
            issues.append(f'response has {len(bullets)} bullets, expected five')
        issues += [f'response omits {t}' for t in ('paging', 'page fault', 'tlb') if t not in stripped.casefold()]
    elif case_id in ('structured-json', 'structured-json-schema'):
        encoded = stripped
        if case_id == 'structured-json':
            match = re.fullmatch(r'```(?:json)?\s*\n(?P<json>.*)\n```', stripped, flags=re.DOTALL | re.IGNORECASE)
            encoded = match.group('json').strip() if match else stripped
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            return ['response is not valid JSON']
        if not isinstance(value, dict) or set(value) != {'path', 'operation', 'line_start', 'line_end', 'rationale'}:
            issues.append('JSON object has the wrong key set')
        elif (value.get('path') != 'src/cache.rs' or value.get('operation') != 'replace'
              or value.get('line_start') != 41 or value.get('line_end') != 47
              or not isinstance(value.get('rationale'), str) or not value['rationale'].strip()):
            issues.append('JSON object does not preserve the requested edit')
    elif case_id == 'multilingual':
        bullets = [line for line in stripped.splitlines() if re.match(r'^\s*(?:[-*•]|[1-4][.)、])\s*', line)]
        if len(bullets) != 4:
            issues.append(f'response has {len(bullets)} bullets, expected four')
        if not ('寫入時複製' in stripped or 'copy-on-write' in stripped.casefold()):
            issues.append('response omits copy-on-write')
        if 'fork' not in stripped.casefold() or '頁' not in stripped:
            issues.append('response omits the requested fork/page example')
    return issues


def aggregate(rows):
    """Pooled weighted rate over `rows` (dicts with case, completion_tokens, decode_seconds)."""
    tokens = sum(WEIGHTS[r['case']] * (r['completion_tokens'] - 1) for r in rows)
    seconds = sum(WEIGHTS[r['case']] * r['decode_seconds'] for r in rows)
    return tokens / seconds if seconds > 0 else 0.0


def request(args, body):
    """One completion; returns (content, completion_tokens, decode seconds, server decode ms or None)."""
    stream = args.timing == 'stream'
    body = dict(body, stream=stream, **({'stream_options': {'include_usage': True}} if stream else {}))
    http = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                  headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(http, timeout=args.timeout) as response:
        if not stream:
            result = json.load(response)
            server_ms = (result.get('metrics') or {}).get('decode_ms')
            if server_ms is None:
                raise SystemExit('--timing server needs metrics.decode_ms in the response (glmrt); use --timing stream')
            return (result['choices'][0]['message'].get('content') or '', result['usage']['completion_tokens'],
                    float(server_ms) / 1e3, float(server_ms), result['choices'][0].get('finish_reason'))
        text, first, finish, usage, reason, server_ms = '', None, None, None, None, None
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            elapsed, event = time.perf_counter() - start, json.loads(data)
            if event.get('error'):
                raise RuntimeError(f"inference error: {event['error']}")
            usage = event.get('usage') or usage
            server_ms = (event.get('metrics') or {}).get('decode_ms', server_ms)
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                piece = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                if piece and first is None:
                    first = elapsed
                text += delta.get('content') or ''
                if choice.get('finish_reason'):
                    finish, reason = elapsed, choice['finish_reason']
    if usage is None or first is None or finish is None:
        raise RuntimeError('stream ended without usage, output or a finish reason')
    return text, usage['completion_tokens'], finish - first, server_ms, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', help='served model id (default: the first of /v1/models)')
    parser.add_argument('--output', type=Path, required=True, help='new JSON file for every request and the summary')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--nonce-seed', type=int, default=2026090801, help='glmrt v9 used 2026090801')
    parser.add_argument('--tokenizer', type=Path, help='tokenizer.json for the nonces (default: the model in HF_HOME)')
    parser.add_argument('--tokenizer-model', default='wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1')
    parser.add_argument('--timing', choices=['stream', 'server'], default='stream')
    parser.add_argument('--case', action='append', choices=[c[0] for c in CASES], help='run only these cases')
    parser.add_argument('--warmup', type=int, default=0, help='untimed requests (hello) before the corpus')
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'refusing to overwrite {args.output}')
    if args.model is None:
        with urllib.request.urlopen(args.base_url.rstrip('/') + '/v1/models', timeout=30) as response:
            args.model = json.load(response)['data'][0]['id']
    tokenizer = args.tokenizer
    if tokenizer is None:
        hub = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'
        repo = hub / ('models--' + args.tokenizer_model.replace('/', '--'))
        tokenizer = repo / 'snapshots' / (repo / 'refs' / 'main').read_text().strip() / 'tokenizer.json'
    cases = [c for c in CASES if not args.case or c[0] in args.case]
    nonces = token_zero_nonces(args.repeats * len(cases), args.nonce_seed, tokenizer)
    body = lambda prompt, max_tokens, schema: dict(
        model=args.model, messages=[dict(role='user', content=prompt)], temperature=0, enable_thinking=False,
        max_tokens=max_tokens, **({'response_format': SCHEMA} if schema else {}))
    for i in range(args.warmup):
        request(args, body(f'warm-up {i}: hi', 32, False))
    rows, started = [], time.time()
    for repeat in range(args.repeats):
        for index, (case_id, max_tokens, weight, schema, prompt) in enumerate(cases):
            prefix = nonces[repeat * len(cases) + index] + 'Treat the preceding request nonce as irrelevant.\n'
            text, tokens, seconds, server_ms, reason = request(args, body(prefix + prompt, max_tokens, schema))
            issues = validate(case_id, text)
            rows.append(dict(repeat=repeat + 1, case=case_id, weight=weight, completion_tokens=tokens,
                             decode_seconds=seconds, server_decode_ms=server_ms, finish_reason=reason,
                             decode_tps=(tokens - 1) / seconds if seconds > 0 else 0.0, issues=issues, content=text))
            print(f"r{repeat + 1} {case_id:24s} {tokens:4d} tok {seconds * 1e3:8.1f} ms {rows[-1]['decode_tps']:6.2f} tok/s"
                  f"{'  ISSUES ' + '; '.join(issues) if issues else ''}", flush=True)
        args.output.write_text(json.dumps(dict(partial=True, rows=rows), ensure_ascii=False, indent=1))
    per_case = {c[0]: aggregate([r for r in rows if r['case'] == c[0]]) for c in cases}
    per_repeat = [aggregate([r for r in rows if r['repeat'] == k + 1]) for k in range(args.repeats)]
    summary = dict(benchmark='glmrt-v9-weighted', label=args.label, base_url=args.base_url, model=args.model,
                   timing=args.timing, repeats=args.repeats, nonce_seed=args.nonce_seed, started=started,
                   weighted_tps=aggregate(rows), median_repeat_tps=statistics.median(per_repeat),
                   repeat_tps=per_repeat, case_tps=per_case,
                   passed=sum(not r['issues'] for r in rows), requests=len(rows))
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1))
    print(f"weighted {summary['weighted_tps']:.2f} tok/s (median repeat {summary['median_repeat_tps']:.2f}; "
          f"{summary['passed']}/{summary['requests']} outputs pass) | "
          + ' '.join(f'{k} {v:.2f}' for k, v in per_case.items()))


if __name__ == '__main__':
    main()
