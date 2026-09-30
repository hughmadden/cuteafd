#!/usr/bin/env python3
"""Score DFlash2 draft-count policies offline from a serve-glm trace.

serve-glm writes CUTEAFD_SPECULATION_TRACE (JSON lines): one "cycle" record per
sequence per verify step (position = tokens already cached, rows verified,
rows committed, planned DFlash2 drafts, the full 7-token draft) and one
"done" record per request with its generated tokens. At temperature 0 the
generated text is the target's greedy continuation, so every cycle's
counterfactual acceptance (how many of its 7 drafts match what the target
went on to emit) is known, and any fixed or oracle draft count can be priced
with a verify-step table:

    tokens/s = sum(1 + min(k, accepted)) / sum(step_ms(1 + k) + draft_ms)

The actual policy is scored the same way (its own k per cycle, same table),
so the comparison isolates the choice of k from step-time noise. Cycles that
used a copy window, and C>1 steps, are priced as single-sequence steps.
"""
import argparse
import json
from collections import defaultdict

# rows -> verify ms: dflash_policy.rs K4_TP4_STEP_MS (1 RTX PRO 6000 at 325 W + 4 Sparks TP4).
K4_TP4 = [(1, 35.0), (2, 50.5), (3, 60.2), (4, 67.2), (5, 77.6), (6, 85.3), (7, 92.8), (8, 102.6)]


def interpolate(table, rows):
    for (r0, m0), (r1, m1) in zip(table, table[1:]):
        if rows <= r1:
            return m0 + (m1 - m0) * (rows - r0) / (r1 - r0)
    return table[-1][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trace')
    parser.add_argument('--table', help='rows:ms,... verify-step table (default: K4_TP4_STEP_MS)')
    parser.add_argument('--draft-ms', type=float, default=4.7, help='drafter step ms added to every cycle')
    args = parser.parse_args()
    table = K4_TP4 if not args.table else [tuple(map(float, p.split(':'))) for p in args.table.split(',')]
    cycles, done = defaultdict(list), {}
    with open(args.trace) as source:
        for line in source:
            record = json.loads(line)
            (cycles[record['id']].append(record) if record['kind'] == 'cycle' else done.__setitem__(record['id'], record))
    rows = []
    for rid, request in done.items():
        generated, prompt = request['generated'], request['prompt_tokens']
        for c in cycles[rid]:
            if not c.get('draft'):
                continue
            # The cycle's first row is generated[i] (emitted, not yet cached); drafts predict what follows it.
            i = c['position'] - prompt
            truth = generated[i + 1:i + 1 + len(c['draft'])]
            accepted = next((j for j, (d, t) in enumerate(zip(c['draft'], truth)) if d != t), len(truth))
            actual_k = c['rows'] - 1
            rows.append(dict(accepted=accepted, actual=actual_k, copy=c['copy'], committed=c['committed']))
    if not rows:
        raise SystemExit('no drafted cycles with a finished request in the trace')
    cost = lambda k: interpolate(table, 1 + k) + args.draft_ms

    def score(choose):
        tokens = sum(1 + min(choose(r), r['accepted']) for r in rows)
        ms = sum(cost(choose(r)) for r in rows)
        return tokens / len(rows), ms / len(rows), 1e3 * tokens / ms

    plain = [r for r in rows if not r['copy']]
    agree = sum(r['committed'] == 1 + min(r['actual'], r['accepted']) for r in plain)
    print(f'consistency: {agree}/{len(plain)} DFlash2 cycles committed 1 + min(verified, counterfactual) rows')
    print(f'{len(rows)} drafted cycles from {len(done)} requests; mean counterfactual acceptance '
          f"{sum(r['accepted'] for r in rows) / len(rows):.2f} of 7; histogram "
          f"{[sum(r['accepted'] == a for r in rows) for a in range(8)]}")
    print(f"{'policy':>10s} {'tok/cycle':>9s} {'ms/cycle':>9s} {'tok/s':>7s}")
    for name, choose in [('actual', lambda r: r['actual'])] + [(f'fixed {k}', lambda r, k=k: k) for k in range(8)] \
            + [('oracle', lambda r: r['accepted'])]:
        t, m, rate = score(choose)
        print(f'{name:>10s} {t:9.2f} {m:9.1f} {rate:7.2f}')


if __name__ == '__main__':
    main()
