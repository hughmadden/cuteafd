# <Model> smoke port: plan

**Inputs.** MODEL `<org/name@SHA>` · CELL `<cards × count, Spark count>` · REFERENCE `<source>` ·
SCOPE text-only, speculator off, one lane.

**Resume here.** `<one paragraph: last green step, branch heads, next action>`

## Fit (from `census.md` and `cuteafd plan`)

| Component | GiB | Lives on |
| --- | ---: | --- |
| routed experts | | Sparks, TP<n> (<GiB>/rank vs <budget>) |
| coordinator weights | | RTX |
| KV per token / pool at the card's budget | | RTX |

Decision: `<layout>`. Refused layouts and why: `<…>`.

## Templates chosen (`references/catalog.md`)

| Subsystem | Source family / code | Tier |
| --- | --- | --- |
| skeleton (config → descriptor → Ops chain → Backend) | | |
| attention | | |
| experts | | |
| API | | |

## Gaps

| # | Gap | Tier | Packet | Shared branch? | Status |
| --- | --- | --- | --- | --- | --- |

## Packets

| Packet | Branch | Owner model | Brief | Gate | Status |
| --- | --- | --- | --- | --- | --- |
| P1 checkpoint + descriptor | | | | `cuteafd plan` names only package/program gaps | |
| P2 experts | | | | unit tests; one MoE layer golden on a Spark | |
| P3 programs | | | | CPU contracts; GPU oracle per program | |
| P4 layer chain + golden | | | | CPU `Ops` oracle; per-layer kinds asserted | |
| P5 serving | | | | `plan --deployment`; `run.sh --dry-run` | |
| P6 reference golden | | | | per-layer dumps for 2-8 windows | |

## Phase 2 log

| When (with zone) | Step | Result | Next |
| --- | --- | --- | --- |
