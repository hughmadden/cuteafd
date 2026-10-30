# <Model> <quant>: reference semantics

Smoke-port phase 0 output for `<org/name>` at revision `<40-hex SHA>`.

**Sources.** Geometry and bytes: `<config file>` and the shard headers (`census.md`, from
`scripts/ckpt_census.py`). Semantics: `<reference: modeling file at commit, or image + manifest
digest + engine version>`. Every number below cites one of them; anything not settled by a source
is listed under Open questions, never guessed.

| Topic | Reference file |
| --- | --- |
| config → model args | |
| decoder layer and loader name map | |
| attention | |
| positional encoding | |
| router / experts | |
| drafter | |
| vision / audio | |
| tokenizer, template, parsers | |

## 1. Geometry

| Item | Value |
| --- | --- |
| Layers (dense / MoE split) | |
| Hidden / vocab / tied embeddings | |
| Norm (type, eps) | |
| Attention (type, heads, head dims, latent ranks) | |
| Positions (max, scaling) | |
| Routed experts (count, top-k, intermediate, activation, clamp) | |
| Shared experts | |
| Drafter | |

### Per-layer pattern (from the census shape table, not from the config alone)

| Property | Layers |
| --- | --- |
| attention kind | |
| positional encoding on/off | |
| KV owner (and which layers alias it) | |
| indexer present / reused from | |
| dense or MoE FFN | |

## 2. Decoder layer (pseudo-code with rounding points)

```text
```

## 3. Attention

## 4. Positional encoding (formula, tables, convention: interleaved or NeoX halves)

## 5. MoE and MLPs (scoring, normalization, scale, bias, groups; expert storage per format)

## 6. Other operators (indexers, gates, scaling, soft-caps)

## 7. Drafter (deferred for a smoke unless the shared drafter supports it unchanged)

## 8. Vision / audio (deferred for a smoke)

## 9. Tokenizer, template, API (special-token ids, stop markers, reasoning and tool syntax, settings)

## 10. Checkpoint name map and quantization declarations

## Open questions (each with how it will be settled)
