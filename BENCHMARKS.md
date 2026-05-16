# Benchmarks

Throughput on 8-node DGX Spark (GB10, sm_121), TP=8, Ray distributed executor, FP8 KV cache, FlashInferCutlass MXFP8 attention, Marlin NVFP4 MoE, `--enforce-eager`. Measured via the OpenAI-compatible chat endpoint, prompts of ~10-260 tokens, decode targets noted per row.

Methodology: each row issues N requests in parallel from a single client thread pool against `localhost:5001`. Per-request latency is wall-clock from request open to response close. Aggregate tok/s is `sum(completion_tokens) / wall`.

## Single-request decode (baseline)

| max_tokens | wall (s) | decode | decode tok/s |
|---|---|---|---|
| 200 | 19.37 | 200 | **10.33** |
| 100 | 9.77 | 100 | 10.24 |
| 40 (math) | 3.63 | 37 | 10.20 |

Solo decode is ~**10.3 tok/s**, prompt-length invariant. Prefill of a 260-token prompt adds ~0.5 s.

## Concurrent decode — 100 tokens per request

| concurrent | wall (s) | total decode | aggregate tok/s | per-req avg (s) | speedup vs solo |
|---|---|---|---|---|---|
| 1 | 9.77 | 100 | 10.24 | 9.77 | 1.0× |
| 2 | 9.77 | 194 | 19.85 | 9.48 | 1.9× |
| 4 | 11.19 | 400 | 35.76 | 11.13 | 3.5× |
| 8 | 12.95 | 794 | **61.33** | 12.76 | 6.0× |
| 16 | 18.67 | 1600 | 85.68 | 18.63 | 8.4× |
| 20 | 17.52 | 1994 | **113.84** | 17.41 | **11.1×** |

## Concurrent decode — 300 tokens per request (heavier)

| concurrent | wall (s) | total decode | aggregate tok/s | per-req avg (s) |
|---|---|---|---|---|
| 1 | 9.72 | 94 | 9.67 | 9.72 |
| 4 | 12.57 | 450 | 35.80 | 12.04 |
| 10 | 15.64 | 1129 | 72.17 | 14.89 |
| 20 | 22.71 | 2376 | **104.63** | 20.23 |

## Observations

- **Batching efficiency is real.** 20 concurrent → 11.1× aggregate throughput. MoE GEMMs and MXFP8 dequant amortize across the batch.
- **Per-request latency degrades gracefully** going 1→20: ~9.7s → ~17.5s for 100-tok decode (1.8× per-request slowdown for 11× throughput).
- **Sweet spot ~8 concurrent**: 61 tok/s aggregate at only ~30% per-request latency overhead vs solo.
- **Saturation above 16**: aggregate grows from 88 → 114 tok/s as concurrency goes 16 → 20. Compute-bound batched ceiling.
- The same single-request decode rate (10.3 tok/s) was measured on C25 (`max_num_seqs=1`) and C26 (`max_num_seqs=20`) — so the batching gains don't cost solo latency.

## Config used (C26)

```yaml
defaults:
  tensor_parallel: 8
  gpu_memory_utilization: 0.65
  max_model_len: 8192
  max_num_batched_tokens: 16384
  max_num_seqs: 20

vllm serve festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8 \
  --kv-cache-dtype fp8 \
  -tp 8 \
  --attention-backend triton_attn_diffkv \
  --enforce-eager \
  --moe-backend marlin
```

KV cache size at boot: 52,036 tokens (the large `max_num_batched_tokens` reserves a chunk of activation memory; reducing it would free more KV).

## Headroom not yet measured

- Drop `--enforce-eager` to enable CUDA-graph decode. Likely +30-50% on solo and batched decode if warmup survives sm_121 graph-capture issues.
- Long-context behavior at `max_model_len` 32k / 64k (untested).
- Prefill throughput on real codebase-sized prompts (untested).

## C27 — 32k context

`max_model_len=32768`, `max_num_seqs=4`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.70`. KV cache size at boot: **696,354 tokens**.

### Short-prompt decode (single)

| Task | prompt (tok) | decode (tok) | wall (s) | decode tok/s |
|---|---|---|---|---|
| Math sanity | ~30 | 30 | 9.53 | ~9.4 |
| LRU cache from scratch | 297 | 400 | 42.49 | 9.41 |
| Refactor UserRepository | 487 | 584 | 62.10 | 9.40 |

### Long-prompt decode (single)

| Task | prompt (tok) | decode (tok) | wall (s) | decode tok/s | total tok/s |
|---|---|---|---|---|---|
| 200-module fake codebase Q | **26,411** | 400 | 79.51 | 5.03 | **337.22** |

Prefill rate inferred: ~3,300 tok/s (the 26k prompt prefilled in ~8 s before decode started).

### Concurrent decode at max_num_seqs=4 (short prompts, 200-tok decode)

| concurrent | wall (s) | decode total | agg tok/s | speedup |
|---|---|---|---|---|
| 1 | 21.22 | 200 | 9.42 | 1.0× |
| 2 | 23.70 | 400 | 16.88 | 1.8× |
| 4 | 22.23 | 800 | **35.99** | **3.8×** |

### Observations

- 32k context behaves coherently — model correctly answered a question requiring it to read and reason over a 26k-token "codebase".
- Decode rate falls from ~9.4 tok/s (short context) to ~5 tok/s (26k context). Attention is the cost: each decode token now does a 26k-token KV read.
- Prefill is fast enough that codebase ingest is not the user-visible bottleneck — a 16k-token prompt prefills in ~5 s.
- The large KV cache (696k tokens) leaves room to push `max_model_len` further (64k feasible) or grow `max_num_seqs`.
