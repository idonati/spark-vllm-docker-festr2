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

## C28 — drop --enforce-eager (CUDA graphs)

Same recipe as C27 minus the `--enforce-eager` flag. CUDA graph capture sizes [1, 2, 4, 8] (PIECEWISE + FULL). Capture completed in ~3 s, init engine 101 s total (70 s compilation).

KV cache size at boot: **619,881 tokens** (small reduction vs C27's 696k due to ~0.39 GiB graph pool memory).

### Comparison to C27 (eager)

| Test | C27 eager | C28 CUDA graphs | speedup |
|---|---|---|---|
| Math sanity wall | 9.5 s | 4.15 s | 2.3× |
| Code task 1 (LRU) | 9.41 tok/s | **13.26 tok/s** | **+41%** |
| Code task 2 (refactor) | 9.40 tok/s | 12.31 tok/s | +31% |
| Long context (26k) decode | 5.03 tok/s | 5.58 tok/s | +11% |
| Long context (26k) total | 337 tok/s | **374 tok/s** | +11% |
| Solo (sweep) | 9.42 tok/s | 12.20 tok/s | +29% |
| 2 concurrent | 16.88 tok/s | 19.48 tok/s | +15% |
| 4 concurrent | 35.99 tok/s | 36.31 tok/s | +1% |

### Output quality

| Task | Snippet |
|---|---|
| LRU cache | `from collections import OrderedDict\nimport threading\n\ndef lru_cache_simple(maxsize):\n    """Simple LRU cache decorator with a fixed maxsize.\n    Caches up to maxsize results keyed by posi…` |
| UserRepository refactor | `class UserRepository:\n    def __init__(self, db):\n        self.db = db\n    def _find_by(self, field, value):\n        """Generic` |
| 26k-token codebase Q | `We are given a series of modules, each with a compute method. The pattern for module i (from 0 to 199) is:\n compute_i(x, y) = (x * (i+1)) + (y * (i+2)) - (3*i)` |

### Observations

- **CUDA graphs work on sm_121 / pr41797 image.** Graph capture (PIECEWISE + FULL) completed without errors. The earlier `--enforce-eager` was conservative; we can drop it.
- **Speedup is largest at solo and small-batch decode** — kernel launch overhead matters most when each layer's GEMM is small. At 4 concurrent the GEMMs are large enough that overhead is already amortized.
- **Long-context decode gains less** (+11%) because attention dominates at 26k context, and the attention kernel doesn't benefit from CUDA-graph fusion as much as the per-step GEMMs do.
- **Coding output quality is solid.** Real-looking Python with imports, docstrings, thread-safety mentions, correct refactor patterns, correct extraction of formulas from long context.

### Recipe note

The working recipe at C28 (in `recipes/4x-spark-cluster/mimo-v2.5-pro-c28.yaml`) is the recommended baseline for coding workloads: 32k context, 4 concurrent, FP8 KV cache, CUDA graphs.

## C32 — MTP speculative decoding (with fused-qkv-split fix for MTP head)

Same recipe as C31 but with `patch_mimo_v2_mtp_qkv_split.py` applied. The MTP head's `mimo_v2_mtp.py` had the SAME naive `chunk(tp_size, dim=0)[tp_rank]` bug as the main model's `mimo_v2.py` — fixed identically.

| Test | C28 (no spec) | C31 (broken MTP) | **C32 (fixed MTP)** | C32 vs C28 |
|---|---|---|---|---|
| LRU code | 13.26 tok/s | 10.39 | **18.83** | **+42%** |
| Refactor | 12.31 tok/s | 10.53 | **18.08** | **+47%** |
| Long context (26k) | 5.58 tok/s | 4.40 | 5.62 | +1% |
| Solo (sweep) | 12.20 tok/s | 10.96 | **17.88** | **+47%** |
| 2 concurrent | 19.48 tok/s | 16.74 | **28.74** | **+47%** |
| 4 concurrent | 36.31 tok/s | 29.60 | **50.89** | **+40%** |

### Output quality (improved)

Refactor task C28 vs C32:

**C28 (`UserRepository` refactor)**:
```python
class UserRepository:
    def __init__(self, db):
        self.db = db

    def _find_by(self, field, value):
        """Generic
```

**C32 (`UserRepository` refactor)**:
```python
class UserRepository:
    # Whitelist of fields that can be used in _find_by queries
    _VALID_FIELDS = {"id", "email", "name"}
```

C32's output adds a field whitelist, a stronger defensive-coding pattern than C28's.

### Observations

- **Solo decode 18 tok/s is in the "actually usable for interactive coding" range** (Cursor/Claude users feel ~30+ tok/s as fluid; this is in the same ballpark for a self-hosted setup).
- **50 tok/s aggregate at 4 concurrent users** — viable for a small team.
- **Long context (26k) only +1%** because attention bandwidth dominates there; MTP can't speed up the full-attention layers' KV reads. Hierarchical KV summarization is the lever for that regime.
- **Output quality improved slightly** — MTP draft + verify means the model often takes paths that look more deliberate.

### Recipe

`recipes/4x-spark-cluster/mimo-v2.5-pro-c31.yaml` (the same recipe that was used for C31; the difference between C31 and C32 is the `patch_mimo_v2_mtp_qkv_split.py` fix wired into the launcher).
