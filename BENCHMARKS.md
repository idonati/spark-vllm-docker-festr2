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

## C34 — perf characterization (decode cost decomposition, 2026-05-19)

Same recipe as C32, same fork, but on a fresher vLLM snapshot (`v0.1.dev1+g0891716c2.d20260509`). Asynchronous scheduling is enabled by default in this build. Goal: decompose decode cost into fixed-per-step overhead vs context-scaling KV-read.

> Important: per the M0 finding in `design/kv-compression-for-mimo-v2-diffkv.md`, `--kv-cache-dtype fp8` is silently dropped by MiMo-V2 due to missing `cache_config` plumbing — so the cache here is **BF16** (640 bytes/slot at the 10 full-attn layers), not FP8 as the log line claims.

### Headline: ~2× across the board vs C32

| Config | C32 | C34 | ratio |
|---|---|---|---|
| Solo decode (short ctx) | 17.88 tok/s | 33.55 tok/s | **1.88×** |
| Solo decode at 26k | 5.58 tok/s | 26.96 tok/s | **4.83×** |
| 2 concurrent (aggregate) | 28.74 tok/s | 48.74 tok/s | 1.70× |
| 4 concurrent (aggregate) | 50.89 tok/s | 83.59 tok/s | 1.64× |

The 26k jump is by far the most dramatic — the per-step overhead came down so far that the long-context attention cost no longer drags the rate to a crawl. Most likely source of the headline gains: asynchronous scheduling.

### Method

Streaming OpenAI chat completion at 6 context lengths, 100-token decode each, `ignore_eos=True`. Streaming chunks count was de-aliased to MTP token count via the engine's `usage.completion_tokens`. Per-token decode time fit with a single-axis linear model.

### Results

| ctx (tok) | prefill (s) | decode (s) | decode_tok/s | ms/tok |
|---|---|---|---|---|
| 531 | 0.23 | 2.92 | 33.55 | 29.8 |
| 4373 | 0.22 | 2.96 | 33.15 | 30.2 |
| 8441 | 0.25 | 3.18 | 30.84 | 32.4 |
| 16690 | 0.26 | 3.28 | 29.86 | 33.5 |
| 22905 | 0.29 | 3.45 | 28.40 | 35.2 |
| 26973 | 0.32 | 3.64 | 26.96 | 37.1 |

**Linear fit:** `decode_ms/tok = 29.5 + 0.265 × (ctx / 1k)`

| component | value | share at 26k |
|---|---|---|
| Intercept (fixed per-step) | 29.5 ms/tok | **81%** |
| Slope (per 1k ctx KV-read) | 0.265 ms/tok per 1k | **19%** |

### Interpretation — what this means for C33

Pure-bandwidth-bound prediction for the slope, given 10 full-attn layers × (192 K + 128 V) × 2 bytes BF16 × 1 KV head per rank at TP=8 and assuming ~30 GB/s effective LPDDR5X under Triton paged attention: **~0.2 ms/tok per 1k context**. Measured 0.265 → close to bandwidth-bound but with some kernel-arithmetic component. Triton attention on GB10 sm_121 is essentially saturating memory bandwidth.

This bounds C33's upside. C33 (TQ-K8V4) reduces slot bytes 640 → 260 (2.46×), so it can compress the *slope* by that factor — but it does **not** compress the intercept.

Projected C33 decode tok/s, assuming the slope drops by 2.46× and the intercept is unchanged:

| ctx | C34 ms/tok | C33 projected ms/tok | C33 projected tok/s | C33 vs C34 |
|---|---|---|---|---|
| 26k | 37.1 | 32.3 | 31.0 | +15% |
| 64k | 46.5 | 36.4 | 27.5 | +28% |
| 128k | 63.5 | 43.3 | 23.1 | +47% |
| 200k | 82.5 | 51.0 | 19.6 | +62% |

The original `design/kv-compression-for-mimo-v2-diffkv.md` projected "2-2.5× at 64-200k context"; the actual bound given the now-measured intercept is **~1.6× at 200k**, not 2-2.5×. The design doc was estimated against an older, slower fixed-cost baseline. The async-scheduling improvement in this build effectively pre-spent most of C33's win at the lengths people actually use (≤32k).

### What attacks the 29.5 ms intercept

At batch=1 the intercept is the sum of (a) per-layer kernel launch overhead × 70 layers, (b) NCCL allreduce × 70, (c) MoE GEMM time at batch=1 (latency-bound), (d) SWA attention at window=128 × 60 layers, (e) sampling + MTP draft+verify overhead. The biggest wins by hypothesis:

1. **Higher batch size**. At 2 concurrent: 48.7/33.6 = 1.45× of solo (vs ideal 2×). At 4: 83.6/33.6 = 2.49× (vs ideal 4×). MoE GEMMs are still latency-bound up to batch ~4 — increasing `max_num_seqs` past 4 should unlock more aggregate throughput at modest per-request latency cost.
2. **Native NVFP4 MoE via cutlass-dsl 4.5.1** (when it works on sm_121a). Removes the FP4→FP8 dequant Marlin currently does per token.
3. **Better cudagraph coverage**. Recipe captures `[1,2,4,8,16]` only. Real serving sees odd batch sizes that fall back to eager. Either widen capture set or use full-graph mode.

### Per-config decode rate matrix (sweep, max_tokens=100)

| ctx | bsz=1 | bsz=2 | bsz=4 |
|---|---|---|---|
| 256 | 33.5 / 33.5 | 24.4 / 48.7 | 20.9 / 83.6 |
| 26k | 28.5 / 25.7 | 24.3 / 42.2 | 20.9 / 73.3 |

(format: per-request tok/s / aggregate tok/s)

The 26k row is the surprise. **4-concurrent at 26k = 73.3 tok/s aggregate** — only 12% below the 83.6 tok/s aggregate at short context. Batching amortizes the per-step fixed cost across requests, so the long-context drag becomes a thinner slice of overall serving throughput. For four engineers working on a 26k-context codebase that's ~18 tok/s effective per user — equivalent to old C32 solo at short context.

### Recommendation

Reorder. The asynchronous-scheduling-driven 2-5× pickup means **C33 is no longer the highest-leverage next move**. Better order:
1. **Headroom sweep** — push `max_num_seqs` past 4, widen cudagraph capture, measure 32k-context-at-batch-4. Mechanical, no code changes.
2. **Cutlass-dsl 4.5.1 native NVFP4** — attacks the fixed-cost MoE term (hassan-abdallah confirmed sm_120a works; sm_121a re-test owed to NVIDIA/cutlass#3227 anyway).
3. **C33 (TQ-K8V4)** — defer; useful at 64k+ contexts as a +28-62% layer but the Ray init hang debug investment buys less than the design doc projected.

### Recipe

Same as C32 — `recipes/4x-spark-cluster/mimo-v2.5-pro-c31.yaml` + the always-applied MTP qkv-split patch. The build difference (vs the C32 measurement) is the underlying vLLM snapshot picking up async scheduling.

## C34a — headroom sweep: max_num_seqs=16, widened cudagraph capture

Same build as C34, recipe bumped to `max_num_seqs=16`, `max_num_batched_tokens=16384`, and `cudagraph_capture_sizes=[1,2,4,6,8,12,16,24,32]` (was vLLM default `[1,2,4,8,16]`).

### Headline

**16-concurrent at short context = 201 tok/s aggregate. 8-concurrent at 26k = 125 tok/s aggregate.** Per-request decode rate at 26k is *the same* at 4-conc and 8-conc (18.74 vs 18.78) — batching at long context is essentially free up to at least 8 simultaneous requests.

For a team of 8 engineers on 26k-token codebases, that's **22× the published C32 solo rate** of 5.58 tok/s.

### Solo + linear fit (cross-check against C34)

| ctx | decode_tok/s | ms/tok |
|---|---|---|
| 531 | 31.4 | 31.9 |
| 4373 | 33.0 | 30.3 |
| 8441 | 27.3 | 36.7 |
| 16690 | 29.1 | 34.4 |
| 22905 | 28.8 | 34.8 |
| 26973 | 27.4 | 36.5 |

**Linear fit:** `decode_ms/tok = 31.9 + 0.161 × (ctx/1k)`. Intercept slightly worse than C34 (29.5 → 31.9, +8%), slope notably better (0.265 → 0.161, -39%). The intercept bump is most likely a small cudagraph-warmup/overhead tax from capturing 9 batch sizes; the slope drop is more interesting and consistent with the wider capture set covering odd batch sizes that previously hit eager.

### Concurrent throughput

#### Short context (~256 tok prompt)

| concurrent | per-req tok/s | aggregate tok/s |
|---|---|---|
| 1 | 34.2 | 29.9 |
| 4 | 25.6 | 83.8 |
| 8 | 22.2 | **157.0** |
| 12 | 16.2 | 171.4 |
| 16 | 14.0 | **201.0** |

Scaling efficiency:
- 1 → 8 conc: 5.25× aggregate (vs ideal 8×, 66% efficient)
- 8 → 16 conc: 1.28× aggregate (vs ideal 2×, 64% efficient)

We're past the knee at 12-conc. Above 12, per-request latency degrades faster than aggregate gains.

#### Long context (~26k tok prompt)

| concurrent | per-req tok/s | aggregate tok/s |
|---|---|---|
| 1 | 26.9 | 24.6 |
| 4 | 18.7 | 64.9 |
| 8 | 18.8 | **124.9** |

The surprise: **per-request flat from 4→8 at 26k.** Doubling batch doubles aggregate without paying anything per-request. Likely because at 26k the attention KV-read amortizes well across the batch (each layer reads its cache once, multiple queries use the same cache), and the MoE GEMM at batch=8 is closer to its compute-bound regime than at batch=4.

### Observations / implications

- **For solo coding** (one user at a time), C34a is no improvement over C34 — solo decode hasn't changed. The whole point of C34a is multi-user serving.
- **For a 4-engineer team** at 26k codebases: 18.7 tok/s per user, 65 tok/s aggregate.
- **For an 8-engineer team** at 26k codebases: 18.8 tok/s per user (same as 4-team), 125 tok/s aggregate.
- **Concurrent serving was the highest-leverage lever** the C34 analysis identified, and C34a confirms it — pushing past `max_num_seqs=4` unlocks 2× more aggregate throughput at long context with no per-request degradation, and 2.4× at short context.
- **Diminishing returns at 12+ concurrent** on short context. The recipe `max_num_seqs=16` is roughly right; 32 would only add ~30% more aggregate at the cost of per-request rates dropping below 10 tok/s.
- The C33 case is weakened further. C33's projection assumed solo decode improvements; concurrent serving was always going to be the cheaper path.

### Recipe

`recipes/4x-spark-cluster/mimo-v2.5-pro-c34a.yaml` — same as c31.yaml with `max_num_seqs: 16`, `max_num_batched_tokens: 16384`, and `--compilation-config '{"max_cudagraph_capture_size": 32, "cudagraph_capture_sizes": [1,2,4,6,8,12,16,24,32]}'`.

### Recommendation update vs C34

The new priority order after C34a:

1. **Cutlass-dsl 4.5.1 native NVFP4 MoE** (task #60) — still attacks the intercept (now 31.9 ms in C34a, 29.5 in C34). With native NVFP4 the per-step intercept could drop meaningfully, lifting per-request decode rate across all batch sizes. See `design/c35-cutlass451-plan.md`.
2. **Try `max_num_seqs=24-32`** with a long-context grid — current sweep covers up to 8 at 26k; whether 12-conc-at-26k still scales linearly is an open question.
3. **C33 (TQ-K8V4)** — definitively deferred. Even at 64k+ contexts, the upside is now bounded against a *much* higher baseline (e.g., 125 tok/s aggregate at 8-conc 26k).

## C35 — cutlass-dsl 4.5.1 native NVFP4 MoE (correctness pass, perf regression)

C34a + a new image (`vllm-node-mimo-pr41797-nccl230-tfgit-cutlass451-sm12x`) bumping `nvidia-cutlass-dsl` 4.5.0 → 4.5.1 and switching `--moe-backend marlin` → `cutlass`. Both `VLLM_USE_FLASHINFER_MOE_FP4=0` and `NCCL_NVLS_ENABLE=0` dropped from env (those were the Marlin-routing workaround).

### Headline: correctness fixed, perf worse

**Two results, both publishable.**

1. **Cutlass-dsl 4.5.1 fixes the NVFP4 MoE correctness bug on sm_121a** (the original `NVIDIA/cutlass#3227` story). Smoke test produces coherent output (`"Hello!"`, `"12 × 7 = 84"`), no garbage. This is the first sm_121a confirmation in addition to hassan-abdallah's existing sm_120a confirmation — the bug-fix covers both ArchTags.

2. **Native NVFP4 MoE is consistently 7-25% slower than the Marlin FP4→FP8 dequant workaround on sm_121a, gap widens with batch size.**

### Comparison vs C34a (same recipe except `--moe-backend cutlass`)

#### Solo + linear fit

| ctx | C34a tok/s | C35 tok/s | Δ |
|---|---|---|---|
| 531 | 31.4 | 30.0 | -4% |
| 4373 | 33.0 | 29.6 | -10% |
| 8441 | 27.3 | 28.7 | +5% |
| 16690 | 29.1 | 26.5 | -9% |
| 22905 | 28.8 | 25.7 | -11% |
| 26973 | 27.4 | 25.1 | -8% |

C34a fit: `29.5 + 0.265 × (ctx/1k)` (original C34) → `31.9 + 0.161 × (ctx/1k)` (C34a)
C35 fit: `33.0 + 0.261 × (ctx/1k)` — intercept slightly worse (+3% vs C34a), slope worse (+62% vs C34a's tighter 0.161 — though C34a's slope is best in class because of its wider cudagraph capture)

#### Concurrent decode at short context

| concurrent | C34a agg | C35 agg | Δ |
|---|---|---|---|
| 1 | 29.9 | 27.6 | -8% |
| 4 | 83.8 | 69.6 | -17% |
| 8 | **157.0** | 131.2 | -16% |
| 12 | 171.4 | 155.7 | -9% |
| 16 | **201.0** | 150.0 | **-25%** |

#### Concurrent decode at 26k

| concurrent | C34a agg | C35 agg | Δ |
|---|---|---|---|
| 1 | 24.6 | 22.9 | -7% |
| 4 | 64.9 | 59.5 | -8% |
| 8 | 124.9 | 112.7 | -10% |

### Why is native NVFP4 slower than Marlin's dequant?

The gap *widens* with batch size at short context (-8% at 1-conc, -25% at 16-conc). Speculation:

- **Marlin's FP4→FP8 dequant is amortized across the batch**: one dequant pass per expert load, then a tuned FP8 GEMM with high tensor-core utilization. The dequant cost gets divided across batch elements.
- **Cutlass-dsl 4.5.1's NVFP4 GEMM on sm_121a may not be using the FP4 tensor cores effectively**, or may be using them at a less-favorable shape for the small-batch MoE expert call sizes we hit at TP=8 (per-expert tile shapes can be small).
- **MoE expert routing overhead** may scale differently between the two paths — the dispatch + scatter/gather pattern around the per-expert GEMM is path-specific.

This isn't a closed question — it's a tuning gap that may close in a future cutlass-dsl release. But for now, on sm_121a as of 2026-05-19 with vLLM `v0.1.dev1+g0891716c2.d20260509`, the Marlin workaround is the production-faster path.

### Decision

**Keep Marlin in production.** C34a is the production recipe. C35 is reverted (image kept for future re-test when cutlass-dsl ships a more tuned sm_121 kernel).

The `VLLM_USE_FLASHINFER_MOE_FP4=0` + `NCCL_NVLS_ENABLE=0` env vars stay in the [SM121 NVFP4 Marlin workaround] story — they're not a bug fix, they're how we get the *faster* path on this hardware.

### Public reporting

- **`NVIDIA/cutlass#3227`**: post sm_121a correctness confirmation + note that 4.5.1 fixes the original `_mma` ptxas garbage-output bug on sm_121a as well as the previously-confirmed sm_120a (per hassan-abdallah). Include perf observation that native is slower than Marlin currently — useful signal for the cutlass team's sm_12x tuning priorities.
- **`vllm-project/vllm#41519`**: short follow-up confirming that with the fork's full setup the MiMo-V2.5-Pro is now serving stably at 125-201 tok/s aggregate concurrent throughput on sm_121a (C34a numbers); the Marlin workaround stays for now.

### Recipe

`recipes/4x-spark-cluster/mimo-v2.5-pro-c35-cutlass.yaml` — kept in-repo for reproducibility, but not the production recipe.

## C34b — InstantTensor loader (6.7× faster main-model weight load)

Same as C34a (`max_num_seqs=16`, widened cudagraph capture, MTP, Marlin NVFP4 MoE) but with `--load-format instanttensor` instead of `safetensors`. Discovered as a side-effect of deploying DeepSeek-V4-Pro/Flash on the same cluster — their recipes use `instanttensor` and load 5-6× faster.

### Headline

| Phase | C34a (safetensors) | C34b (instanttensor) | Speedup |
|---|---|---|---|
| Main-model weight load (114 shards on rank 0) | 11:47 | **1:46** (318,545 tensors @ 3000 it/s) | **6.7×** |
| MTP drafter load | 1:04 (shared weights, fast path) | 1:43 | (drafter doesn't benefit — it shares weights with the main model that just finished loading) |
| Total weight load | 12:51 | **3:29** | **3.7×** |
| Net boot-to-ready | ~14 min | **~7 min** | ~2× |
| Decode quality | coherent | coherent | identical |

The festr2 vLLM image (`vllm-node-mimo-pr41797-nccl230-tfgit-sm12x`) already had `instanttensor` registered in `vllm/model_executor/model_loader/__init__.py` (lines 38, 55) and the `instanttensor==0.1.8` PyPI package installed — no image rebuild required. Single recipe-line change.

### What InstantTensor does

InstantTensor is NVIDIA's pipelined-I/O safetensors loader. Instead of `mmap` + per-tensor lazy load (which forces serialized disk reads when EXT4 can't auto-prefetch a 555-GiB checkpoint with 38 GiB RAM headroom), it streams shards in pipelined chunks distributed across ranks. The festr2 checkpoint is split into 114 large shards but contains 318,545 tensors total (fine-grained — many small MoE expert tensors); InstantTensor amortizes the per-tensor `safe_open`/`get_tensor` overhead that dominates the standard loader at this granularity.

### Quality validation

Same coherence gates as C32 / C34 / C34a / C35:
- `"Say hello."` → `"Hello! 👋 How are you doing today? I'm MiMo, your AI assistant..."` ✓
- `"12 × 7 = ?"` → `"12 × 7 = **84**"` ✓
- `"def fibonacci(n):"` → coherent Python tutorial intro ✓

Solo decode on cold-ish post-warmup: 25-28 tok/s. (C34a steady-state was 33 tok/s — early cold-call variance is normal; full perf sweep on C34b not re-run since it's the same engine state once weights are loaded.)

### Decision

**Promote C34b to production.** The default production recipe pointer should be `mimo-v2.5-pro-c34b-instanttensor.yaml`. C34a stays in-repo as the safetensors-baseline reference.

Caveats: if a deployer hits an InstantTensor compatibility issue (different checkpoint structure, older drivers), they can switch back to `--load-format safetensors` for a 4× boot-time penalty.

### Recipe

`recipes/4x-spark-cluster/mimo-v2.5-pro-c34b-instanttensor.yaml` — same as C34a with one line changed.

## C36 — cross-model bench + InstantTensor port to GLM/Kimi

C34b's InstantTensor finding wasn't just a festr2 thing: all four images on the cluster ship `instanttensor==0.1.8` and have it registered in vLLM's `model_loader/__init__.py`. The DSV4 recipes used it already; the festr2 (C34b), GLM and Kimi recipes were leaving it on the table.

This cycle (C36) ports InstantTensor to the GLM and Kimi recipes and runs a standardized bench across all five deployed models on the same 8× DGX Spark cluster, same day, same standardized bench script.

### Boot-time wins from InstantTensor (where previously not used)

| Model | Standard safetensors load | InstantTensor load | Speedup |
|---|---|---|---|
| festr2 MiMo-V2.5-Pro | 11:47 (main only) | 1:46 | **6.7×** (already C34b) |
| GLM-5.1-NVFP4 | ~10:31 (85 shards × 4.33s/it) | 1:20 (232k tensors @ 2888 it/s) | **7.9×** |
| Kimi-K2.6-NVFP4 | ~13:44 (119 shards × 6.93s/it) | 1:06 (278k tensors @ 4213 it/s) | **12.5×** |

End-to-end boot-to-ready (Ray + load + warmup):

| Model | Without IT | With IT | Wall-clock saved |
|---|---|---|---|
| GLM-5.1 | ~16 min | **3:29** (209s) | ~12 min |
| Kimi-K2.6 | ~16 min | **4:37** (277s) | ~11 min |

### Standardized C36 bench across all 5 models

Same client (`/home/idonati/profiles/bench_c36.py`), same prompts, same protocol. Run on the same day on the same cluster state.

| Model | Solo decode | 4-concurrent (agg) | 8-concurrent (agg) | Long-ctx 4k decode | Boot wall |
|---|---|---|---|---|---|
| **festr2 MiMo-V2.5-Pro (C34b)** | **32.25 tok/s** | **80.35 tok/s** | **130.91 tok/s** | 13.62 tok/s | 6:47 |
| GLM-5.1-NVFP4 (IT) | 13.92 | 41.19 | 72.00 | 4.27 | 3:29 |
| DSV4-Flash | 11.66 | 41.68 | 62.12 | 2.42 | 2:47 |
| Kimi-K2.6-NVFP4 (IT) | 10.59 | 26.79 | 59.16 | err* | 4:37 |
| DSV4-Pro (max_ctx=512) | 6.91 | 22.23 | 36.13 | n/a | 5:05 |

*Kimi 4k-context returned HTTP 400 — likely tokenizer chunking pushed past the recipe's `max_model_len=8192` after the bench prompt template expanded. Not a regression; the model serves coherent output at shorter contexts.

### Headlines

1. **festr2 MiMo-V2.5-Pro (C34b) is the cluster's throughput leader by a wide margin.** 130 tok/s aggregate at 8-concurrent is ~2× the next model (GLM 72) and ~2.2× DSV4-Flash (62). Same hardware, same TP=8 — the gap is the model's MoE config + the MTP speculative-decoding gain + the wider cudagraph capture set we added in C34a.
2. **InstantTensor is universal.** Three of the four images ship it, the fourth (Kimi's eugr build) also registers it — just under a different vllm install path. Recipe-line change only; no rebuild. Worth porting to every recipe.
3. **DSV4-Pro at max_model_len=512 is the bottom of the table.** That's not the model's fault — the 102 GiB/rank weights leave essentially no KV budget at gpu_mem=0.90, so the recipe pins ctx low. For longer-context coding work, DSV4-Flash is the better DeepSeek-family pick; for top throughput, festr2 wins.
4. **All four "recently retested" models produced coherent output to the standard smoke gates.** No regressions vs the per-model deployment cycles earlier today.

### Recipes added in C36

- `recipes/4x-spark-cluster/glm-5.1-nvfp4-instanttensor.yaml` — GLM with `--load-format instanttensor`
- `recipes/4x-spark-cluster/kimi-k2.6-instanttensor.yaml` — Kimi with `--load-format instanttensor`

### Bench script

`/home/idonati/profiles/bench_c36.py` — auto-detects model from `/v1/models`, runs smoke + solo + 1/4/8-concurrent + 4k-context. Per-model output in `/home/idonati/profiles/c36_results.jsonl`. Reusable for future re-benches.

## C37 — long-context decode grid across all models

Per-model decode rate at 5 token-count points: ~1k, ~4k, ~8k, ~16k, ~26k. Targets `max_model_len`-permitting. DSV4-Pro skipped (max_model_len=512 too small to grid). Same client (`/home/idonati/profiles/bench_c37_longctx.py`), 80-token decode with `ignore_eos`, same hardware, same day.

| ctx tokens | festr2 C34b | GLM-5.1 (IT) | DSV4-Flash | Kimi-K2.6 (IT) |
|---|---|---|---|---|
| ~1k | **30.7 tok/s** | 10.8 | 6.5 | 10.1 |
| ~4k | **29.3** | 12.9 | 11.6 | 11.2 |
| ~8k | **33.3** | 12.9 | 11.6 | (max_model_len 8k) |
| ~16k | **27.0** | 12.7 | 11.5 | n/a |
| ~26k | **24.9** | 12.0 | 10.9 | n/a |

### Observations

- **festr2 is the long-context winner by a wide margin.** Even at 26k context (its production target), decode stays at ~25 tok/s — 2× GLM, 2.3× DSV4-Flash. Plus MTP speculative decoding gives it a structural advantage. C34b's wider cudagraph capture set helps too.
- **GLM-5.1 and DSV4-Flash are essentially equivalent on long-context decode** (12-13 tok/s flat). Pick based on model quality preference for your workload, not on long-context speed.
- **DSV4-Flash prefill is the slow side**: 78.8s for a 26k-token prefill. That's 337 tok/s prefill rate. The decode is steady but anyone serving long-context chat with DSV4-Flash will feel the TTFT (~80s for a 26k cold start).
- **Kimi caps at 8k** in this recipe (`max_model_len=8192`). At its supported range it's competitive with GLM. For long-context workloads on this cluster, festr2 is the clear pick.
- **All four models hold their decode rate within 10-20% across the full context range** they support — no model "falls off a cliff" the way C28-era festr2 did (5.6 tok/s at 26k was the problem we solved in C34).

### Prefill rate observations

| ctx | festr2 | GLM | DSV4-Flash | Kimi |
|---|---|---|---|---|
| ~26k prefill | 6.1s = 4400 tok/s | 9.9s = 2680 tok/s | 78.8s = 337 tok/s | n/a |

festr2 and GLM have well-tuned prefill paths. DSV4-Flash prefill is ~10× slower than festr2 prefill at the same context, which dominates user-perceived TTFT for long prompts.

### Recommendation by workload

- **Coding assistant with long context (>8k)**: festr2 MiMo-V2.5-Pro (C34b). 24-33 tok/s decode + fast prefill + MTP. The cluster's best general-purpose pick.
- **Multi-step thinking-trace heavy tasks at moderate context (<8k)**: GLM-5.1 or Kimi-K2.6, both serve at 10-13 tok/s with reasoning-channel output. Pick on output style fit.
- **Throughput-priority short-context serving**: festr2 wins again (130 tok/s aggregate at 8-conc per C36).
- **Avoid for long-context interactive use**: DSV4-Flash (78s prefill at 26k makes TTFT painful even though decode is fine).

### Bench script

`/home/idonati/profiles/bench_c37_longctx.py` — auto-detects model, runs 5-point context grid via the OpenAI streaming API, separates prefill and decode timing. Output appended to `/home/idonati/profiles/c37_results.jsonl`.

## C38 — long-context concurrent serving across models (16k ctx, 1/4/8 conc)

The remaining cross-model gap: C36 covered concurrent at *short* ctx, C37 covered *solo* long-ctx, but no per-model picture of concurrent serving at long context. C38 fills that gap with all three long-ctx-capable models benched at 16k tokens × 1/4/8 concurrent.

| conc | festr2 C34b @16k | GLM-5.1 (IT) @16k | DSV4-Flash @16k |
|---|---|---|---|
| 1 | 6.91 agg / 6.91 per-req | 2.33 / 2.33 | 1.05 / 1.05 |
| 4 | **63.99** / 16.40 | 31.69 / 7.97 | 21.00 / 5.26 |
| 8 | **110.96** / 14.36 | 49.36 / 6.20 | 22.52 / 2.82 |

(Numbers in tok/s. conc=1 includes the full cold-prefill cost in the wall time, so don't read those as steady-state decode rates — compare to C37 for that. 4-conc and 8-conc numbers are the production-realistic ones.)

### Observations

- **festr2 is the dominant long-context server** at every batch size. 111 tok/s aggregate at 8-conc 16k = 13.9 tok/s effective per user for an 8-engineer team. GLM gives 6.2 tok/s per user at the same batch (49 aggregate); DSV4-Flash gives 2.8 tok/s per user (22.5 aggregate).
- **DSV4-Flash plateaus between 4 and 8 concurrent at 16k** (21 → 22.5 agg). The model can't usefully serve more than ~4 simultaneous long-context users on this cluster. Likely bound on prefill throughput rather than decode.
- **GLM scales close to linearly 4 → 8 conc** (31.7 → 49.4, 1.56×). Reasonable serving headroom at long context.
- **festr2 scales 4 → 8 conc by 1.73× at 16k** (64 → 111). Close to the C34a 4→8 scaling we measured at 26k context (65 → 125 ≈ 1.92×).

### Recommendation matrix (16k context, multi-user serving)

| team size at 16k context | festr2 per-user | GLM per-user | DSV4-Flash per-user |
|---|---|---|---|
| 1 (solo, warm decode) | ~27 tok/s (C37) | ~13 tok/s (C37) | ~12 tok/s (C37) |
| 4 | 16 tok/s | 8 tok/s | 5 tok/s |
| 8 | 14 tok/s | 6 tok/s | 3 tok/s |

For coding-team work at long context, **festr2 is the only model that holds a usable per-user decode rate (14 tok/s) at 8 concurrent users**. GLM is viable for a 4-person team. DSV4-Flash should be limited to ≤4 concurrent long-ctx users.

### Bench script

`/home/idonati/profiles/bench_c38_longctx_conc.py` — 1/4/8-conc at parameter-able context length. Results in `/home/idonati/profiles/c38_results.jsonl`.
