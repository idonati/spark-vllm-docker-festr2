# KV-cache compression for MiMo-V2 (diffkv) — design doc

## Status

Draft / not yet implemented. This doc captures the problem, what vLLM already ships, and the realistic implementation paths.

## Problem

At long context, decode rate on `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8` on the 8× DGX Spark cluster degrades because the 10 full-attention layers (out of 70) read the full KV cache every decode step. Measured at C30 (`max_model_len=262144`, `--kv-cache-dtype fp8`-on-the-command-line-but-silently-BF16-in-practice, CUDA graphs, no MTP):

| context | sustained decode tok/s |
|---|---|
| 16k | 6.9 |
| 64k | 2.1 |
| 128k | 1.0 |
| 200k | 0.6 |

The 60 sliding-window (window=128) layers are constant-cost per decode step (window doesn't grow with context); the bottleneck is the 10 full-attn layers, whose KV reads scale linearly with context.

The MoE expert weights are NVFP4 and the qkv weights are MXFP8 — those are *weight* quantizations, separate from KV-cache compression.

**Important M0 finding (see below)**: the `--kv-cache-dtype fp8` flag is silently ignored for all 70 MiMo-V2 attention layers because `mimo_v2.py:368,388` instantiate `MiMoV2Attention` without a `cache_config`. So the production KV cache is actually BF16, not FP8 — and the compression opportunity is ~2× larger than this doc originally estimated.

## What vLLM ships today (image `pr41797-nccl230-tfgit-sm12x`, May 2026-05-09 snapshot)

KV cache dtypes supported by the *engine*:

```
auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc,
int8_per_token_head, fp8_per_token_head, nvfp4
```

But the attention backend MiMo-V2 requires (`triton_attn_diffkv`, for unequal Q/V head_dim 192/128) only supports:

```
auto, bfloat16
```

The `__init__` check at `triton_attn_diffkv.py:147` actively raises `NotImplementedError` for any quantized kv_cache_dtype. In production this never fires — see M0 finding below; the path is that the dtype is never actually passed.

TurboQuant ships with its own attention backend (`TurboQuantAttentionBackend`) and four KV variants (k8v4, k3v4_nc, 4bit_nc, 3bit_nc) implementing 2-4× compression vs BF16, BUT that backend assumes a single per-layer `head_size` and does not understand diffkv. PR vllm#40108 (open) adds sliding-window support to TurboQuant; nothing in flight adds diffkv support.

`sparse_attn_indexer` exists for DeepSeek-V3.2 only. No SnapKV / H2O / StreamingLLM / Quest implementations in this build.

### Per-variant compression at MiMo-V2 diffkv head dims (head_size_q=192, head_size_v=128, num_kv_heads=8)

Per-position-per-head slot bytes for the 10 full-attn diffkv layers (the bottleneck):

| variant | K bytes | V bytes | slot bytes | ratio vs BF16 (640) |
|---|---|---|---|---|
| BF16 (current production) | 384 | 256 | 640 | 1.00× |
| FP8 (claimed by CLI, never actually applied) | 192 | 128 | 320 | 2.00× |
| turboquant_k8v4 | 192 | 68 | 260 | 2.46× |
| turboquant_4bit_nc | 98 | 68 | 166 | 3.86× |
| turboquant_k3v4_nc | 74 | 68 | 142 | 4.51× |
| turboquant_3bit_nc | 74 | 52 | 126 | 5.08× |

K bytes by config: `key_fp8 ⇒ head_size_q` else `ceil(head_size_q * key_quant_bits / 8) + 2` (norm fp16).
V bytes: `ceil(head_size_v * value_quant_bits / 8) + 4` (scale + zero fp16).

Implication: K8V4 yields ~1.23× over a (hypothetical) FP8 baseline but ~2.46× over the *actual* production BF16. K3V4_nc moves to ~4.5× over production BF16 — that's the variant worth chasing if quality holds.

## Why TurboQuant > SnapKV for our case

- **TurboQuant** is uniform, lossless-ish compression of *every* KV slot. Quality cost is from the quantization noise (small, well-characterized). Pairs naturally with vLLM's PagedAttention block structure.
- **SnapKV** evicts low-importance tokens. Quality cost is from the eviction policy (potentially large if the eviction policy misranks an important past token). Eviction also conflicts with vLLM's block-based KV cache management.

Given the engineering effort is comparable and TurboQuant is partially merged upstream, that is the path to pursue.

## Milestones

### M0 — Understand the current FP8/diffkv silent path — SOLVED 2026-05-16

**The `--kv-cache-dtype fp8` flag is silently dropped on the floor for MiMo-V2.** Mechanism (verified by reading the deployed image's source):

1. `mimo_v2.py:368` and `mimo_v2.py:388` (the SWA and full-attn branches of `MiMoV2FlashDecoderLayer.__init__`) instantiate `MiMoV2Attention(...)` *without* passing `cache_config`.
2. That gets passed through `MiMoV2Attention.__init__(cache_config=None, ...)` and `MiMoV2Attention.__init__` propagates the default `cache_config=None` into `Attention(...)` at `mimo_v2.py:322`.
3. `Attention.__init__` (attention.py:224-228) reads:
   ```python
   if cache_config is not None:
       kv_cache_dtype = cache_config.cache_dtype
   else:
       kv_cache_dtype = "auto"  # <-- this branch hits
   ```
4. So every `TritonAttentionDiffKVImpl` gets constructed with `kv_cache_dtype="auto"`. `is_quantized_kv_cache("auto") == False`, the `NotImplementedError` does not fire, the engine starts, and the KV cache for the 70 diffkv layers is in fact BF16 (the model's compute dtype).

Verified by:
- Direct instantiation in the live container — `TritonAttentionDiffKVImpl(..., kv_cache_dtype="fp8")` does raise; `kv_cache_dtype="auto"` does not.
- Reading `is_quantized_kv_cache` in the container — returns True for "fp8", False for "auto".
- Reading the model file — the absent `cache_config=` kwarg is the smoking gun.

**Practical implications**:
- The production cluster is reading 2 bytes per K element and 2 bytes per V element on all 10 full-attn layers every decode step (BF16, not FP8 as the documented config suggests). The decode bottleneck is *worse* than the "FP8 baseline" naming implies.
- Conversely, the headroom for compression is *larger* — turboquant_k8v4 is 2.46× compression vs production, not 1.23× vs FP8.
- Fixing the model code to pass `cache_config` through is a one-line patch (mimo_v2.py:368 and :388) and would immediately move production to FP8 KV (1 byte per K/V element). That's a free 2× decode bandwidth win at long context, independent of any TurboQuant work, but **only if FP8 doesn't tickle some other latent bug** — the check at triton_attn_diffkv.py:147 exists for a reason; the bytes are stored interleaved with V at a different element size and the decode kernel needs to slice them as different views. M1 needs to handle this branch correctly anyway.

**Upstream doc PR**: open against `vllm-project/vllm` once M1 lands, documenting:
- `TritonAttentionDiffKVBackend.supported_kv_cache_dtypes` (currently `["auto", "bfloat16"]`).
- That MiMo-V2 layers never pass `cache_config` so `--kv-cache-dtype` is a no-op for those models.

**Deliverable status**: ✅ root cause documented above; upstream doc PR queued for after M1.

### M1 — Bridge `triton_attn_diffkv` to TurboQuant K8V4 (~2-4 days)

K8V4 is FP8 keys + 4-bit values (uniform-quant + per-vector scale/zero). For MiMo-V2's diffkv head dims (Hq=192, Hv=128), the per-position-per-head slot becomes 260 bytes: 192 FP8 K bytes + 64 packed-4bit V bytes + 4 (V fp16 scale+zero). vs the current production BF16's 640 bytes = 2.46× compression.

Concrete file changes (this repo will land as patches, upstream will land as PR):

| Change | Owner | Status |
|---|---|---|
| `mimo_v2.py:368,388` — pass `cache_config=cache_config` to MiMoV2Attention | `patch_mimo_v2_cache_config.py` | drafted |
| `triton_attn_diffkv.py:78` — extend `supported_kv_cache_dtypes` with TQ variants | `patch_triton_attn_diffkv_turboquant.py` | drafted |
| `triton_attn_diffkv.py:147` — let TQ k8v4 through, keep `fp8`/`nvfp4`/etc rejected | same patch | drafted |
| `TritonAttentionDiffKVBackend.get_kv_cache_shape` — return `(num_blocks, block_size, num_kv_heads, tq_slot_size_aligned)` when TQ; else current behavior | same patch | drafted |
| `do_kv_cache_update` — when TQ, call new `triton_turboquant_diffkv_store(K, V, cache, slot_map, hq, hv, kps, val_data_bytes, vqb)` | same patch | drafted (calls new kernel) |
| New `triton_turboquant_diffkv_store.py` — wraps existing `_tq_fused_store_fp8` with a `D_K` / `D_V` split (currently the kernel assumes a single `D`) | new file | skeleton only |
| New `triton_turboquant_diffkv_decode.py` — adapted from `_tq_decode_stage1` with `HEAD_DIM_K` / `HEAD_DIM_V` split; output is `HEAD_DIM_V`-shaped | new file | skeleton only |
| `forward()` path — when TQ, route to new decode stage1+stage2 instead of `unified_attention_diffkv` | same patch as backend | drafted |

What "skeleton" means: the kernel exists as code that compiles and matches the K8V4 layout, but has not been benchmarked, gradient-checked, or validated against a BF16 reference. **Estimated additional work after skeleton lands: 1-2 days of numerical correctness validation, 0.5 day of perf tuning, before C33 is ready to serve.**

#### Why K8V4 first, not K3V4_nc

- K8V4 reuses the model's existing per-tensor FP8 K scales (festr2 stores `k_scale` already; we already declare `MXFP8` attention quant). No rotation, no centroids, no MSE table — the only new code is value 4-bit packing/unpacking.
- K3V4_nc adds Hadamard rotation, Lloyd-Max centroids, and 3-bit packing. ~2× the lines of kernel code and ~4× the numerical surface to validate.
- K8V4 gives the smaller compression (2.46× vs 4.51× over production), but it is the safer first step. K2/M2 picks up the more aggressive variants.

**Deliverable**: a vLLM PR (initially against this fork) that lets MiMo-V2 deployments use `--kv-cache-dtype turboquant_k8v4 --attention-backend triton_attn_diffkv`. Smoke-tested with C33 recipe. Long-context decode should improve roughly 2× at 64-200k context (slot 260 vs 640 bytes; V is the smaller bandwidth fraction so the speedup is somewhat less than the raw ratio).

#### Status — what landed 2026-05-16

- ✅ `patch_mimo_v2_cache_config.py` — applies cleanly; reverted from live container after testing to avoid disturbing C32.
- ✅ `patch_triton_attn_diffkv_turboquant.py` — applies cleanly; reverted from live container after testing.
- ✅ `triton_turboquant_diffkv_store.py` — installed into the container's `/opt/vllm/vllm/v1/attention/ops/`; smoke-tested. Store roundtrip errors: K L2 err 2.6% (FP8 intrinsic), V L2 err 9.7% (4-bit intrinsic). Both are within expected quant noise bounds.
- ✅ `triton_turboquant_diffkv_decode.py` — installed; smoke-tested. Decode output matches a BF16 reference (computed against the same dequantized cache contents) to **1.1×10⁻⁸ absolute / 2.1×10⁻⁷ relative**. The kernel is numerically sound for the simple case (head_size_q=192, head_size_v=128, NH_kv=1, NH_q=4, seq_len=32, single decode token).
- ✅ `launch-cluster.sh` — patches wired behind `ENABLE_TQ_K8V4=1` env (off by default so C32 is unaffected).
- ✅ `recipes/4x-spark-cluster/mimo-v2.5-pro-c33.yaml` — sets the env flag and `--kv-cache-dtype turboquant_k8v4`.

#### Status — what's NOT done (blocks C33 production-readiness)

- ❌ **Prefill path** — currently raises NotImplementedError. The first inference request needs prefill (q_len > 1); without a working prefill the C33 recipe will crash immediately. Plan: copy the dequant-and-SDPA prefill pattern from `TurboQuantAttentionImpl._flash_attn_varlen` (it materializes the cached K/V to BF16 and calls flash-attn). Adapt for diffkv head dims. Estimated 0.5-1 day.
- ❌ **Per-tile-load striding correctness** — store kernel uses `key.contiguous()`/`value.contiguous()`; need to verify the diffkv attention path produces K and V with the assumed contiguous strides under all batch shapes (TP=8 produces num_kv_heads=1 per worker; verify TP=4 case too).
- ❌ **Engine-shape validation** — smoke test ran on 1×1×4×32 micro-shape. Real shapes: query [B=1..4, 128, 192], cache shape [num_blocks, 16, 1, 260]. The KV-group broadcast inside the kernel (Hq//Hk = 16) hasn't been exercised; the smoke test only used Hq//Hk = 4.
- ❌ **CUDA graph capture** — TritonAttentionDiffKVMetadataBuilder allocates a `softmax_segm_output` buffer for the BF16 path. The TQ decode kernel doesn't use that buffer; need to verify capture works with the alternate execution path.
- ❌ **MTP draft-head path** — `mimo_v2_mtp.py` instantiates its own attention layers (the MTP draft head) which also need cache_config plumbing. Not patched in this round.
- ❌ **Numerical validation against real production-shape decode** — the 1e-8 match was against a manually-dequantized reference. Real test should be: serve a known-good prompt at FP8 (after fix Patch A) vs at TQ-K8V4, compare token logprobs/output.

End-to-end C33 readiness estimate: **2-3 focused days** after this session. Of that, the prefill fallback is the largest single piece (~1 day), and shape-correctness debugging is hard to time-box.

### M2 — Extend to `turboquant_k3v4_nc` / `4bit_nc` (~2 days after M1 lands in production)

Once K8V4 works in production (M1 complete + C33 validated), the asymmetric slot layout work is done — adding more aggressive K compression is incremental.

Concrete additions:
- **K3 store** — extend `_tq_diffkv_store_k8v4` to a `_tq_diffkv_store_mse` variant: bucketize K vector to centroids via binary search, pack 3 bits per coordinate, store fp16 vec_norm. Pattern is in the existing `_tq_fused_store_mse` (single-D); the diffkv version splits dim_K from dim_V same as M1.
- **K3 decode** — extend `_tq_diffkv_decode_stage1_k8v4` to load K via MSE indices + centroids table + per-vec norm.
- **Norm-correction (NC)** — re-normalize centroid vectors to unit norm before inverse rotation. Pattern in the existing TQ decode at the "NORM_CORRECTION" branch.
- **Hadamard rotation pre-step** — the MSE K variants apply Hadamard rotation to K before quantization. Need to pre-rotate K in the store kernel and Q in the decode kernel (or fold rotation into Q matrix offline). Pattern in `triton_turboquant_decode.py`.

| Variant | New code over M1 | Risk |
|---|---|---|
| `turboquant_k8v4` | (M1 baseline) | 4-bit V quality |
| `turboquant_4bit_nc` | + 4-bit K + NC | K rotation correctness, more places quant noise enters |
| `turboquant_k3v4_nc` | + 3-bit K + NC | K precision dropping below 8 levels |
| `turboquant_3bit_nc` | + 3-bit K + 3-bit V + NC | both extremes; long-range retrieval may degrade noticeably |

**Deliverable**: second commit on the PR. Long-context decode should improve to 3-4× vs production BF16. Quality validation (M3) decides which variant is the production setting.

### M3 — Quality validation (~half day)

The user's workload is "primary coder for big projects." The right quality test is:
- Long-input retrieval probes: paste a known string at position N in a 100k+ token codebase prompt, ask the model to retrieve it. Plot retrieval rate vs N for each KV dtype.
- Real coding tasks at long context: produce a multi-file refactor over a 32k+ codebase context. Compare diff quality across KV dtypes.

If quality at `turboquant_k3v4_nc` is acceptable for coding tasks, that's the production setting (≈3× decode speedup at 200k).

**Deliverable**: a `QUALITY-LONG-CONTEXT.md` table in this repo with retrieval-probe results.

### M4 — Upstream the work (~half day)

The patches from M1+M2 should go upstream. The contribution is "diffkv support for TurboQuant" which benefits not just festr2 but all MiMo-V2 family models on TurboQuant.

**Deliverable**: an `[Attention] TurboQuant: diffkv support for unequal Q/V head_dim` PR against vllm-project/vllm.

## Out of scope (in this doc)

- **SnapKV / H2O / Quest** — postponed in favor of TurboQuant for the reasons above.
- **RAG / retrieval at the application layer** — orthogonal. The production answer is to combine TurboQuant *and* RAG, not pick one.
- **MTP integration with TurboQuant** — open question. The MTP draft head also has full-attention layers and would benefit similarly; design will need to handle MTP-side compression coherently with main-model side.

## References

- TurboQuant in vLLM: PR vllm-project/vllm#38479 (merged), #40194 (merged), #40108 (open — SWA support)
- SnapKV paper: Li et al. 2024 — https://arxiv.org/abs/2404.14469
- StreamingLLM: Xiao et al. 2024 — https://arxiv.org/abs/2309.17453
- This repo's prior work on diffkv: `patch_mimo_qkv_split.py`, `patch_mimo_v2_mtp_qkv_split.py`
