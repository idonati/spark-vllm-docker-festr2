# KV-cache compression for MiMo-V2 (diffkv) — design doc

## Status

Draft / not yet implemented. This doc captures the problem, what vLLM already ships, and the realistic implementation paths.

## Problem

At long context, decode rate on `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8` on the 8× DGX Spark cluster degrades because the 10 full-attention layers (out of 70) read the full KV cache every decode step. Measured at C30 (`max_model_len=262144`, FP8 KV, CUDA graphs, no MTP):

| context | sustained decode tok/s |
|---|---|
| 16k | 6.9 |
| 64k | 2.1 |
| 128k | 1.0 |
| 200k | 0.6 |

The 60 sliding-window (window=128) layers are constant-cost; the bottleneck is the 10 full-attn layers, which scale linearly with context.

The MoE expert weights are NVFP4 and the qkv weights are MXFP8 — those are *weight* quantizations, separate from KV-cache compression.

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

Plus FP8 silently — see `is_quantized_kv_cache` check at `triton_attn_diffkv.py:147`. In practice the cycles in this repo run with `--kv-cache-dtype fp8 --attention-backend triton_attn_diffkv` and the `NotImplementedError` does not fire. The path that lets that through is the open question of Milestone 0 below.

TurboQuant ships with its own attention backend (`TurboQuantAttentionBackend`) and four KV variants (k8v4, k3v4_nc, 4bit_nc, 3bit_nc) implementing 2-4× compression vs BF16, BUT that backend assumes a single per-layer `head_size` and does not understand diffkv. PR vllm#40108 (open) adds sliding-window support to TurboQuant; nothing in flight adds diffkv support.

`sparse_attn_indexer` exists for DeepSeek-V3.2 only. No SnapKV / H2O / StreamingLLM / Quest implementations in this build.

## Why TurboQuant > SnapKV for our case

- **TurboQuant** is uniform, lossless-ish compression of *every* KV slot. Quality cost is from the quantization noise (small, well-characterized). Pairs naturally with vLLM's PagedAttention block structure.
- **SnapKV** evicts low-importance tokens. Quality cost is from the eviction policy (potentially large if the eviction policy misranks an important past token). Eviction also conflicts with vLLM's block-based KV cache management.

Given the engineering effort is comparable and TurboQuant is partially merged upstream, that is the path to pursue.

## Milestones

### M0 — Understand the current FP8/diffkv silent path (a few hours)

`triton_attn_diffkv.py:147` raises `NotImplementedError("TritonAttentionDiffKVBackend does not yet support quantized KV cache")` when `is_quantized_kv_cache(self.kv_cache_dtype)` returns True, which it does for `fp8`. The deployment in this repo runs with `--kv-cache-dtype fp8 --attention-backend triton_attn_diffkv` without that error firing.

Either there's a code path that bypasses the check (e.g. per-layer attention spec selecting a non-DiffKV impl for SWA layers and DiffKV only for full-attn layers, and full-attn layers somehow not hitting the check), or there's a recent change that silently relaxes the check, or our setup is hitting a different impl class.

**Deliverable**: a one-paragraph answer in this doc, plus an upstream issue or doc PR documenting the actual support matrix for diffkv + KV dtypes.

### M1 — Bridge `triton_attn_diffkv` to TurboQuant K8V4 (~1-2 days)

`turboquant_k8v4` is the gentlest TurboQuant variant: 8-bit K (essentially FP8 quality) and 4-bit V. Since the *current* deployment is already (apparently) running 8-bit K and 8-bit V in the diffkv path, going to 4-bit V is a smaller delta in quality risk than going to k3v4 / 3bit_nc / 4bit_nc.

Implementation outline:
1. Extend `TritonAttentionDiffKVImpl.__init__` to accept `turboquant_k8v4` (remove from the `NotImplementedError` branch).
2. In `do_kv_cache_update`, route through the TurboQuant K/V packer (`triton_turboquant_store.py`) instead of the standard FP8 store. The packer needs to know `head_size_q` (192) and `head_size_v` (128) separately for the asymmetric slot layout.
3. In the decode path (`unified_attention_diffkv`), the K-load + score and V-gather need to unpack TurboQuant rather than read FP8.

Risk: the TurboQuant Triton kernel was written for a single per-head head size. A new Triton kernel variant for diffkv is the largest single piece of work here. Could amount to a new `triton_turboquant_diffkv_decode.py` file.

**Deliverable**: a vLLM PR (initially against this fork) that lets MiMo-V2 deployments use `--kv-cache-dtype turboquant_k8v4 --attention-backend triton_attn_diffkv`. Smoke-tested with the C32 recipe. Long-context decode should improve roughly 1.5-2× at 64-200k context (V bandwidth halves; K stays the same).

### M2 — Extend to `turboquant_k3v4_nc` / `4bit_nc` (~1 day after M1)

Once K8V4 works, the harder bit (asymmetric slot layout) is done. Adding the K3 and 4-bit variants is mostly Triton-kernel work to handle the additional pack/unpack widths.

**Deliverable**: a second commit on the PR adding the 3-4 bit variants. Long-context decode should improve roughly 2-3× over FP8 at 64-200k context. Quality validation needed at this point — the more aggressive quant levels do start hurting on long-range retrieval tasks.

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
