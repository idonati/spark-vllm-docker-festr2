# C35 — cutlass-dsl 4.5.1 rebuild + sm_121a re-test (plan)

## Status

Drafted 2026-05-19. Not yet executed. Blocked on: (a) prerequisites pass, (b) free cluster slot for the boot + bench window.

## Goal

Replace the Marlin NVFP4 → FP8 dequant MoE path with **native cutlass-dsl 4.5.1 NVFP4 MoE** on sm_121a, and quantify the impact on C34's 29.5 ms decode intercept (MoE GEMM at batch=1 lives in there). Also fulfills the public commitment to `NVIDIA/cutlass#3227` — we said we'd report sm_121a once the cluster is back.

## Background

- Cutlass-dsl 4.5.0 has a `_mma` ptxas lowering bug on `sm_120/120f/121a` (NVIDIA/cutlass#3227). On sm_121a this produces garbage NVFP4 MoE output. We work around it with Marlin (FP4 → FP8 dequant on every MoE call), via the three env vars in [SM121 NVFP4 Marlin workaround] memory.
- hassan-abdallah confirmed cutlass-dsl 4.5.1 fixes the bug on `sm_120a`. `sm_121a` is a different ArchTag — needs separate verification.
- C34 finding: 81% of decode cost at 26k is fixed per-step intercept. MoE GEMM at batch=1 is in there. Native NVFP4 (no per-call dequant) is the lever; this rebuild quantifies it.
- depaulmillz's wheel-collision fix (4.5.0) involved `nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13` overlapping 168 `.py` files with different SHA256. 4.5.1 may have a different split; confirm during prereqs.

## Prerequisites (~5 min)

1. Read `Dockerfile.mimo-pr41797-nccl230-tfgit` on AO1 → confirm current cutlass-dsl pin (likely 4.5.0 with depaulmillz workaround).
2. `df -h /var/lib/docker` on AO1 + each worker → confirm headroom for another ~30 GB image. AO1 had 2.7 TB free at last check; verify peers.
3. Verify the festr2 patches (mimo_qkv_split, modelopt_mxfp8_dispatch, marlin_chunked_repack, MTP qkv-split) still target the vllm-pr41797 source paths — none of them touch cutlass paths so they should all still apply.

## Build (~30-40 min on AO1)

1. Copy `Dockerfile.mimo-pr41797-nccl230-tfgit` → `Dockerfile.mimo-pr41797-nccl230-tfgit-cutlass451`.
2. Pin in the new Dockerfile (depaulmillz collision-fix flags carried forward; drop if 4.5.1 no longer needs them):

   ```dockerfile
   RUN pip install --no-deps --force-reinstall \
       nvidia-cutlass-dsl[cu13]==4.5.1
   ```

3. Build tagged as `vllm-node-mimo-pr41797-nccl230-tfgit-cutlass451-sm12x:latest`. **Keep the existing image** so rollback is `--container` swap, no rebuild.
4. Broadcast image to AO2-AO8 via the launcher's existing image-copy path, or one-off `docker save | ssh peer 'docker load'`.

## Smoke-test gate (~15-20 min, single boot)

Create `recipes/4x-spark-cluster/mimo-v2.5-pro-c35-cutlass.yaml`:

- `container: vllm-node-mimo-pr41797-nccl230-tfgit-cutlass451-sm12x` (new image)
- `--moe-backend cutlass` (replaces marlin)
- **Drop** the `VLLM_USE_FLASHINFER_MOE_FP4=0` + `NCCL_NVLS_ENABLE=0` env vars (those routed around Marlin; revisit with native cutlass)
- Keep everything else from C34a (`max_num_seqs=16`, widened cudagraph capture set).

Single-prompt smoke check via `bench_client.py --mode smoke`:

- Expected: `"Hey there"` / `"Hello!"` or similar coherent text.
- **Garbage signature** (`あるい`, `ייך`, `*\0`, `!!!!!!!`) → 4.5.1 still hits the sm_121a ptxas bug. Stop, write up negative result, revert.

## Bench (~10 min, same boot if smoke passes)

Run `bench_client.py --mode full-sweep` + `longctx_concurrent.py` from `/home/idonati/profiles/`. Direct comparison vs C34a baseline:

| Metric | C34a baseline | C35 (native cutlass) | Δ |
|---|---|---|---|
| Solo decode, short ctx | 33.5 tok/s | TBD | TBD |
| Solo decode, 26k | 27.0 tok/s | TBD | TBD |
| 4-concurrent agg, short | 83.6 tok/s | TBD | TBD |
| 4-concurrent agg, 26k | 73.3 tok/s | TBD | TBD |
| Intercept (linear fit) | 29.5 ms/tok | TBD | TBD |
| Slope (per 1k ctx) | 0.265 ms/tok | ≈0.265 expected (slope is KV-read, not MoE) | sanity check |

The slope should be roughly unchanged — MoE is per-step, not per-context-token. If the slope changes meaningfully, something else is moving and we need to investigate before attributing intercept changes to cutlass.

## Decision tree (post-bench)

- **Intercept drops ≥ 4 ms AND output is coherent**: promote C35 to production recipe. Update BENCHMARKS.md C35 section. Post positive result on cutlass#3227 and vllm#41519. Native cutlass NVFP4 MoE is the new default.
- **Smoke passes but intercept drops < 2 ms**: write up as "tested, no significant gain, keeping Marlin." Post that on the issues — still useful data. Cutlass-dsl 4.5.1 is then a correctness option (if Marlin ever regresses) but not a perf win.
- **Smoke fails (garbage output)**: 4.5.1's sm_121a path still broken. Post detailed repro on cutlass#3227 (hassan-abdallah confirmed sm_120a — our sm_121a regression would narrow the bug to a specific ArchTag, useful for NVIDIA).

## Risks / gotchas

- **Wheel collision regression**: 4.5.1 may have a different libs-base/cu13 split. If install errors on `nvidia-cutlass-dsl-libs-base` overlap, retry with explicit `--ignore-installed` on the conflicting wheel.
- **CUDA runtime mismatch**: 4.5.1's cu13 build is for CUDA 13. The current image is on CUDA 13, but a CUDA-13-minor mismatch could fault at first MoE call. The smoke test catches this.
- **Build OOM**: pip install can spike memory. Use `--memory 80g --memory-swap 90g` build constraints if OOM appears.
- **Image distribution time**: 30 GB × 7 workers over rail-1 ~ 5-10 min depending on NIC speed.
- **Negative result still publishable**: hassan-abdallah sm_120a-passes + our sm_121a-fails would help NVIDIA narrow the bug surface — that's a contribution to #3227 even if it doesn't help us serve faster.

## Time budget

| Phase | Estimate |
|---|---|
| Prerequisites | 5 min |
| Build + broadcast | 35-50 min |
| Boot c35 | 15-20 min |
| Smoke + bench | 10 min |
| Writeup + commits + issue replies | 15 min |
| **Total** | **~80-100 min** |

## Out of scope

- Migrating the festr2 HF model card to recommend `--moe-backend cutlass`. Decide after C35 lands.
- Touching `VLLM_USE_FLASHINFER_MOE_FP4` defaults. Separate cleanup once we know whether cutlass-dsl or FlashInfer is the right path on sm_121a.
- Combining C35 (cutlass) with C33 (TQ-K8V4). Sequential, not in this task.

## References

- `NVIDIA/cutlass#3227` — the open issue tracking the `_mma` ptxas lowering bug
- `vllm#41519` — MiMo-V2.5 on SM12x thread; we owe a follow-up there too
- C34 results in `BENCHMARKS.md` — establishes the baseline this plan compares against
- [SM121 NVFP4 Marlin workaround] auto-memory entry — the three env vars we're trying to remove
