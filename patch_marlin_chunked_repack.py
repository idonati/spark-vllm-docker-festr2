"""
Patch: stream Marlin NVFP4 MoE weight + scale repack into a preallocated output
instead of accumulating a Python list and torch.cat'ing at the end.

Target function: prepare_nvfp4_moe_layer_for_marlin (marlin_utils_fp4.py:288)
This is what gets called for ModelOpt-NvFp4 + MARLIN backend (`elif nvfp4_backend
== NvFp4MoeBackend.MARLIN:` in oracle/nvfp4.py:367). The earlier patch (which
targeted prepare_moe_fp4_layer_for_marlin / line 397) was for the
compressed-tensors MXFP4 path, not the modelopt NVFP4 path — wrong function.

Without this patch, the inner repack_weight + premute_scales helpers each do:
    for i in range(E):
        ...
        tensor_list.append(per_expert_tensor)
    return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

which holds E (=48 local-experts on TP=8) full per-expert tensors plus the
torch.cat result simultaneously — three "full" copies in flight at peak:
input weight + accumulated list + concat result. For festr2/MiMo-V2.5-Pro
on TP=8 GB10, that pushes peak temporary past the 121 GiB unified-mem
ceiling on the Ray head node (which loses ~3-4 GiB to API + EngineCore +
Ray GCS overhead vs worker nodes).

After this patch each helper:
  1. runs expert 0 to learn the per-expert marlin output shape,
  2. pre-allocates a single (E, ...) output tensor,
  3. streams expert results straight into output[i] via .copy_().

Peak temp memory: just one per-expert scratch (~750 MiB) instead of the
accumulated ~36 GiB. Saves enough headroom for the head node to fit.

Author: Soares Mission Control (Cycle 12 v2), 2026-05-13.
"""
import re
import sys

CANDIDATES = [
    "/opt/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
]

path = None
for p in CANDIDATES:
    try:
        with open(p) as f:
            content = f.read()
        path = p
        break
    except FileNotFoundError:
        continue

if path is None:
    print("SKIP-not-found")
    sys.exit(0)

MARKER_V2 = "# SOARES-C12-v2: chunked nvfp4 marlin repack"
if MARKER_V2 in content:
    print("NOOP-already-patched-v2")
    sys.exit(0)

# Strip the v1 patch first if present (it patched the wrong function and is now
# harmless but confusing).
if "# SOARES-C12: chunked marlin repack" in content:
    print("INFO: v1 marker present (wrong function), continuing with v2 on the right one")

old_repack_weight = '''    # WEIGHT
    # Repack weights to marlin format
    def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        assert weight.shape == (E, size_n, size_k // 2)

        for i in range(E):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            tensor_list.append(marlin_qweight)

        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
'''

new_repack_weight = '''    # WEIGHT
    # Repack weights to marlin format (chunked-stream, ''' + MARKER_V2 + ''')
    def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        assert weight.shape == (E, size_n, size_k // 2)

        # Process expert 0 to learn output shape.
        _qw0 = weight[0].view(torch.int32).T.contiguous()
        _mqw0 = ops.gptq_marlin_repack(
            b_q_weight=_qw0,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        out = torch.empty(
            (E,) + tuple(_mqw0.shape),
            dtype=_mqw0.dtype,
            device=_mqw0.device,
        )
        out[0].copy_(_mqw0)
        del _qw0, _mqw0

        for i in range(1, E):
            qw = weight[i].view(torch.int32).T.contiguous()
            mqw = ops.gptq_marlin_repack(
                b_q_weight=qw,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            out[i].copy_(mqw)
            del qw, mqw

        import torch as _t; _t.cuda.empty_cache()
        return out
'''

if old_repack_weight not in content:
    print("FAIL-repack_weight-no-match")
    idx = content.find("def repack_weight")
    if idx >= 0:
        print("--- context ---", file=sys.stderr)
        print(content[idx:idx + 1500], file=sys.stderr)
    sys.exit(1)

content = content.replace(old_repack_weight, new_repack_weight, 1)

# Now patch the scales helper similarly. It's smaller per-tensor (~few MiB
# per expert) so less critical, but follows the same anti-pattern — patching
# it for consistency and a few more GiB of headroom.
old_premute_scales = '''    # WEIGHT SCALES
    # Permute scales
    def premute_scales(
        scales: torch.Tensor, g_scales: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scales = scales.to(param_dtype)

        tensor_list = []
        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        # All experts share one global_scale, so compute the max
        # scale_factor across all experts first, then apply uniformly.
        combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)

        for i in range(E):
            scale = scales[i].T
            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=GROUP_SIZE,
                is_a_8bit=is_a_8bit,
            )
            marlin_scales, _ = nvfp4_marlin_process_scales(
                marlin_scales, scale_factor=combined_scale_factor, a_dtype=param_dtype
            )
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)
        g_scales = g_scales / combined_scale_factor
        return scales, g_scales
'''

new_premute_scales = '''    # WEIGHT SCALES
    # Permute scales (chunked-stream, ''' + MARKER_V2 + ''')
    def premute_scales(
        scales: torch.Tensor, g_scales: torch.Tensor, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scales = scales.to(param_dtype)

        num_shards = 2 if is_act_and_mul else 1
        if "w13" in name:
            size_n, size_k = N * num_shards, K
        else:
            size_n, size_k = K, N

        # All experts share one global_scale, so compute the max
        # scale_factor across all experts first, then apply uniformly.
        combined_scale_factor = _nvfp4_compute_scale_factor(scales, param_dtype)

        # Expert 0 to learn output shape.
        _s0 = scales[0].T
        _ms0 = marlin_permute_scales(
            s=_s0,
            size_k=size_k,
            size_n=size_n,
            group_size=GROUP_SIZE,
            is_a_8bit=is_a_8bit,
        )
        _ms0, _ = nvfp4_marlin_process_scales(
            _ms0, scale_factor=combined_scale_factor, a_dtype=param_dtype
        )
        out = torch.empty(
            (E,) + tuple(_ms0.shape),
            dtype=_ms0.dtype,
            device=_ms0.device,
        )
        out[0].copy_(_ms0)
        del _s0, _ms0

        for i in range(1, E):
            s = scales[i].T
            ms = marlin_permute_scales(
                s=s,
                size_k=size_k,
                size_n=size_n,
                group_size=GROUP_SIZE,
                is_a_8bit=is_a_8bit,
            )
            ms, _ = nvfp4_marlin_process_scales(
                ms, scale_factor=combined_scale_factor, a_dtype=param_dtype
            )
            out[i].copy_(ms)
            del s, ms

        g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)
        g_scales = g_scales / combined_scale_factor
        return out, g_scales
'''

if old_premute_scales not in content:
    print("FAIL-premute_scales-no-match")
    idx = content.find("def premute_scales")
    if idx >= 0:
        print("--- context ---", file=sys.stderr)
        print(content[idx:idx + 1500], file=sys.stderr)
    sys.exit(1)

content = content.replace(old_premute_scales, new_premute_scales, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED-v2")
