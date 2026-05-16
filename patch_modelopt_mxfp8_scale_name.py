"""
Patch: rename ModelOptMxFp8LinearMethod's scale parameter from `weight_scale`
to `weight_scale_inv` so it matches festr2/modelopt's checkpoint key, then
rename it back during process_weights_after_loading so the dequant kernels
(Marlin / Emulation / FlashInferCutlass) keep reading `layer.weight_scale`
unchanged.

NO byte-flip. C19 verified empirically that festr2's scales are stored as
forward E8M0 bytes (typical value range 112-119 -> 2^(byte-127) ~ 1e-4,
matching real weight amax magnitudes). The `_inv` suffix is festr2's naming
convention, not an inverse-bias encoding. Applying `254 - byte` blew scales
up by a factor ~10^7 and produced one-token-dominated logits (`ERRQ` spam).

Also: emit a one-shot diagnostic per rank printing the loaded scale's range
and a few sample bytes, so we can verify the load worked without re-running
with extra logging glue.

Author: Soares Mission Control (Cycle 20), 2026-05-15.
"""
import sys

CANDIDATES = [
    "/opt/vllm/vllm/model_executor/layers/quantization/modelopt.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py",
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

MARKER = "# SOARES-C20: rename weight_scale -> weight_scale_inv (no flip)"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

old_create = '''        # Weight scale tensor (E8M0 encoded as uint8), one scale per block of 32 along K
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)'''
new_create = '''        # ''' + MARKER + '''
        weight_scale_inv = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_inv", weight_scale_inv)'''
if old_create not in content:
    print("FAIL-create_weights-not-found")
    sys.exit(1)
content = content.replace(old_create, new_create, 1)

old_proc = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Validate weight tensor
        if layer.weight.ndim != 2:
            raise ValueError(
                f"MXFP8 weight must be 2D tensor [N, K], got {layer.weight.ndim}D "
                f"with shape {tuple(layer.weight.shape)}"
            )'''
new_proc = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ''' + MARKER + '''
        if hasattr(layer, "weight_scale_inv") and not hasattr(layer, "weight_scale"):
            import torch.nn as _nn
            import os as _os
            _ws = layer.weight_scale_inv.data
            del layer._parameters["weight_scale_inv"]
            layer.register_parameter(
                "weight_scale",
                _nn.Parameter(_ws, requires_grad=False),
            )
            if not globals().get("_SOARES_C20_DIAG", False):
                _rank = _os.environ.get("RANK", _os.environ.get("LOCAL_RANK", "?"))
                _sample = _ws.flatten()[:8].tolist()
                print(
                    f"[SOARES-C20-DIAG rank={_rank}] "
                    f"weight.shape={tuple(layer.weight.shape)} "
                    f"weight.dtype={layer.weight.dtype} "
                    f"scale.shape={tuple(_ws.shape)} "
                    f"scale.dtype={_ws.dtype} "
                    f"scale[:8]={_sample} "
                    f"scale.min={_ws.min().item()} "
                    f"scale.max={_ws.max().item()}",
                    flush=True,
                )
                globals()["_SOARES_C20_DIAG"] = True

        # Validate weight tensor
        if layer.weight.ndim != 2:
            raise ValueError(
                f"MXFP8 weight must be 2D tensor [N, K], got {layer.weight.ndim}D "
                f"with shape {tuple(layer.weight.shape)}"
            )'''
if old_proc not in content:
    print("FAIL-process_weights-not-found")
    sys.exit(1)
content = content.replace(old_proc, new_proc, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
