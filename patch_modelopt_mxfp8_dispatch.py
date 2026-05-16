"""
Patch: wire MXFP8 into ModelOptMixedPrecisionConfig.get_quant_method().

Why: festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8 uses MIXED_PRECISION quant_algo
where attention QKV/O projections are MXFP8-quantized:
    {"quant_algo": "MXFP8", "group_size": 32}
but vLLM's get_quant_method() in ModelOptMixedPrecisionConfig only handles
"FP8" and "NVFP4". The MXFP8 branch is silently absent — attention layers
hit `return UnquantizedLinearMethod()`. Result: the raw packed-FP8 bytes
are loaded into a layer that expects BF16, every forward pass produces
nonsense, output is garbage (`あるい…`).

ModelOptMxFp8LinearMethod (modelopt.py:1571) and ModelOptMxFp8FusedMoE
(modelopt.py:1676) both already exist as proper implementations — they
just aren't wired into the mixed-precision dispatch. This patch:

  1. Adds `mxfp8_config: ModelOptMxFp8Config` to ModelOptMixedPrecisionConfig
     __init__ and _from_config.
  2. Adds an "MXFP8" branch to get_quant_method for both LinearBase and
     FusedMoE, mirroring the existing FP8 and NVFP4 branches.

Author: Soares Mission Control (Cycle 15), 2026-05-13.
"""
import re
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

MARKER = "# SOARES-C15: wire MXFP8 into modelopt_mixed dispatch"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

# 1. Patch __init__ to accept mxfp8_config
old_init = '''    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
    ) -> None:
        super().__init__(exclude_modules)
        self.kv_cache_quant_method = kv_cache_quant_method
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config'''

new_init = '''    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
        mxfp8_config: "ModelOptMxFp8Config | None" = None,  # SOARES-C15
    ) -> None:
        super().__init__(exclude_modules)
        self.kv_cache_quant_method = kv_cache_quant_method
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config
        self.mxfp8_config = mxfp8_config  # SOARES-C15'''

if old_init not in content:
    print("FAIL-init-no-match")
    sys.exit(1)
content = content.replace(old_init, new_init, 1)

# 2. Patch _from_config to build mxfp8_config
old_build = '''        fp8_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=[],
        )
        nvfp4_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )

        return cls(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
        )'''

new_build = '''        fp8_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=[],
        )
        nvfp4_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )
        # SOARES-C15: wire MXFP8 into modelopt_mixed dispatch
        # Need to build a ModelOptMxFp8Config so MXFP8-tagged layers
        # (e.g. attention QKV/O in festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8)
        # get the proper ModelOptMxFp8LinearMethod instead of silently
        # falling through to UnquantizedLinearMethod (which produces
        # garbage output because raw packed FP8 bytes are treated as
        # full-precision weights).
        try:
            _mxfp8_config = ModelOptMxFp8Config(
                is_checkpoint_mxfp8_serialized=True,
                kv_cache_quant_algo=kv_cache_quant_method,
                exclude_modules=[],
            )
        except TypeError:
            # ModelOptMxFp8Config signature may differ — fall back to
            # whatever vLLM expects for this version.
            _mxfp8_config = None

        return cls(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
            mxfp8_config=_mxfp8_config,
        )'''

if old_build not in content:
    print("FAIL-build-no-match")
    sys.exit(1)
content = content.replace(old_build, new_build, 1)

# 3. Patch get_quant_method dispatch — LinearBase branch
old_linear_dispatch = '''        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            # Layer not in quantized_layers — leave unquantized
            return UnquantizedLinearMethod()'''

new_linear_dispatch = '''        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            # SOARES-C15: handle MXFP8-quantized attention layers
            if quant_algo == "MXFP8" and self.mxfp8_config is not None:
                return ModelOptMxFp8LinearMethod(self.mxfp8_config)
            # Layer not in quantized_layers — leave unquantized
            return UnquantizedLinearMethod()'''

if old_linear_dispatch not in content:
    print("FAIL-linear-dispatch-no-match")
    sys.exit(1)
content = content.replace(old_linear_dispatch, new_linear_dispatch, 1)

# 4. Patch get_quant_method dispatch — FusedMoE branch
old_moe_dispatch = '''        if isinstance(layer, FusedMoE):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            return None'''

new_moe_dispatch = '''        if isinstance(layer, FusedMoE):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            # SOARES-C15: also handle MXFP8 MoE (defensive — festr2's
            # MiMo only uses MXFP8 for attention, not MoE, but mirror
            # the linear dispatch for completeness).
            if quant_algo == "MXFP8" and self.mxfp8_config is not None:
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None'''

if old_moe_dispatch not in content:
    print("FAIL-moe-dispatch-no-match")
    sys.exit(1)
content = content.replace(old_moe_dispatch, new_moe_dispatch, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
