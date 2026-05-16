"""
Patch: thread cache_config through vLLM's mimo_v2_mtp.py MTP draft-head
chain so the MTP self-attention layer also honors --kv-cache-dtype.

Without this, even after patch_mimo_v2_cache_config.py fixes the main
model's 70 layers, the MTP draft-head layer still instantiates
MiMoV2Attention without cache_config, defaulting kv_cache_dtype="auto"
for that single layer. For TurboQuant K8V4 this is fatal: the MTP slot
allocation would mismatch the main-model slot shape, and the engine
would crash at first speculative-decode step.

The chain to thread:
  MiMoV2MultiTokenPredictor.__init__   (has vllm_config)
    └→ _MiMoV2MTPLayers.__init__       (add cache_config param)
        └→ MiMoV2MTPLayer.__init__     (add cache_config param)
            └→ MiMoV2Attention(... cache_config=cache_config, ...)

Idempotent: skip if MIMO-MTP-CACHE-CONFIG-PLUMBING marker already present.
"""
import sys

CANDIDATES = [
    "/opt/vllm/vllm/model_executor/models/mimo_v2_mtp.py",
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

MARKER = "# MIMO-MTP-CACHE-CONFIG-PLUMBING"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

# -------------------------------------------------------------------
# Edit 1: MiMoV2MTPLayer.__init__ — add cache_config param + pass to
# MiMoV2Attention(...) call.
# -------------------------------------------------------------------
old_layer_sig = '''    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()'''

new_layer_sig = '''    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        cache_config=None,  ''' + MARKER + '''
    ) -> None:
        super().__init__()'''

if old_layer_sig not in content:
    print("FAIL-anchor-layer-sig-not-found")
    sys.exit(1)
content = content.replace(old_layer_sig, new_layer_sig, 1)

# Inject cache_config kwarg into the MiMoV2Attention(...) call inside
# MiMoV2MTPLayer.__init__. Anchor on max_position_embeddings line which
# is unique to this call site (the file has only one MiMoV2Attention).
old_attn = '''            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            quant_config=quant_config,'''

new_attn = '''            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            cache_config=cache_config,  ''' + MARKER + '''
            quant_config=quant_config,'''

if old_attn not in content:
    print("FAIL-anchor-attn-not-found")
    sys.exit(1)
content = content.replace(old_attn, new_attn, 1)

# -------------------------------------------------------------------
# Edit 2: _MiMoV2MTPLayers.__init__ — add cache_config param + forward.
# -------------------------------------------------------------------
old_layers_sig = '''    def __init__(
        self,
        config: PretrainedConfig,
        num_mtp_layers: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): MiMoV2MTPLayer(
                    config=config,
                    prefix=f"{prefix}.{i}",
                    quant_config=quant_config,
                )
                for i in range(num_mtp_layers)
            }
        )'''

new_layers_sig = '''    def __init__(
        self,
        config: PretrainedConfig,
        num_mtp_layers: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        cache_config=None,  ''' + MARKER + '''
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): MiMoV2MTPLayer(
                    config=config,
                    prefix=f"{prefix}.{i}",
                    quant_config=quant_config,
                    cache_config=cache_config,  ''' + MARKER + '''
                )
                for i in range(num_mtp_layers)
            }
        )'''

if old_layers_sig not in content:
    print("FAIL-anchor-layers-sig-not-found")
    sys.exit(1)
content = content.replace(old_layers_sig, new_layers_sig, 1)

# -------------------------------------------------------------------
# Edit 3: MiMoV2MultiTokenPredictor.__init__ — pass vllm_config.cache_config
# to _MiMoV2MTPLayers(...).
# -------------------------------------------------------------------
old_predictor_call = '''        self.mtp = _MiMoV2MTPLayers(
            config=config,
            num_mtp_layers=num_mtp_layers,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "mtp.layers"),
        )'''

new_predictor_call = '''        self.mtp = _MiMoV2MTPLayers(
            config=config,
            num_mtp_layers=num_mtp_layers,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "mtp.layers"),
            cache_config=vllm_config.cache_config,  ''' + MARKER + '''
        )'''

if old_predictor_call not in content:
    print("FAIL-anchor-predictor-call-not-found")
    sys.exit(1)
content = content.replace(old_predictor_call, new_predictor_call, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
