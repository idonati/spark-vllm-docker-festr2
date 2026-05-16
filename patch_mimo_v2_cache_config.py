"""
Patch: fix vLLM's mimo_v2.py decoder layers to pass cache_config to
MiMoV2Attention.

Without this, both branches of MiMoV2FlashDecoderLayer.__init__
(SWA and full-attn) instantiate MiMoV2Attention without cache_config.
That means MiMoV2Attention.__init__ passes cache_config=None into
Attention(...), which in turn (per attention.py:224-228) defaults
kv_cache_dtype to "auto" — so the global --kv-cache-dtype flag is
silently ignored for all 70 attention layers in MiMo-V2.

Concretely, --kv-cache-dtype fp8 ends up as kv_cache_dtype="auto"
(i.e. BF16) in every diffkv impl. This patch wires cache_config through
so the CLI flag actually applies. This is a prerequisite for the
TurboQuant K8V4 work — without it, --kv-cache-dtype turboquant_k8v4
would also be silently ignored.

Idempotent: skip if MIMO-CACHE-CONFIG-PLUMBING marker already present.

Safety: keeps existing behavior unchanged for users who don't set
--kv-cache-dtype (cache_config.cache_dtype defaults to "auto"); only
changes behavior when the user has explicitly asked for a non-auto KV
cache dtype, which previously was a silent no-op.
"""
import sys

CANDIDATES = [
    "/opt/vllm/vllm/model_executor/models/mimo_v2.py",
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

MARKER = "# MIMO-CACHE-CONFIG-PLUMBING"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

# There are two instantiation sites of MiMoV2Attention in
# MiMoV2FlashDecoderLayer.__init__ — one for the SWA branch, one for
# the full-attn branch. Both omit cache_config. We patch both.

# Site 1: SWA branch
old_swa = '''            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )'''

new_swa = '''            # ''' + MARKER + '''
            # Pass cache_config so --kv-cache-dtype actually applies.
            # Without this, Attention.__init__ sees cache_config=None and
            # defaults kv_cache_dtype="auto" regardless of the CLI flag.
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )'''

if old_swa not in content:
    print("FAIL-anchor-swa-not-found")
    sys.exit(1)
content = content.replace(old_swa, new_swa, 1)

# Site 2: full-attn branch — locate by reading more lines after the else:
# We anchor on the exact opening 4 lines of the full-attn block.
old_full = '''            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,'''

# Match the same prefix and inject cache_config kwarg right after
# max_position_embeddings (preserving everything below).
new_full = '''            # ''' + MARKER + '''
            # Pass cache_config so --kv-cache-dtype actually applies.
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,'''

if old_full not in content:
    print("FAIL-anchor-full-not-found")
    sys.exit(1)
content = content.replace(old_full, new_full, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
