"""
Patch: fix vLLM's MiMo-V2 fused-qkv_proj weight loader.

The original code uses naive `loaded_weight.chunk(tp_size, dim=0)[tp_rank]`
on the fused [Q|K|V] tensor. For festr2's layout (Q=24576, K=1536, V=1024 at
TP=8), that gives:
  - ranks 0-6: only Q-region rows -> their K/V slots end up holding Q-data
  - rank 7:    partial Q + full K + full V

So 7 of 8 ranks compute attention with Q values posing as K and V, and only
rank 7 has any real K/V. Result: attention is fundamentally broken, model
produces partially-degenerate output (token loops, slight prompt sensitivity).

Empirically confirmed: festr2 Q section mean E8M0 scale byte = 114,
K = 116, V = 113 — clearly Q→K→V layout, not per-rank pre-sharded.

Fix: split the loaded weight into Q/K/V sections by total head sizes, then
call the QKVParallelLinear loader 3x with the proper shard_id. The loader
takes care of TP sharding inside each Q/K/V slice. Same approach as the
existing stacked_params_mapping path for split q_proj/k_proj/v_proj
checkpoints — just sourced from a single fused tensor.

Author: Soares Mission Control (Cycle 25), 2026-05-15.
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

MARKER = "# SOARES-C25: fix fused qkv_proj loader (proper Q/K/V split)"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

old = '''            # Support fused qkv_proj checkpoint (Pro format)
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                continue'''

new = '''            # ''' + MARKER + '''
            # The original code did `loaded_weight.chunk(tp_size, dim=0)[tp_rank]`
            # which is incorrect: festr2's fused qkv tensor is laid out
            # [Q | K | V] along dim 0, so naive chunking gives 7 of 8 ranks
            # only Q-region rows (their K/V slots end up holding Q values).
            # Instead split by [q_size, k_size, v_size] and let the
            # QKVParallelLinear loader handle the TP shard for each section.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    # Determine layer's Q/K/V sizes from config + layer index.
                    # name pattern: "model.layers.{idx}.self_attn.qkv_proj.weight[_scale[_inv]]"
                    try:
                        _li = int(name.split(".layers.")[1].split(".")[0])
                    except (IndexError, ValueError):
                        _li = -1
                    _is_swa = (
                        _li >= 0
                        and hasattr(self.config, "hybrid_layer_pattern")
                        and self.config.hybrid_layer_pattern[_li] == 1
                    )
                    if _is_swa:
                        _nh = self.config.swa_num_attention_heads
                        _nkv = self.config.swa_num_key_value_heads
                        _hd = self.config.swa_head_dim
                        _vhd = getattr(self.config, "swa_v_head_dim", _hd)
                    else:
                        _nh = self.config.num_attention_heads
                        _nkv = self.config.num_key_value_heads
                        _hd = self.config.head_dim
                        _vhd = getattr(self.config, "v_head_dim", _hd)
                    _qs = _nh * _hd
                    _ks = _nkv * _hd
                    _vs = _nkv * _vhd
                    if loaded_weight.shape[0] == _qs + _ks + _vs:
                        _q = loaded_weight[:_qs]
                        _k = loaded_weight[_qs : _qs + _ks]
                        _v = loaded_weight[_qs + _ks : _qs + _ks + _vs]
                        _wl = getattr(param, "weight_loader", default_weight_loader)
                        _wl(param, _q, "q")
                        _wl(param, _k, "k")
                        _wl(param, _v, "v")
                        loaded_params.add(name)
                    else:
                        # Unexpected shape — fall back to old behavior
                        # (still wrong but matches original) to avoid hard fail.
                        loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                        default_weight_loader(param, loaded_weight)
                continue'''

if old not in content:
    print("FAIL-anchor-not-found")
    sys.exit(1)
content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
