"""
Patch: fix vLLM's mimo_v2_mtp.py fused-qkv_proj loader (same root cause as
the mimo_v2.py patch in patch_mimo_qkv_split.py).

mimo_v2_mtp.py at line ~298 does `loaded_weight.chunk(tp_size, dim=0)[tp_rank]`
on the fused [Q|K|V] tensor stored in festr2's MTP weights. Identical bug
shape to the main-model one: 7 of 8 ranks (at TP=8) get Q-region rows in
their K/V parameter slots, so MTP attention runs on Q values posing as K/V,
producing near-random draft tokens (observed acceptance 3-24% vs the 70%+
that a properly-loaded MTP head should achieve).

Fix: split the fused tensor by full-attention [q_size, k_size, v_size]
(festr2's MTP layers use full-attention head dims, not SWA), then call the
QKVParallelLinear loader 3x with shard_id "q"/"k"/"v" — same approach as
patch_mimo_qkv_split.py.
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

MARKER = "# MIMO-MTP-FUSED-QKV-SPLIT-FIX"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

old = '''            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue'''

new = '''            # ''' + MARKER + '''
            # Original code: loaded_weight.chunk(tp_size, dim=0)[tp_rank] on the
            # fused [Q|K|V] tensor mis-slots Q values into K/V slots on
            # (tp_size - 1) of tp_size ranks, breaking MTP attention.
            # Split [Q|K|V] using full-attention head dims (festr2 MTP uses
            # full attention, not SWA) and use the QKVParallelLinear loader
            # with shard_id "q"/"k"/"v" -- identical fix to mimo_v2.py.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
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
                        # Fall back to old behavior if shape unexpected.
                        loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                        default_weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                continue'''

if old not in content:
    print("FAIL-anchor-not-found")
    sys.exit(1)
content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
