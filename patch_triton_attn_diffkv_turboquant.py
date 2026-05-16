"""
Patch: extend TritonAttentionDiffKVBackend / Impl to handle TurboQuant K8V4
(and lay groundwork for the more aggressive variants in M2).

Three logical edits to /opt/vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:

1. supported_kv_cache_dtypes — extend to include turboquant variants so
   the backend's `supports_kv_cache_dtype` advertises them.

2. __init__ — relax the NotImplementedError so it lets TQ k8v4 through.
   Other quantized dtypes (fp8, nvfp4, per_token_head) still raise.

3. get_kv_cache_shape — when kv_cache_dtype starts with "turboquant_",
   return the TQ slot shape (single combined K+V slot per head per
   position) instead of the standard packed [head_size_q + head_size_v].

We also stash the layer's TurboQuant config and a flag we use later in
forward() / do_kv_cache_update to dispatch to the new
triton_turboquant_diffkv_{store,decode} kernels (those land in
companion patches).

Idempotent: skip if MIMO-DIFFKV-TURBOQUANT marker already present.
"""
import sys

CANDIDATES = [
    "/opt/vllm/vllm/v1/attention/backends/triton_attn_diffkv.py",
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

MARKER = "# MIMO-DIFFKV-TURBOQUANT"
if MARKER in content:
    print("NOOP-already-patched")
    sys.exit(0)

# ---- Edit 1: supported_kv_cache_dtypes -------------------------------
old1 = '''    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]'''

new1 = '''    # ''' + MARKER + '''
    # Originally fp16/bf16/fp32 only.  We add TurboQuant variants here;
    # whether the kernels can actually serve a given variant is decided
    # in __init__ below (currently only k8v4 is implemented).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "turboquant_k8v4",
        # Reserved for M2; __init__ will still raise until decode kernel
        # supports them:
        # "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc",
    ]'''

if old1 not in content:
    print("FAIL-anchor-1-not-found")
    sys.exit(1)
content = content.replace(old1, new1, 1)

# ---- Edit 2: __init__ check ------------------------------------------
old2 = '''    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )'''

new2 = '''    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ''' + MARKER + '''
        # K8V4 is supported via the new triton_turboquant_diffkv_* path.
        # Other quantized KV dtypes still raise.
        self._tq_config = None
        if self.kv_cache_dtype == "turboquant_k8v4":
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            # K cache is sized by head_size_q (Hq=192 for MiMo-V2).
            self._tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, head_dim=self.head_size
            )
            # V cache is sized by head_size_v (Hv=128 for MiMo-V2).
            # We carry both so the store/decode kernels can split.
            self._tq_head_size_v = TritonAttentionDiffKVBackend.head_size_v
        elif is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )'''

if old2 not in content:
    print("FAIL-anchor-2-not-found")
    sys.exit(1)
content = content.replace(old2, new2, 1)

# ---- Edit 3: get_kv_cache_shape branch on TQ ---------------------------
old3 = '''    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + TritonAttentionDiffKVBackend.head_size_v,
        )'''

new3 = '''    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # ''' + MARKER + '''
        # For TurboQuant the per-position-per-head slot is a single
        # combined byte buffer [K_packed | V_packed | padding].  We
        # compute K with head_size (==Hq) and V with head_size_v (==Hv).
        if cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_k = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
            tq_v = TurboQuantConfig.from_cache_dtype(
                cache_dtype_str, TritonAttentionDiffKVBackend.head_size_v
            )
            slot = tq_k.key_packed_size + tq_v.value_packed_size
            slot_aligned = slot + (slot % 2)
            return (num_blocks, block_size, num_kv_heads, slot_aligned)
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + TritonAttentionDiffKVBackend.head_size_v,
        )'''

if old3 not in content:
    print("FAIL-anchor-3-not-found")
    sys.exit(1)
content = content.replace(old3, new3, 1)

# ---- Edit 4: do_kv_cache_update → route to TQ store --------------------
old4 = '''    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        # Cache is packed [..., head_size_qk + head_size_v]; the diffkv
        # reshape kernel writes K to [..., :head_size_qk] and V to
        # [..., head_size_qk:hqk+hv].
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )'''

new4 = '''    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        # ''' + MARKER + '''
        # When TurboQuant is enabled, route through the diffkv-aware TQ
        # store kernel (writes packed [K_fp8 | V_4bit | scale/zero] slot).
        # Otherwise fall back to the standard diffkv reshape + cache.
        if self._tq_config is not None:
            from vllm.v1.attention.ops.triton_turboquant_diffkv_store import (
                triton_turboquant_diffkv_store,
            )

            triton_turboquant_diffkv_store(
                key=key,
                value=value,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                tq_config_k=self._tq_config,
                head_size_v=self._tq_head_size_v,
            )
            return
        # Cache is packed [..., head_size_qk + head_size_v]; the diffkv
        # reshape kernel writes K to [..., :head_size_qk] and V to
        # [..., head_size_qk:hqk+hv].
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )'''

if old4 not in content:
    print("FAIL-anchor-4-not-found")
    sys.exit(1)
content = content.replace(old4, new4, 1)

# ---- Edit 5: forward() → route to TQ decode -----------------------------
old5 = '''        # Slice the packed cache into K / V views.  Strides on dims 0/1/2
        # match the original cache; dim 3 stays contiguous (stride 1).
        key_cache = kv_cache[..., :head_size_qk]
        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        unified_attention_diffkv('''

new5 = '''        # ''' + MARKER + '''
        # When TurboQuant is enabled, route to the TQ-aware decode kernel
        # (it reads packed [K_fp8 | V_4bit] slots directly — no view slice).
        if self._tq_config is not None:
            from vllm.v1.attention.ops.triton_turboquant_diffkv_decode import (
                triton_turboquant_diffkv_decode,
            )

            triton_turboquant_diffkv_decode(
                query=query[:num_actual_tokens],
                kv_cache=kv_cache,
                output=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                seqused_k=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                max_seqlen_q=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                head_size_q=head_size_qk,
                head_size_v=head_size_v,
                tq_config_k=self._tq_config,
            )
            return output

        # Slice the packed cache into K / V views.  Strides on dims 0/1/2
        # match the original cache; dim 3 stays contiguous (stride 1).
        key_cache = kv_cache[..., :head_size_qk]
        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        unified_attention_diffkv('''

if old5 not in content:
    print("FAIL-anchor-5-not-found")
    sys.exit(1)
content = content.replace(old5, new5, 1)

with open(path, "w") as f:
    f.write(content)
print("PATCHED")
