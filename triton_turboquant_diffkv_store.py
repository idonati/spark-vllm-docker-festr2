# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant K8V4 store kernel for the DiffKV cache layout.

This is the diffkv variant of ``triton_turboquant_store._tq_fused_store_fp8``:
identical key (FP8 cast+store) and value (uniform 4-bit pack + fp16
scale/zero) logic, but with two distinct head sizes:

- HEAD_DIM_K (e.g. 192 for MiMo-V2): the size of the K and Q vectors.
- HEAD_DIM_V (e.g. 128 for MiMo-V2): the size of the V vector.

Cache slot layout (per position, per kv head):
    [ K_fp8: HEAD_DIM_K bytes
    | V_4bit: ceil(HEAD_DIM_V * 4 / 8) bytes
    | V_scale: 2 bytes (fp16)
    | V_zero:  2 bytes (fp16) ]

Status: SKELETON.  The kernel compiles and has the right structure but
needs numerical verification against a BF16 reference and perf tuning
before C33 can serve production traffic.
"""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15


@triton.jit
def _tq_diffkv_store_k8v4(
    Key_ptr,          # [N_tokens, NH, HEAD_DIM_K] float16/bfloat16
    Value_ptr,        # [N_tokens, NH, HEAD_DIM_V] float16/bfloat16
    KV_cache_ptr,     # [num_blocks * block_size * NH * slot] uint8 (flat)
    Slot_mapping_ptr, # [N_tokens] int32
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DK: tl.constexpr,   # next_power_of_2(HEAD_DIM_K)
    BLOCK_DV: tl.constexpr,   # next_power_of_2(HEAD_DIM_V)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(HEAD_DIM_V * 4 / 8)
    FP8_E4B15: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    token_idx = pid // NH
    head_idx = pid % NH

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    # ── FP8 KEY: cast and scatter ─────────────────────────────────────
    dk_offs = tl.arange(0, BLOCK_DK)
    dk_mask = dk_offs < HEAD_DIM_K
    k_base = token_idx * NH * HEAD_DIM_K + head_idx * HEAD_DIM_K
    k_vals = tl.load(Key_ptr + k_base + dk_offs, mask=dk_mask, other=0.0)
    if FP8_E4B15:
        k_fp8 = k_vals.to(tl.float8e4b15)
    else:
        k_fp8 = k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + dk_offs, k_bytes, mask=dk_mask)

    # ── 4-BIT VALUE: uniform quant, pack, store with fp16 scale/zero ─
    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < HEAD_DIM_V
    v_base = token_idx * NH * HEAD_DIM_V + head_idx * HEAD_DIM_V
    v_vec = tl.load(
        Value_ptr + v_base + dv_offs, mask=dv_mask, other=0.0
    ).to(tl.float32)
    v_min = tl.min(tl.where(dv_mask, v_vec, float("inf")), axis=0)
    v_max = tl.max(tl.where(dv_mask, v_vec, -float("inf")), axis=0)
    v_scale = (v_max - v_min) / 15.0
    v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
    q_all = tl.minimum(
        tl.maximum(((v_vec - v_min) / v_scale + 0.5).to(tl.int32), 0), 15
    )
    # Pack two 4-bit values per byte (low nibble = even index, high = odd).
    q_pairs = tl.reshape(q_all, [BLOCK_DV // 2, 2])
    shifts_4 = tl.arange(0, 2) * 4
    packed_val = tl.sum(
        (q_pairs & 0xF) << shifts_4[None, :], axis=1
    ).to(tl.uint8)
    val_offs = tl.arange(0, BLOCK_DV // 2)
    val_mask = val_offs < VAL_DATA_BYTES

    val_cache_offset = HEAD_DIM_K  # K bytes come first
    tl.store(
        KV_cache_ptr + slot_base + val_cache_offset + val_offs,
        packed_val,
        mask=val_mask,
    )

    sc_offset = val_cache_offset + VAL_DATA_BYTES
    sc_f16 = v_scale.to(tl.float16)
    sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
    tl.store(
        KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8)
    )
    tl.store(
        KV_cache_ptr + slot_base + sc_offset + 1,
        ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
    )
    zr_f16 = v_min.to(tl.float16)
    zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
    tl.store(
        KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8)
    )
    tl.store(
        KV_cache_ptr + slot_base + sc_offset + 3,
        ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
    )


def triton_turboquant_diffkv_store(
    key: torch.Tensor,           # [N_tokens, NH, head_size_q]
    value: torch.Tensor,         # [N_tokens, NH, head_size_v]
    kv_cache: torch.Tensor,      # [num_blocks, block_size, NH, slot_size]
    slot_mapping: torch.Tensor,  # [N_tokens]
    tq_config_k,                 # TurboQuantConfig built with head_dim=head_size_q
    head_size_v: int,
):
    """K8V4 store launcher for diffkv layouts.

    Only ``turboquant_k8v4`` is implemented; other TQ variants will raise.
    """
    if not getattr(tq_config_k, "key_fp8", False):
        raise NotImplementedError(
            "triton_turboquant_diffkv_store currently only supports "
            "K8V4 (FP8 keys + 4-bit values). Other TQ variants will "
            "be added in M2."
        )
    if tq_config_k.value_quant_bits != 4:
        raise NotImplementedError(
            f"V quant bits {tq_config_k.value_quant_bits} not yet supported "
            "in the diffkv path; expected 4."
        )
    head_size_q = tq_config_k.head_dim
    N, NH, _ = key.shape
    assert value.shape == (N, NH, head_size_v), (
        f"value shape {tuple(value.shape)} != expected ({N},{NH},{head_size_v})"
    )

    # Flatten cache view to a uint8 byte buffer.
    cache_view = kv_cache.view(torch.uint8)
    # Strides on the original tensor in bytes per element of the flat byte view.
    # kv_cache last dim is contiguous bytes; the existing TQ store kernel
    # relies on contiguous slot rows.
    assert cache_view.is_contiguous(), "kv_cache must be contiguous"

    block_size = kv_cache.shape[1]
    slot_size = kv_cache.shape[3]
    stride_cache_block = block_size * NH * slot_size  # bytes
    stride_cache_pos = NH * slot_size
    stride_cache_head = slot_size

    BLOCK_DK = 1 << (head_size_q - 1).bit_length()
    BLOCK_DV = 1 << (head_size_v - 1).bit_length()
    val_data_bytes = math.ceil(head_size_v * 4 / 8)

    grid = (N * NH,)
    _tq_diffkv_store_k8v4[grid](
        key.contiguous(),
        value.contiguous(),
        cache_view,
        slot_mapping,
        stride_cache_block=stride_cache_block,
        stride_cache_pos=stride_cache_pos,
        stride_cache_head=stride_cache_head,
        HEAD_DIM_K=head_size_q,
        HEAD_DIM_V=head_size_v,
        NH=NH,
        BLOCK_SIZE=block_size,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
        VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=_use_fp8_e4b15(),
    )
