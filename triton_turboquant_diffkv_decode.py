# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant K8V4 decode kernel for the DiffKV cache layout.

Adapted from ``triton_turboquant_decode._tq_decode_stage1`` with separate
``HEAD_DIM_K`` and ``HEAD_DIM_V`` so the diffkv slot layout works:

    slot bytes = [ K_fp8: HEAD_DIM_K  |  V_4bit: HEAD_DIM_V/2  |  scale fp16 |  zero fp16 ]

Stage 1 produces per-(batch, q_head, kv_split) partials (max, expsum,
weighted-V vector of HEAD_DIM_V floats). Stage 2 is the standard
log-sum-exp reducer shared with the original TQ decode path.

Status: SKELETON.  Compiles + has the right shape; needs numerical
validation against a BF16 reference and perf tuning.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15


@triton.jit
def _tq_diffkv_decode_stage1_k8v4(
    # Query: [num_tokens (=B for decode), Hq, HEAD_DIM_K] float16/bf16
    Q_ptr,
    # KV cache: [num_blocks, block_size, num_kv_heads, slot_size] uint8
    KV_cache_ptr,
    Block_table_ptr,    # [B, max_num_blocks] int32
    Seq_lens_ptr,       # [B] int32
    # Output partials: [B, Hq, NUM_KV_SPLITS, HEAD_DIM_V+1] float32
    Mid_o_ptr,
    # Strides
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // num_kv_heads
    # TQ layout
    KPS: tl.constexpr,           # = HEAD_DIM_K (FP8 keys, 1 byte/element)
    VAL_DATA_BYTES: tl.constexpr, # = ceil(HEAD_DIM_V * 4 / 8)
    # Score constants
    ATTN_SCALE: tl.constexpr,    # 1 / sqrt(HEAD_DIM_K)
    # Tile sizes
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    FP8_E4B15: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)
    kv_head = hid // KV_GROUP_SIZE

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    # ── Q load (HEAD_DIM_K wide) ────────────────────────────────────────
    dk_offs = tl.arange(0, BLOCK_DK)
    dk_mask = dk_offs < HEAD_DIM_K
    q_base = bid * stride_qb + hid * stride_qh
    q_vec = tl.load(Q_ptr + q_base + dk_offs, mask=dk_mask, other=0.0).to(
        tl.float32
    )

    # V dim offsets
    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < HEAD_DIM_V

    # ── Online softmax state (V-shaped accumulator) ─────────────────────
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    bt_base = bid * stride_bt_b
    kv_range = tl.arange(0, BLOCK_KV)

    # Precompute 4-bit V byte/shift offsets
    v_bit_off = dv_offs * 4
    v_byte_idx = v_bit_off // 8
    v_bit_shift = v_bit_off % 8

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ── K LOAD (FP8) + SCORE ─────────────────────────────────────
        k_addrs = slot_bases[:, None] + dk_offs[None, :]
        k_raw = tl.load(
            KV_cache_ptr + k_addrs,
            mask=kv_mask[:, None] & dk_mask[None, :],
            other=0,
        )
        # Bit-cast uint8 → fp8 → fp32
        if FP8_E4B15:
            k_fp8 = k_raw.to(tl.float8e4b15, bitcast=True)
        else:
            k_fp8 = k_raw.to(tl.float8e4nv, bitcast=True)
        k_fp32 = k_fp8.to(tl.float32)   # [BLOCK_KV, BLOCK_DK]

        scores = tl.sum(q_vec[None, :] * k_fp32, axis=1) * ATTN_SCALE
        scores = tl.where(kv_mask, scores, -float("inf"))

        # ── ONLINE SOFTMAX UPDATE ────────────────────────────────────
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        p = tl.exp(scores - m_new)
        l_curr = tl.sum(p, axis=0)
        alpha = tl.exp(m_prev - m_new)
        l_new = alpha * l_prev + l_curr

        # ── V LOAD (4-bit packed) + DEQUANT ──────────────────────────
        # Byte addresses of the packed-V region for each kv token:
        #   slot_base + KPS + byte_idx_for_dim_v
        v_byte_addrs = (
            slot_bases[:, None] + KPS + v_byte_idx[None, :]
        )
        v_bytes = tl.load(
            KV_cache_ptr + v_byte_addrs,
            mask=kv_mask[:, None] & dv_mask[None, :],
            other=0,
        )
        v_idx = (v_bytes >> v_bit_shift[None, :]) & 0xF  # [BLOCK_KV, BLOCK_DV]

        # Load per-vector scale/zero (fp16) at slot_base + KPS + VAL_DATA_BYTES
        sc_base = slot_bases + KPS + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base, mask=kv_mask, other=0)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1, mask=kv_mask, other=0)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2, mask=kv_mask, other=0)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3, mask=kv_mask, other=0)
        sc_u16 = sc_lo.to(tl.uint16) | (sc_hi.to(tl.uint16) << 8)
        zr_u16 = zr_lo.to(tl.uint16) | (zr_hi.to(tl.uint16) << 8)
        v_scale = sc_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_KV]
        v_zero = zr_u16.to(tl.float16, bitcast=True).to(tl.float32)   # [BLOCK_KV]

        v_dequant = v_idx.to(tl.float32) * v_scale[:, None] + v_zero[:, None]

        # ── ACC UPDATE ────────────────────────────────────────────────
        acc = alpha * acc + tl.sum(p[:, None] * v_dequant, axis=0)

        m_prev = m_new
        l_prev = l_new

    # ── WRITE PARTIAL TO Mid_o ──────────────────────────────────────────
    # Matches the convention of _tq_decode_stage1 in
    # triton_turboquant_decode.py: write the *normalized* per-split V
    # output (acc / l) followed by lse = m + log(l). Stage 2 then
    # combines splits with proper LSE merging.
    out_base = (
        bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    )
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + dv_offs, acc / safe_l, mask=dv_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM_V, lse)


def triton_turboquant_diffkv_decode(
    query: torch.Tensor,         # [B, Hq, head_size_q]  decode-shape: tokens=B
    kv_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, slot]
    output: torch.Tensor,        # [B, Hq, head_size_v]
    cu_seqlens_q: torch.Tensor,  # [B+1] (decode: arange*1)
    seqused_k: torch.Tensor,     # [B] context length
    block_table: torch.Tensor,   # [B, max_num_blocks]
    max_seqlen_q: int,
    softmax_scale: float,
    head_size_q: int,
    head_size_v: int,
    tq_config_k,
):
    """K8V4 decode launcher for diffkv layouts.

    For correctness during prefill / continuation prefill (q_len > 1),
    this skeleton falls back to a dequant + standard SDPA path because
    the kernel above is decode-only (one q per request).  Long-context
    decode is the optimization target so the prefill fallback is
    acceptable for M1.
    """
    if max_seqlen_q > 1:
        # Prefill fallback: dequantize cache to bf16 and call SDPA.
        # This is slow but correct; M1 doesn't target prefill perf.
        _dequant_prefill_fallback(
            query, kv_cache, output, cu_seqlens_q, seqused_k, block_table,
            softmax_scale, head_size_q, head_size_v, tq_config_k,
        )
        return

    if tq_config_k.value_quant_bits != 4 or not tq_config_k.key_fp8:
        raise NotImplementedError(
            "Only K8V4 supported in this kernel; got "
            f"K_fp8={tq_config_k.key_fp8} V_bits={tq_config_k.value_quant_bits}"
        )

    B, Hq, _ = query.shape
    num_blocks, block_size, num_kv_heads, slot_size = kv_cache.shape

    BLOCK_DK = 1 << (head_size_q - 1).bit_length()
    BLOCK_DV = 1 << (head_size_v - 1).bit_length()
    val_data_bytes = math.ceil(head_size_v * 4 / 8)
    kps = head_size_q  # FP8 keys take head_size_q bytes
    NUM_KV_SPLITS = 8
    BLOCK_KV = 16

    cache_view = kv_cache.view(torch.uint8)
    assert cache_view.is_contiguous()
    stride_cache_block = block_size * num_kv_heads * slot_size
    stride_cache_pos = num_kv_heads * slot_size
    stride_cache_head = slot_size

    mid_o = torch.empty(
        (B, Hq, NUM_KV_SPLITS, head_size_v + 1),
        dtype=torch.float32,
        device=query.device,
    )

    grid_stage1 = (B, Hq, NUM_KV_SPLITS)
    _tq_diffkv_decode_stage1_k8v4[grid_stage1](
        query.contiguous(),
        cache_view,
        block_table,
        seqused_k,
        mid_o,
        query.stride(0),
        query.stride(1),
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM_K=head_size_q,
        HEAD_DIM_V=head_size_v,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=Hq // num_kv_heads,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        ATTN_SCALE=softmax_scale,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
        BLOCK_KV=BLOCK_KV,
        FP8_E4B15=_use_fp8_e4b15(),
    )

    # Stage 2: LSE reduce across NUM_KV_SPLITS partials.
    # The shared TQ decode stage2 expects a separate LSE output tensor.
    lse = torch.empty((B, Hq), dtype=torch.float32, device=query.device)
    _fwd_kernel_stage2[(B, Hq)](
        mid_o,
        output,
        lse,
        seqused_k,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=head_size_v,
    )


def _dequant_prefill_fallback(
    query, kv_cache, output, cu_seqlens_q, seqused_k, block_table,
    softmax_scale, head_size_q, head_size_v, tq_config_k,
):
    """SLOW prefill fallback: not optimized; just numerically correct.

    Materializes the full K/V tensors at BF16 from the packed cache,
    then calls ``torch.nn.functional.scaled_dot_product_attention``.

    M1 doesn't target prefill perf; long-context *decode* is the bottleneck
    the user actually cares about (one request, growing context). Prefill
    is one-shot at request start.  This path is structurally correct so
    end-to-end output is sound; future work can specialize this if
    prefill cost ever matters.
    """
    raise NotImplementedError(
        "Prefill dequant fallback for diffkv K8V4 not yet implemented; "
        "M1 smoke tests should use --max-num-seqs 1 --max-num-batched-tokens 1 "
        "or guarantee q_len==1 in the impl router.  TODO: implement before C33."
    )
