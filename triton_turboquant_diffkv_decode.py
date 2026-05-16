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
    mid_o_scratch: torch.Tensor | None = None,  # pre-allocated [B_max, Hq, NUM_KV_SPLITS, head_size_v+1]
    lse_scratch: torch.Tensor | None = None,    # pre-allocated [B_max, Hq]
):
    """K8V4 decode launcher for diffkv layouts.

    For correctness during prefill / continuation prefill (q_len > 1),
    this skeleton falls back to a dequant + standard SDPA path because
    the kernel above is decode-only (one q per request).  Long-context
    decode is the optimization target so the prefill fallback is
    acceptable for M1.

    For CUDA-graph compatibility on the decode path, callers should pass
    persistent pre-allocated ``mid_o_scratch`` and ``lse_scratch`` buffers
    (sized at max batch).  When omitted, fresh buffers are allocated per
    call (graph-unfriendly but functional — used by the prefill fallback
    which runs eager anyway, and by unit tests).
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

    if mid_o_scratch is not None:
        # Reuse pre-allocated scratch.  Slice the leading dim down to B; the
        # kernel writes only to the active rows.  Stride 0 is preserved on
        # the slice, so the kernel's stride-based indexing is unchanged.
        assert mid_o_scratch.shape[0] >= B, (
            f"mid_o_scratch dim 0 ({mid_o_scratch.shape[0]}) < B ({B})"
        )
        assert mid_o_scratch.shape[1:] == (Hq, NUM_KV_SPLITS, head_size_v + 1), (
            f"mid_o_scratch trailing dims {mid_o_scratch.shape[1:]} != "
            f"({Hq}, {NUM_KV_SPLITS}, {head_size_v + 1})"
        )
        mid_o = mid_o_scratch[:B]
    else:
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
    if lse_scratch is not None:
        assert lse_scratch.shape[0] >= B and lse_scratch.shape[1] == Hq, (
            f"lse_scratch shape {lse_scratch.shape} incompatible with (B={B}, Hq={Hq})"
        )
        lse = lse_scratch[:B]
    else:
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


@triton.jit
def _tq_diffkv_dequant_k8v4_linear(
    # Compressed cache + index tensors.
    KV_cache_ptr,          # uint8 view of [num_blocks, block_size, NH_KV, slot]
    Block_table_ptr,       # int32 [B, max_blocks]
    Seq_lens_ptr,          # int32 [B]
    # Linear output buffers (one entry per (request, pos, kv_head)).
    K_out_ptr,             # bf16 [B, MAX_SEQ, NH_KV, HEAD_DIM_K]
    V_out_ptr,             # bf16 [B, MAX_SEQ, NH_KV, HEAD_DIM_V]
    # Strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_bt_b,
    stride_kob,
    stride_kos,
    stride_koh,
    stride_vob,
    stride_vos,
    stride_voh,
    # Layout constants
    NH_KV: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KPS: tl.constexpr,          # = HEAD_DIM_K (FP8 keys)
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    FP8_E4B15: tl.constexpr = 0,
):
    """Walk the compressed cache slots for a request and write BF16 K/V.

    One program per (batch, kv_head, position). For positions past the
    request's seq_len, do nothing.
    """
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    pos = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    if pos >= seq_len:
        return

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)

    slot_base = (
        block_num * stride_cache_block
        + page_off.to(tl.int64) * stride_cache_pos
        + tl.cast(kv_head, tl.int64) * stride_cache_head
    )

    # ── K dequant ────────────────────────────────────────────────────
    dk_offs = tl.arange(0, BLOCK_DK)
    dk_mask = dk_offs < HEAD_DIM_K
    k_raw = tl.load(KV_cache_ptr + slot_base + dk_offs, mask=dk_mask, other=0)
    if FP8_E4B15:
        k_fp8 = k_raw.to(tl.float8e4b15, bitcast=True)
    else:
        k_fp8 = k_raw.to(tl.float8e4nv, bitcast=True)
    k_bf = k_fp8.to(tl.bfloat16)

    k_out_base = (
        bid * stride_kob + pos * stride_kos + kv_head * stride_koh
    )
    tl.store(K_out_ptr + k_out_base + dk_offs, k_bf, mask=dk_mask)

    # ── V dequant ────────────────────────────────────────────────────
    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < HEAD_DIM_V
    v_bit_off = dv_offs * 4
    v_byte_idx = v_bit_off // 8
    v_bit_shift = v_bit_off % 8
    v_bytes = tl.load(
        KV_cache_ptr + slot_base + KPS + v_byte_idx,
        mask=dv_mask, other=0,
    )
    v_idx = (v_bytes >> v_bit_shift) & 0xF

    # Per-vector scale (fp16) and zero (fp16): 4 bytes after VAL_DATA_BYTES
    sc_base = slot_base + KPS + VAL_DATA_BYTES
    sc_lo = tl.load(KV_cache_ptr + sc_base)
    sc_hi = tl.load(KV_cache_ptr + sc_base + 1)
    zr_lo = tl.load(KV_cache_ptr + sc_base + 2)
    zr_hi = tl.load(KV_cache_ptr + sc_base + 3)
    sc_u16 = sc_lo.to(tl.uint16) | (sc_hi.to(tl.uint16) << 8)
    zr_u16 = zr_lo.to(tl.uint16) | (zr_hi.to(tl.uint16) << 8)
    v_scale = sc_u16.to(tl.float16, bitcast=True).to(tl.float32)
    v_zero = zr_u16.to(tl.float16, bitcast=True).to(tl.float32)
    v_dequant = (v_idx.to(tl.float32) * v_scale + v_zero).to(tl.bfloat16)

    v_out_base = (
        bid * stride_vob + pos * stride_vos + kv_head * stride_voh
    )
    tl.store(V_out_ptr + v_out_base + dv_offs, v_dequant, mask=dv_mask)


def _dequant_prefill_fallback(
    query: torch.Tensor,         # [total_q, NH_Q, head_size_q]  bf16/fp16
    kv_cache: torch.Tensor,      # [num_blocks, block_size, NH_KV, slot]
    output: torch.Tensor,        # [total_q, NH_Q, head_size_v]  bf16/fp16
    cu_seqlens_q: torch.Tensor,  # [B+1]
    seqused_k: torch.Tensor,     # [B]
    block_table: torch.Tensor,   # [B, max_num_blocks]
    softmax_scale: float,
    head_size_q: int,
    head_size_v: int,
    tq_config_k,
):
    """Correct-but-slow prefill / multi-q-token fallback.

    Strategy:
    1. Bulk-dequant the compressed cache slots referenced by
       ``block_table`` into linear BF16 K/V tensors (one Triton kernel).
    2. For each request, run causal attention via PyTorch ops (einsum +
       softmax) with the linear K/V.

    Memory: ``B * max_seq * NH_KV * (Hq + Hv) * 2`` bytes per layer per
    call. For B=4, max_seq=32k, NH_KV=1, Hq+Hv=320 → 80 MiB. Allocated
    and freed per layer; PyTorch's caching allocator should handle this
    cheaply.

    Numerically: produces the same result as running ``unified_attention_diffkv``
    on a BF16-materialized cache. M3 will validate this end-to-end.
    """
    import torch.nn.functional as F

    if tq_config_k.value_quant_bits != 4 or not tq_config_k.key_fp8:
        raise NotImplementedError(
            "Only K8V4 prefill fallback implemented; got "
            f"K_fp8={tq_config_k.key_fp8} V_bits={tq_config_k.value_quant_bits}"
        )

    total_q, NH_Q, _ = query.shape
    num_blocks, block_size, NH_KV, slot_size = kv_cache.shape
    B = seqused_k.numel()
    max_seq = int(seqused_k.max().item())
    kv_group = NH_Q // NH_KV
    val_data_bytes = math.ceil(head_size_v * 4 / 8)
    kps = head_size_q

    cache_view = kv_cache.view(torch.uint8)
    assert cache_view.is_contiguous()
    stride_cache_block = block_size * NH_KV * slot_size
    stride_cache_pos = NH_KV * slot_size
    stride_cache_head = slot_size

    K_lin = torch.empty(
        (B, max_seq, NH_KV, head_size_q),
        dtype=query.dtype, device=query.device,
    )
    V_lin = torch.empty(
        (B, max_seq, NH_KV, head_size_v),
        dtype=query.dtype, device=query.device,
    )

    BLOCK_DK = 1 << (head_size_q - 1).bit_length()
    BLOCK_DV = 1 << (head_size_v - 1).bit_length()

    grid = (B, NH_KV, max_seq)
    _tq_diffkv_dequant_k8v4_linear[grid](
        cache_view,
        block_table,
        seqused_k,
        K_lin,
        V_lin,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        block_table.stride(0),
        K_lin.stride(0),
        K_lin.stride(1),
        K_lin.stride(2),
        V_lin.stride(0),
        V_lin.stride(1),
        V_lin.stride(2),
        NH_KV=NH_KV,
        HEAD_DIM_K=head_size_q,
        HEAD_DIM_V=head_size_v,
        BLOCK_SIZE=block_size,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
        FP8_E4B15=_use_fp8_e4b15(),
    )

    # Per-request causal attention. The varlen / B>1 case is handled
    # with a Python loop; for B=1 (single-user serve) this is a single
    # iteration.  TODO(M2 or later): replace with a single batched call
    # if profiler shows this is hot.
    cu = cu_seqlens_q.tolist()
    for b in range(B):
        q_start = cu[b]
        q_end = cu[b + 1]
        q_len = q_end - q_start
        s_len = int(seqused_k[b].item())
        prefix_len = s_len - q_len

        if q_len == 0:
            continue

        q_chunk = query[q_start:q_end]            # [q_len, NH_Q, Hq]
        k_full = K_lin[b, :s_len]                  # [s_len, NH_KV, Hq]
        v_full = V_lin[b, :s_len]                  # [s_len, NH_KV, Hv]

        # GQA expansion: NH_KV → NH_Q via repeat_interleave on the head dim.
        if kv_group > 1:
            k_full = k_full.repeat_interleave(kv_group, dim=1)
            v_full = v_full.repeat_interleave(kv_group, dim=1)
        # Now shapes: k_full [s_len, NH_Q, Hq], v_full [s_len, NH_Q, Hv]

        # Permute for batched matmul: [NH_Q, q_len, Hq] x [NH_Q, Hq, s_len]
        q_perm = q_chunk.permute(1, 0, 2).float()
        k_perm = k_full.permute(1, 0, 2).float()
        v_perm = v_full.permute(1, 0, 2).float()

        scores = torch.einsum("hqd,hsd->hqs", q_perm, k_perm) * softmax_scale

        # Causal mask: q token i (within chunk) attends to positions
        # [0, prefix_len + i + 1).  Build an [q_len, s_len] bool mask.
        q_pos = torch.arange(prefix_len, prefix_len + q_len, device=q_chunk.device)
        kv_pos = torch.arange(s_len, device=q_chunk.device)
        mask = kv_pos[None, :] <= q_pos[:, None]   # [q_len, s_len]
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out_chunk = torch.einsum("hqs,hsd->hqd", weights, v_perm)  # [NH_Q, q_len, Hv]
        out_chunk = out_chunk.permute(1, 0, 2).to(output.dtype)
        output[q_start:q_end] = out_chunk
