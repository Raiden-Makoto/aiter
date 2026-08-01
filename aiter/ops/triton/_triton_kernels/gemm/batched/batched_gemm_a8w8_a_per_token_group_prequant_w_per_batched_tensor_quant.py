# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr = make_kernel_repr(
    "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "EVEN_MN",
        "EMIT_PTPC",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr,
    do_not_specialize=["M"],
)
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    c_ptpc_ptr,
    c_scale_ptr,
    row_amax_ptr,
    row_counter_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    B: tl.constexpr,
    M,
    N: tl.constexpr,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_cm,
    stride_in_cn,
    stride_ptpc_m,
    stride_ptpc_n,
    stride_scale_m,
    stride_in_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    EMIT_PTPC: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_biasb = tl.cast(stride_in_biasb, tl.int64)

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = tl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    batch_id = tl.cast(batch_id, tl.int64)
    pid_m = tl.cast(pid_m, tl.int64)
    pid_n = tl.cast(pid_n, tl.int64)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(batch_id >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if EVEN_MN:
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
    b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
        a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
        a_scale_recip = 1.0 / a_scale
        a = tl.clamp(a * a_scale_recip, DTYPE_MIN, DTYPE_MAX).to(b_ptr.dtype.element_ty)

        accumulator += tl.dot(a, b) * a_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator *= b_scale

    if HAS_BIAS:
        if EVEN_MN:
            offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    if EVEN_MN:
        if EMIT_PTPC:
            tl.store(c_ptrs, c, cache_modifier=".wt")
        else:
            tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if EMIT_PTPC:
            tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")
        else:
            tl.store(c_ptrs, c, mask=c_mask)

    if EMIT_PTPC:
        # The logical PTPC row spans every batch/head and N tile. Each GEMM
        # program contributes its BF16-rounded local amax, then the final
        # program for a row quantizes the complete contiguous (B*N) scratch row.
        valid_m = offs_cm < M
        valid_n = offs_cn < N
        c_for_amax = tl.where(valid_n[None, :], tl.abs(c.to(tl.float32)), 0.0)
        tile_amax = tl.max(c_for_amax, axis=1)
        tl.atomic_max(
            row_amax_ptr + offs_cm,
            tile_amax,
            mask=valid_m,
            sem="relaxed",
            scope="gpu",
        )
        _ = tl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0",
            constraints="=s",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

        # PTPC mode is restricted to a residency-safe low-M grid. All programs
        # can therefore rendezvous without starving unscheduled producers.
        complete_ptr = row_counter_ptr + pid_m
        ready_ptr = row_counter_ptr + M + pid_m
        quant_done_ptr = row_counter_ptr + 2 * M + pid_m
        finished = tl.atomic_add(
            complete_ptr,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        expected_tiles = B * num_pid_n
        if finished == expected_tiles - 1:
            row_amax = tl.load(row_amax_ptr + offs_cm, mask=valid_m, other=1.0)
            row_scale = tl.maximum(row_amax, 1.0e-10) / DTYPE_MAX
            tl.store(
                c_scale_ptr + offs_cm * stride_scale_m,
                row_scale,
                mask=valid_m,
            )
            _ = tl.inline_asm_elementwise(
                asm="s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0",
                constraints="=s",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            tl.atomic_add(ready_ptr, 1, sem="release", scope="gpu")
        else:
            while tl.atomic_add(
                ready_ptr, 0, sem="acquire", scope="gpu"
            ) == 0:
                _ = tl.inline_asm_elementwise(
                    asm="s_sleep 1\ns_mov_b32 $0, 0",
                    constraints="=s",
                    args=[],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )

        row_scale = tl.load(c_scale_ptr + offs_cm * stride_scale_m, mask=valid_m)
        flat_n = batch_id * N + offs_cn
        q_mask = valid_m[:, None] & valid_n[None, :]
        quant_tile = tl.clamp(
            c.to(tl.float32) / row_scale[:, None],
            DTYPE_MIN,
            DTYPE_MAX,
        ).to(c_ptpc_ptr.dtype.element_ty)
        tl.store(
            c_ptpc_ptr
            + offs_cm[:, None] * stride_ptpc_m
            + flat_n[None, :] * stride_ptpc_n,
            quant_tile,
            mask=q_mask,
            cache_modifier=".cs",
        )
        _ = tl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0",
            constraints="=s",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        quant_done = tl.atomic_add(
            quant_done_ptr,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if quant_done == expected_tiles - 1:
            # The caller initializes these buffers once. Self-reset makes them
            # safe for eager reuse and graph replay without captured memsets.
            tl.store(row_amax_ptr + offs_cm, 0.0, mask=valid_m)
            tl.store(complete_ptr, 0)
            tl.store(ready_ptr, 0)
            tl.store(quant_done_ptr, 0)


def _get_config(
    M: int,
    N: int,
    K: int,
):

    return get_gemm_config(
        "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT",
        M,
        N,
        K,
    )


def _get_ptpc_config(
    M: int,
    N: int,
    K: int,
):
    if M > 64 or (N, K) != (128, 512):
        return _get_config(M, N, K)
    config, is_tuned = get_gemm_config(
        "BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-PTPC_OUTPUT",
        M,
        N,
        K,
    )
    return (config, True) if is_tuned else _get_config(M, N, K)
