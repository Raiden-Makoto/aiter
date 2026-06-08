# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import math
from typing import Optional, Dict
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton._gluon_kernels.utils.prefetch import (
    gemm_l2_prefetch,
    gemm_l2_prefetch_prologue,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_LOGGER = AiterTritonLogger()

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "NUM_BUFFERS",
    "PHYSICAL_MK",
    "PHYSICAL_KN",
    "USE_ACTIVATION",
    "ADD_BIAS",
    "L2_PREFETCH_DISTANCE",
]

_gemm_a16w16_basic_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_basic_kernel", _GLUON_REPR_KEYS
)

_gemm_a16w16_warp_priority_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_warp_priority_kernel", _GLUON_REPR_KEYS
)

_gemm_a16w16_k_subtiling_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_k_subtiling_kernel", _GLUON_REPR_KEYS
)

_gemm_a16w16_lds_pipeline_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_lds_pipeline_kernel", _GLUON_REPR_KEYS
)


def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return config, is_tuned


def create_shared_layouts(
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    if PHYSICAL_MK:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
        )
    else:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_M, 8]], [BLOCK_K, BLOCK_M], [1, 0]
        )

    if PHYSICAL_KN:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit
def _make_biased_descriptors(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    # Bake this block's (M, N) offset into the base pointer so the K loop only
    # varies the K position (either by advancing the descriptor or by passing an
    # absolute K offset to async_load).
    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_base,
            shape=(M - pid_m * BLOCK_M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_base,
            shape=(K, M - pid_m * BLOCK_M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(K, N - pid_n * BLOCK_N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(N - pid_n * BLOCK_N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    return a_desc, b_desc


@gluon.jit
def _alloc_ab_buffers(
    a_ptr,
    b_ptr,
    NUM_BUFFERS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    return a_buffer, b_buffer


@gluon.jit
def _async_load_ab(a_desc, b_desc, a_buffer, b_buffer, idx, NUM_BUFFERS: gl.constexpr):
    """Issue the TDM loads for one K tile into buffer slot ``idx``."""
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_buffer.index(idx % NUM_BUFFERS))
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b_buffer.index(idx % NUM_BUFFERS))


@gluon.jit
def _advance_ab(
    a_desc,
    b_desc,
    BLOCK_K: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    """Walk both descriptors forward one K tile."""
    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[BLOCK_K, 0]
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[BLOCK_K, 0]
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )

    return a_desc, b_desc


@gluon.jit
def _async_load_ab_masked(
    a_desc,
    b_desc,
    a_buffer,
    b_buffer,
    idx,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    """Issue the TDM loads for K tile ``idx`` into buffer slot ``idx % NUM_BUFFERS``.

    Offsets are absolute K positions (``idx * BLOCK_K``) against the fixed
    descriptor bounds set in ``_make_biased_descriptors`` (no descriptor advance).
    Because the descriptors are created with zero padding, any tile whose K
    offset reaches or exceeds K is detected as out-of-bounds by the TDM engine
    and zero-filled. This keeps the result correct even when the pipeline issues
    more tiles than exist along K (``num_k_tiles < NUM_BUFFERS``): the extra
    tiles read as zeros and contribute nothing to the accumulator, so there is
    no need to clamp NUM_BUFFERS on the host.
    """
    off_k = idx * BLOCK_K
    slot = idx % NUM_BUFFERS
    if PHYSICAL_MK:
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, off_k], a_buffer.index(slot))
    else:
        gl.amd.gfx1250.tdm.async_load(a_desc, [off_k, 0], a_buffer.index(slot))

    if PHYSICAL_KN:
        gl.amd.gfx1250.tdm.async_load(b_desc, [off_k, 0], b_buffer.index(slot))
    else:
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, off_k], b_buffer.index(slot))


@gluon.jit
def _read_ab(
    a_buffer,
    b_buffer,
    idx,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
):
    """Read one K tile of A and B from LDS into registers (WMMA operands)."""
    if PHYSICAL_MK:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
    else:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_A
        )

    if PHYSICAL_KN:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(idx % NUM_BUFFERS), OPERAND_LAYOUT_B
        )
    else:
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
        )

    return a, b


@gluon.jit
def _epilogue_buffer_store(
    accumulator,
    c_ptr,
    bias_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """Optional bias + activation, then a masked buffer_store of the tile."""
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]

    if USE_ACTIVATION:
        accumulator = activation(accumulator)

    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty), c_ptr, offs_c, mask=mask_c
    )


@gluon.jit(repr=_gemm_a16w16_basic_repr)
def _gemm_a16w16_basic_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = _make_biased_descriptors(
        a_ptr, b_ptr, pid_m, pid_n, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B,
        PHYSICAL_MK, PHYSICAL_KN,
    )
    a_buffer, b_buffer = _alloc_ab_buffers(
        a_ptr, b_ptr, NUM_BUFFERS, BLOCK_M, BLOCK_N, BLOCK_K,
        SHARED_LAYOUT_A, SHARED_LAYOUT_B, PHYSICAL_MK, PHYSICAL_KN,
    )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # L2 prefetch helpers compute offsets relative to the descriptor's current
    # position. Since the per-block (M, N) offset is baked into the base, we
    # pass 0 for off_am/off_bn/load_idx so off_k is purely a K-direction delta.
    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(
            L2_PREFETCH_DISTANCE,
            0,
            a_desc,
            b_desc,
            0,
            0,
            BLOCK_K,
            NUM_BUFFERS,
            not PHYSICAL_MK,
            not PHYSICAL_KN,
        )

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        _async_load_ab(a_desc, b_desc, a_buffer, b_buffer, load_idx, NUM_BUFFERS)
        a_desc, b_desc = _advance_ab(a_desc, b_desc, BLOCK_K, PHYSICAL_MK, PHYSICAL_KN)
        load_idx += 1

    # Main pipeline loop
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        _async_load_ab(a_desc, b_desc, a_buffer, b_buffer, load_idx, NUM_BUFFERS)
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        a_desc, b_desc = _advance_ab(a_desc, b_desc, BLOCK_K, PHYSICAL_MK, PHYSICAL_KN)
        load_idx += 1

        if USE_L2_PREFETCH:
            gemm_l2_prefetch(
                L2_PREFETCH_DISTANCE - 1,
                0,
                a_desc,
                b_desc,
                0,
                0,
                BLOCK_K,
                not PHYSICAL_MK,
                not PHYSICAL_KN,
            )

        cur_a, cur_b = _read_ab(
            a_buffer, b_buffer, compute_idx,
            OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
            PHYSICAL_MK, PHYSICAL_KN,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Epilogue: no more loads
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        cur_a, cur_b = _read_ab(
            a_buffer, b_buffer, compute_idx,
            OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
            PHYSICAL_MK, PHYSICAL_KN,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    _epilogue_buffer_store(
        accumulator, c_ptr, bias_ptr, pid_m, pid_n, M, N,
        stride_cm, stride_cn, BLOCK_M, BLOCK_N, WMMA_LAYOUT,
        activation, USE_ACTIVATION, ADD_BIAS,
    )


@gluon.jit(repr=_gemm_a16w16_warp_priority_repr)
def _gemm_a16w16_warp_priority_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0
    gl.static_assert(NUM_BUFFERS >= 3, "Warp priority kernel requires NUM_BUFFERS >= 3")

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(
            L2_PREFETCH_DISTANCE,
            load_idx,
            a_desc,
            b_desc,
            off_am,
            off_bn,
            BLOCK_K,
            NUM_BUFFERS,
            not PHYSICAL_MK,
            not PHYSICAL_KN,
        )

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS),
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS),
            )

        load_idx += 1

    # Main pipeline loop with warp pipelining
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

        with gl.amd.warp_pipeline_stage("stage0", priority=1):
            if PHYSICAL_MK:
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
                )
            else:
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                    OPERAND_LAYOUT_A,
                )

            if PHYSICAL_KN:
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
                )
            else:
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                    OPERAND_LAYOUT_B,
                )

            if PHYSICAL_MK:
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [pid_m * BLOCK_M, load_idx * BLOCK_K],
                    a_buffer.index(load_idx % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [load_idx * BLOCK_K, pid_m * BLOCK_M],
                    a_buffer.index(load_idx % NUM_BUFFERS),
                )

            if PHYSICAL_KN:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [load_idx * BLOCK_K, pid_n * BLOCK_N],
                    b_buffer.index(load_idx % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [pid_n * BLOCK_N, load_idx * BLOCK_K],
                    b_buffer.index(load_idx % NUM_BUFFERS),
                )

            load_idx += 1

            if USE_L2_PREFETCH:
                gemm_l2_prefetch(
                    L2_PREFETCH_DISTANCE - 1,
                    load_idx,
                    a_desc,
                    b_desc,
                    off_am,
                    off_bn,
                    BLOCK_K,
                    not PHYSICAL_MK,
                    not PHYSICAL_KN,
                )

        with gl.amd.warp_pipeline_stage("stage1", priority=0):
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            compute_idx += 1

    # Epilogue: drain remaining tiles
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_A,
            )

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_B,
            )

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    _epilogue_buffer_store(
        accumulator, c_ptr, bias_ptr, pid_m, pid_n, M, N,
        stride_cm, stride_cn, BLOCK_M, BLOCK_N, WMMA_LAYOUT,
        activation, USE_ACTIVATION, ADD_BIAS,
    )


@gluon.jit(repr=_gemm_a16w16_k_subtiling_repr)
def _gemm_a16w16_k_subtiling_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0
    SUBTILE_LEN: gl.constexpr = 32
    NUM_SUBTILES: gl.constexpr = BLOCK_K // SUBTILE_LEN
    gl.static_assert(NUM_SUBTILES >= 2, "BLOCK_K must be >= 64 for k-subtiling")

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    producer = 0
    consumer = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(
            L2_PREFETCH_DISTANCE,
            producer,
            a_desc,
            b_desc,
            off_am,
            off_bn,
            BLOCK_K,
            NUM_BUFFERS,
            not PHYSICAL_MK,
            not PHYSICAL_KN,
        )

    # Prologue: fill pipeline with NUM_BUFFERS - 1 tiles
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS),
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS),
            )

        producer += 1

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Main loop: issue TDM load for next tile, then process current tile via subtile loop
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS),
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS),
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS),
            )

        producer += 1

        if USE_L2_PREFETCH:
            gemm_l2_prefetch(
                L2_PREFETCH_DISTANCE - 1,
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                BLOCK_K,
                not PHYSICAL_MK,
                not PHYSICAL_KN,
            )

        # Subtile loop: load subtile s+1 while computing subtile s
        idx = consumer % NUM_BUFFERS
        if PHYSICAL_MK:
            cur_a = (
                a_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 1)
                .load(layout=OPERAND_LAYOUT_A)
            )
        else:
            cur_a = (
                a_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 0)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_A)
            )
        if PHYSICAL_KN:
            cur_b = (
                b_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 0)
                .load(layout=OPERAND_LAYOUT_B)
            )
        else:
            cur_b = (
                b_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 1)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_B)
            )

        for s in gl.static_range(1, NUM_SUBTILES):
            if PHYSICAL_MK:
                next_a = (
                    a_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                    .load(layout=OPERAND_LAYOUT_A)
                )
            else:
                next_a = (
                    a_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                    .permute([1, 0])
                    .load(layout=OPERAND_LAYOUT_A)
                )
            if PHYSICAL_KN:
                next_b = (
                    b_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                    .load(layout=OPERAND_LAYOUT_B)
                )
            else:
                next_b = (
                    b_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                    .permute([1, 0])
                    .load(layout=OPERAND_LAYOUT_B)
                )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a = next_a
            cur_b = next_b
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        consumer += 1

    # Epilogue: drain remaining pipeline stages (no new TDM loads)
    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)

        idx = consumer % NUM_BUFFERS
        if PHYSICAL_MK:
            cur_a = (
                a_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 1)
                .load(layout=OPERAND_LAYOUT_A)
            )
        else:
            cur_a = (
                a_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 0)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_A)
            )
        if PHYSICAL_KN:
            cur_b = (
                b_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 0)
                .load(layout=OPERAND_LAYOUT_B)
            )
        else:
            cur_b = (
                b_buffer.index(idx)
                .slice(0, SUBTILE_LEN, 1)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_B)
            )

        for s in gl.static_range(1, NUM_SUBTILES):
            if PHYSICAL_MK:
                next_a = (
                    a_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                    .load(layout=OPERAND_LAYOUT_A)
                )
            else:
                next_a = (
                    a_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                    .permute([1, 0])
                    .load(layout=OPERAND_LAYOUT_A)
                )
            if PHYSICAL_KN:
                next_b = (
                    b_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                    .load(layout=OPERAND_LAYOUT_B)
                )
            else:
                next_b = (
                    b_buffer.index(idx)
                    .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                    .permute([1, 0])
                    .load(layout=OPERAND_LAYOUT_B)
                )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a = next_a
            cur_b = next_b
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        consumer += 1

    # Final tile
    gl.amd.gfx1250.tdm.async_wait(0)

    idx = consumer % NUM_BUFFERS
    if PHYSICAL_MK:
        cur_a = (
            a_buffer.index(idx).slice(0, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
        )
    else:
        cur_a = (
            a_buffer.index(idx)
            .slice(0, SUBTILE_LEN, 0)
            .permute([1, 0])
            .load(layout=OPERAND_LAYOUT_A)
        )
    if PHYSICAL_KN:
        cur_b = (
            b_buffer.index(idx).slice(0, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
        )
    else:
        cur_b = (
            b_buffer.index(idx)
            .slice(0, SUBTILE_LEN, 1)
            .permute([1, 0])
            .load(layout=OPERAND_LAYOUT_B)
        )

    for s in gl.static_range(1, NUM_SUBTILES):
        if PHYSICAL_MK:
            next_a = (
                a_buffer.index(idx)
                .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                .load(layout=OPERAND_LAYOUT_A)
            )
        else:
            next_a = (
                a_buffer.index(idx)
                .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_A)
            )
        if PHYSICAL_KN:
            next_b = (
                b_buffer.index(idx)
                .slice(s * SUBTILE_LEN, SUBTILE_LEN, 0)
                .load(layout=OPERAND_LAYOUT_B)
            )
        else:
            next_b = (
                b_buffer.index(idx)
                .slice(s * SUBTILE_LEN, SUBTILE_LEN, 1)
                .permute([1, 0])
                .load(layout=OPERAND_LAYOUT_B)
            )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        cur_a = next_a
        cur_b = next_b
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    _epilogue_buffer_store(
        accumulator, c_ptr, bias_ptr, pid_m, pid_n, M, N,
        stride_cm, stride_cn, BLOCK_M, BLOCK_N, WMMA_LAYOUT,
        activation, USE_ACTIVATION, ADD_BIAS,
    )


@gluon.jit(repr=_gemm_a16w16_lds_pipeline_repr)
def _gemm_a16w16_lds_pipeline_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    """Local-load pipelining across K-tiles.

    Manually places load_shared_relaxed for tile i+1 *before* the wmma for
    tile i so the hardware LDS unit and matrix unit can run in parallel.
    LLVM fails to schedule this reordering on its own in the basic kernel.

    Requires NUM_BUFFERS >= 2.  With NUM_BUFFERS == 2 the TDM must complete
    fully before each ds_read batch (async_wait(0)), but the ds_read/wmma
    overlap is still preserved.  NUM_BUFFERS >= 3 is recommended.
    """
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0
    gl.static_assert(NUM_BUFFERS >= 2, "lds_pipeline kernel requires NUM_BUFFERS >= 2")

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc, b_desc = _make_biased_descriptors(
        a_ptr, b_ptr, pid_m, pid_n, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M, BLOCK_N, BLOCK_K, SHARED_LAYOUT_A, SHARED_LAYOUT_B,
        PHYSICAL_MK, PHYSICAL_KN,
    )
    a_buffer, b_buffer = _alloc_ab_buffers(
        a_ptr, b_ptr, NUM_BUFFERS, BLOCK_M, BLOCK_N, BLOCK_K,
        SHARED_LAYOUT_A, SHARED_LAYOUT_B, PHYSICAL_MK, PHYSICAL_KN,
    )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # L2 prefetch helpers compute offsets relative to the descriptor's current
    # position. Since the per-block (M, N) offset is baked into the base, we
    # pass 0 for off_am/off_bn/load_idx so off_k is purely a K-direction delta.
    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(
            L2_PREFETCH_DISTANCE,
            0,
            a_desc,
            b_desc,
            0,
            0,
            BLOCK_K,
            NUM_BUFFERS,
            not PHYSICAL_MK,
            not PHYSICAL_KN,
        )

    # TDM prologue: fill the pipeline with NUM_BUFFERS-1 tiles
    for _ in gl.static_range(NUM_BUFFERS):
        _async_load_ab_masked(
            a_desc, b_desc, a_buffer, b_buffer, load_idx,
            BLOCK_K, NUM_BUFFERS, PHYSICAL_MK, PHYSICAL_KN,
        )
        load_idx += 1

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Register pre-load prologue: wait for tile 0 then read it into cur_a/cur_b.
    # After TDM prologue there are (NUM_BUFFERS-1)*2 ops in-flight; waiting for
    # (NUM_BUFFERS-2)*2 lets exactly one tile (tile 0) complete.
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    cur_a, cur_b = _read_ab(
        a_buffer, b_buffer, compute_idx,
        OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
        PHYSICAL_MK, PHYSICAL_KN,
    )

    # Main pipeline loop — first iteration peeled out below, then loop runs
    # for (num_k_tiles - NUM_BUFFERS - 1) remaining iterations.

    # ---- Peeled first iteration ----
    # WMMA for the current tile — uses operands pre-loaded in the
    # *previous* iteration so no ds_read stall before the matrix op.
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    _async_load_ab_masked(
        a_desc, b_desc, a_buffer, b_buffer, load_idx,
        BLOCK_K, NUM_BUFFERS, PHYSICAL_MK, PHYSICAL_KN,
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    load_idx += 1

    if USE_L2_PREFETCH:
        gemm_l2_prefetch(
            L2_PREFETCH_DISTANCE - 1,
            load_idx,
            a_desc,
            b_desc,
            0,
            0,
            BLOCK_K,
            not PHYSICAL_MK,
            not PHYSICAL_KN,
        )

    cur_a, cur_b = _read_ab(
        a_buffer, b_buffer, compute_idx + 1,
        OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
        PHYSICAL_MK, PHYSICAL_KN,
    )
    compute_idx += 1

    # ---- Remaining main-loop iterations ----
    for _ in range(num_k_tiles - NUM_BUFFERS - 1):
        # WMMA for the current tile — uses operands pre-loaded in the
        # *previous* iteration so no ds_read stall before the matrix op.
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        _async_load_ab_masked(
            a_desc, b_desc, a_buffer, b_buffer, load_idx,
            BLOCK_K, NUM_BUFFERS, PHYSICAL_MK, PHYSICAL_KN,
        )
        # Tighter wait: after issuing the new TDM there are (NUM_BUFFERS-1)*2
        # ops in-flight.  Waiting for (NUM_BUFFERS-2)*2 guarantees that tile
        # compute_idx+1 has landed in LDS.
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        load_idx += 1

        if USE_L2_PREFETCH:
            gemm_l2_prefetch(
                L2_PREFETCH_DISTANCE - 1,
                load_idx,
                a_desc,
                b_desc,
                0,
                0,
                BLOCK_K,
                not PHYSICAL_MK,
                not PHYSICAL_KN,
            )

        cur_a, cur_b = _read_ab(
            a_buffer, b_buffer, compute_idx + 1,
            OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
            PHYSICAL_MK, PHYSICAL_KN,
        )
        compute_idx += 1

    # Epilogue: no more TDM loads; drain the remaining NUM_BUFFERS-1 tiles.
    # The first NUM_BUFFERS-2 iterations still use the pre-load / WMMA pattern.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        next_a, next_b = _read_ab(
            a_buffer, b_buffer, compute_idx + 1,
            OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, NUM_BUFFERS,
            PHYSICAL_MK, PHYSICAL_KN,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        cur_a = next_a
        cur_b = next_b
        compute_idx += 1

    # Final WMMA for the last pre-loaded tile
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    # Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]

    # Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)

    # TDM Store: accumulator → shared memory → global memory
    SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0]
    )
    c_buffer = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_M, BLOCK_N],
        layout=SHARED_LAYOUT_C,
    )
    c_buffer.store(accumulator.to(c_ptr.type.element_ty))

    # Ensure all wavefronts have finished writing to LDS before TDM reads it.
    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_LAYOUT_C,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_buffer
    )
    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def _gemm_a16w16_v9_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """2x2 tile-sliced GEMM for gfx1250 (ported from the standalone "v9" kernel).

    Computes C = A @ B for the standard AITER A16W16 weight layout, i.e. with
    A (M, K) K-contiguous (stride_ak == 1) and B == w^T (K, N) K-contiguous
    (stride_bk == 1). This kernel does not support bias or activation and is
    intended for large, compute-bound shapes.

    Each block computes a 256x256 output tile as a 2x2 grid of 128x128 WMMA
    accumulators (C_tl, C_bl, C_tr, C_br). A is split into top/bottom halves,
    B into left/right halves. The K loop is software-pipelined with a
    double-buffered LDS staging area fed by the TDM engine, 2x unrolled so the
    two buffers alternate. The epilogue slices the last two K tiles into 64-row
    sub-tiles and interleaves the closing WMMAs with LDS-staged TDM stores.

    Preconditions (enforced on the host):
    - BLOCK_M == BLOCK_N == 256, BLOCK_K == 64, num_warps == 4.
    - K is a multiple of 2*BLOCK_K (the K loop is unrolled by 2) and
      iterMax = K / BLOCK_K > 4.
    """
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2

    out_dtype: gl.constexpr = c_ptr.type.element_ty

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # LDS padding for bank-conflict avoidance.
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [HALF_M, BLOCK_K], [1, 0]
    )
    # B is K-contiguous; store it as (N, K) tiles so the shared-memory order is
    # the descending [1, 0] the TDM descriptor requires, then permute on read to
    # present (K, N) WMMA operands.
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [HALF_N, BLOCK_K], [1, 0]
    )

    # TDM tensor descriptors. The block (M, N) offset is baked into the base
    # pointer and the descriptor bounds are set to the remaining extent so the
    # TDM engine zero-fills partial tiles at the matrix edge (N may not be a
    # multiple of BLOCK_N). K is padded on the host so it is never partial.
    m_rem = M - pid_m * BLOCK_M
    n_rem = N - pid_n * BLOCK_N
    a_top_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(m_rem, K), strides=(stride_am, stride_ak),
        block_shape=(HALF_M, BLOCK_K), layout=sharedLayoutA
    )
    a_bot_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + (pid_m * BLOCK_M + HALF_M) * stride_am,
        shape=(m_rem - HALF_M, K), strides=(stride_am, stride_ak),
        block_shape=(HALF_M, BLOCK_K), layout=sharedLayoutA
    )
    b_left_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(n_rem, K), strides=(stride_bn, stride_bk),
        block_shape=(HALF_N, BLOCK_K), layout=sharedLayoutB
    )
    b_right_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + (pid_n * BLOCK_N + HALF_N) * stride_bn,
        shape=(n_rem - HALF_N, K), strides=(stride_bn, stride_bk),
        block_shape=(HALF_N, BLOCK_K), layout=sharedLayoutB
    )

    # Double-buffered shared memory.
    smemA_top = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, HALF_M, BLOCK_K], layout=sharedLayoutA)
    smemA_bot = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, HALF_M, BLOCK_K], layout=sharedLayoutA)
    smemB_left = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, HALF_N, BLOCK_K], layout=sharedLayoutB)
    smemB_right = gl.allocate_shared_memory(b_ptr.type.element_ty, [2, HALF_N, BLOCK_K], layout=sharedLayoutB)

    # WMMA layout (hardcoded for the 256x256x64 / num_warps=4 bank assignment).
    wmmaLayout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 32]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmmaLayout, k_width=8)

    acc_tl = gl.zeros((HALF_M, HALF_N), gl.float32, wmmaLayout)
    acc_bl = gl.zeros((HALF_M, HALF_N), gl.float32, wmmaLayout)
    acc_tr = gl.zeros((HALF_M, HALF_N), gl.float32, wmmaLayout)
    acc_br = gl.zeros((HALF_M, HALF_N), gl.float32, wmmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 4)

    # Prologue: TDM tiles 0,1; LR 2 sub-tiles from buf 0.
    # TDM order: B_left, A_top, A_bot, B_right.
    gl.amd.gfx1250.tdm.async_load(b_left_desc, [0, 0], smemB_left.index(0))
    gl.amd.gfx1250.tdm.async_load(a_top_desc, [0, 0], smemA_top.index(0))
    gl.amd.gfx1250.tdm.async_load(a_bot_desc, [0, 0], smemA_bot.index(0))
    gl.amd.gfx1250.tdm.async_load(b_right_desc, [0, 0], smemB_right.index(0))

    b_left_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_left_desc, add_offsets=[0, BLOCK_K])
    a_top_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_top_desc, add_offsets=[0, BLOCK_K])
    a_bot_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_bot_desc, add_offsets=[0, BLOCK_K])
    b_right_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_right_desc, add_offsets=[0, BLOCK_K])

    gl.amd.gfx1250.tdm.async_load(b_left_desc, [0, 0], smemB_left.index(1))
    gl.amd.gfx1250.tdm.async_load(a_top_desc, [0, 0], smemA_top.index(1))
    gl.amd.gfx1250.tdm.async_load(a_bot_desc, [0, 0], smemA_bot.index(1))
    gl.amd.gfx1250.tdm.async_load(b_right_desc, [0, 0], smemB_right.index(1))

    # Wait for B_left[0] and A_top[0] (2 of 8 loads).
    gl.amd.gfx1250.tdm.async_wait(6)
    b_left_reg = smemB_left.index(0).permute([1, 0]).load(layout=dotOpLayoutB)
    a_top_reg = smemA_top.index(0).load(layout=dotOpLayoutA)

    # Main loop: 2x unrolled for buffer alternation.
    # Compute: C_tl -> C_bl -> C_tr -> C_br.
    # LR:      A_bot -> B_right -> B_left(next) -> A_top(next).
    # TDM:     B_left -> A_top -> A_bot -> B_right.
    for k in range(0, iterMax - 3, 2):
        # --- Even half: TDM loads -> buf 0 ---
        acc_tl = gl.amd.gfx1250.wmma(a_top_reg, b_left_reg, acc_tl)
        b_left_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_left_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(b_left_desc, [0, 0], smemB_left.index(0))
        gl.amd.gfx1250.tdm.async_wait(6)
        a_bot_reg = smemA_bot.index(0).load(layout=dotOpLayoutA)

        acc_bl = gl.amd.gfx1250.wmma(a_bot_reg, b_left_reg, acc_bl)
        a_top_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_top_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(a_top_desc, [0, 0], smemA_top.index(0))
        gl.amd.gfx1250.tdm.async_wait(6)
        b_right_reg = smemB_right.index(0).permute([1, 0]).load(layout=dotOpLayoutB)

        acc_tr = gl.amd.gfx1250.wmma(a_top_reg, b_right_reg, acc_tr)
        a_bot_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_bot_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(a_bot_desc, [0, 0], smemA_bot.index(0))
        gl.amd.gfx1250.tdm.async_wait(6)
        b_left_reg = smemB_left.index(1).permute([1, 0]).load(layout=dotOpLayoutB)

        acc_br = gl.amd.gfx1250.wmma(a_bot_reg, b_right_reg, acc_br)
        b_right_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_right_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(b_right_desc, [0, 0], smemB_right.index(0))
        gl.amd.gfx1250.tdm.async_wait(6)
        a_top_reg = smemA_top.index(1).load(layout=dotOpLayoutA)

        # --- Odd half: TDM loads -> buf 1 ---
        acc_tl = gl.amd.gfx1250.wmma(a_top_reg, b_left_reg, acc_tl)
        b_left_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_left_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(b_left_desc, [0, 0], smemB_left.index(1))
        gl.amd.gfx1250.tdm.async_wait(6)
        a_bot_reg = smemA_bot.index(1).load(layout=dotOpLayoutA)

        acc_bl = gl.amd.gfx1250.wmma(a_bot_reg, b_left_reg, acc_bl)
        a_top_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_top_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(a_top_desc, [0, 0], smemA_top.index(1))
        gl.amd.gfx1250.tdm.async_wait(6)
        b_right_reg = smemB_right.index(1).permute([1, 0]).load(layout=dotOpLayoutB)

        acc_tr = gl.amd.gfx1250.wmma(a_top_reg, b_right_reg, acc_tr)
        a_bot_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(a_bot_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(a_bot_desc, [0, 0], smemA_bot.index(1))
        gl.amd.gfx1250.tdm.async_wait(6)
        b_left_reg = smemB_left.index(0).permute([1, 0]).load(layout=dotOpLayoutB)

        acc_br = gl.amd.gfx1250.wmma(a_bot_reg, b_right_reg, acc_br)
        b_right_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(b_right_desc, add_offsets=[0, BLOCK_K])
        gl.amd.gfx1250.tdm.async_load(b_right_desc, [0, 0], smemB_right.index(1))
        gl.amd.gfx1250.tdm.async_wait(6)
        a_top_reg = smemA_top.index(0).load(layout=dotOpLayoutA)

    # Epilogue: last 2 tiles with sliced WMMA + interleaved stores.
    SLICE_M: gl.constexpr = HALF_M // 2  # 64

    sharedLayoutC: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [SLICE_M, HALF_N], [1, 0]
    )

    smemC0 = gl.allocate_shared_memory(out_dtype, [SLICE_M, HALF_N], layout=sharedLayoutC)
    smemC1 = gl.allocate_shared_memory(out_dtype, [SLICE_M, HALF_N], layout=sharedLayoutC)
    smemC2 = gl.allocate_shared_memory(out_dtype, [SLICE_M, HALF_N], layout=sharedLayoutC)
    smemC3 = gl.allocate_shared_memory(out_dtype, [SLICE_M, HALF_N], layout=sharedLayoutC)

    # Use the full (M, N) tensor extent as the descriptor bounds and pass each
    # store its absolute [row, col] offset. TDM masks the store against these
    # bounds, so partial M/N tiles at the matrix edge are dropped instead of
    # spilling OOB writes into the neighbouring rows via the row stride. NOTE:
    # the position MUST be supplied via the async_store offset, not by moving the
    # descriptor with update_tensor_descriptor(add_offsets=...) -- add_offsets
    # shifts the tile position without updating the bounds, which defeats OOB
    # masking on the column (dim-1) axis.
    c_base_m = pid_m * BLOCK_M
    c_base_n = pid_n * BLOCK_N
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        block_shape=(SLICE_M, HALF_N), layout=sharedLayoutC
    )

    # --- Tile iterMax-2 (even half) — unsliced WMMAs ---
    acc_tl = gl.amd.gfx1250.wmma(a_top_reg, b_left_reg, acc_tl)
    gl.amd.gfx1250.tdm.async_wait(5)
    a_bot_reg = smemA_bot.index(0).load(layout=dotOpLayoutA)

    acc_bl = gl.amd.gfx1250.wmma(a_bot_reg, b_left_reg, acc_bl)
    gl.amd.gfx1250.tdm.async_wait(4)
    b_right_reg = smemB_right.index(0).permute([1, 0]).load(layout=dotOpLayoutB)

    acc_tr = gl.amd.gfx1250.wmma(a_top_reg, b_right_reg, acc_tr)
    gl.amd.gfx1250.tdm.async_wait(3)
    b_left_reg = smemB_left.index(1).permute([1, 0]).load(layout=dotOpLayoutB)

    acc_br = gl.amd.gfx1250.wmma(a_bot_reg, b_right_reg, acc_br)
    gl.amd.gfx1250.tdm.async_wait(2)
    a_top_reg = smemA_top.index(1).load(layout=dotOpLayoutA)

    # --- Tile iterMax-1 (odd half) — sliced WMMAs + interleaved stores ---
    a_top0 = gl.amd.gfx1250.extract_slice(a_top_reg, [SLICE_M, BLOCK_K], [0, 0])
    a_top1 = gl.amd.gfx1250.extract_slice(a_top_reg, [SLICE_M, BLOCK_K], [SLICE_M, 0])

    acc_tl0 = gl.amd.gfx1250.extract_slice(acc_tl, [SLICE_M, HALF_N], [0, 0])
    acc_tl1 = gl.amd.gfx1250.extract_slice(acc_tl, [SLICE_M, HALF_N], [SLICE_M, 0])
    acc_bl0 = gl.amd.gfx1250.extract_slice(acc_bl, [SLICE_M, HALF_N], [0, 0])
    acc_bl1 = gl.amd.gfx1250.extract_slice(acc_bl, [SLICE_M, HALF_N], [SLICE_M, 0])
    acc_tr0 = gl.amd.gfx1250.extract_slice(acc_tr, [SLICE_M, HALF_N], [0, 0])
    acc_tr1 = gl.amd.gfx1250.extract_slice(acc_tr, [SLICE_M, HALF_N], [SLICE_M, 0])
    acc_br0 = gl.amd.gfx1250.extract_slice(acc_br, [SLICE_M, HALF_N], [0, 0])
    acc_br1 = gl.amd.gfx1250.extract_slice(acc_br, [SLICE_M, HALF_N], [SLICE_M, 0])

    # R4
    acc_tl0 = gl.amd.gfx1250.wmma(a_top0, b_left_reg, acc_tl0)
    gl.amd.gfx1250.tdm.async_wait(1)
    a_bot_reg = smemA_bot.index(1).load(layout=dotOpLayoutA)
    a_bot0 = gl.amd.gfx1250.extract_slice(a_bot_reg, [SLICE_M, BLOCK_K], [0, 0])
    a_bot1 = gl.amd.gfx1250.extract_slice(a_bot_reg, [SLICE_M, BLOCK_K], [SLICE_M, 0])
    gl.amd.gfx1250.tdm.async_wait(0)
    b_right_reg = smemB_right.index(1).permute([1, 0]).load(layout=dotOpLayoutB)

    # R5
    acc_tl1 = gl.amd.gfx1250.wmma(a_top1, b_left_reg, acc_tl1)
    acc_tl0 = acc_tl0.to(out_dtype)
    smemC0.store(acc_tl0)

    # R6: tl0 -> (row 0, left half)
    acc_bl0 = gl.amd.gfx1250.wmma(a_bot0, b_left_reg, acc_bl0)
    acc_tl1 = acc_tl1.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m, c_base_n], smemC0)
    smemC1.store(acc_tl1)

    # R7: tl1 -> (row SLICE_M, left half)
    acc_bl1 = gl.amd.gfx1250.wmma(a_bot1, b_left_reg, acc_bl1)
    acc_bl0 = acc_bl0.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + SLICE_M, c_base_n], smemC1)
    smemC2.store(acc_bl0)

    # R8: bl0 -> (row 2*SLICE_M, left half)
    acc_tr0 = gl.amd.gfx1250.wmma(a_top0, b_right_reg, acc_tr0)
    acc_bl1 = acc_bl1.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + 2 * SLICE_M, c_base_n], smemC2)
    smemC3.store(acc_bl1)

    # R9: bl1 -> (row 3*SLICE_M, left half). smemC0 reused for tr0
    # (original tl0 was consumed by R6's TDM store).
    acc_tr1 = gl.amd.gfx1250.wmma(a_top1, b_right_reg, acc_tr1)
    acc_tr0 = acc_tr0.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + 3 * SLICE_M, c_base_n], smemC3)
    smemC0.store(acc_tr0)

    # R10: tr0 -> (row 0, right half).
    acc_br0 = gl.amd.gfx1250.wmma(a_bot0, b_right_reg, acc_br0)
    acc_tr1 = acc_tr1.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m, c_base_n + HALF_N], smemC0)
    smemC1.store(acc_tr1)

    # R11: tr1 -> (row SLICE_M, right half)
    acc_br1 = gl.amd.gfx1250.wmma(a_bot1, b_right_reg, acc_br1)
    acc_br0 = acc_br0.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + SLICE_M, c_base_n + HALF_N], smemC1)
    smemC2.store(acc_br0)

    # Tail: br0 -> (row 2*SLICE_M, right half), br1 -> (row 3*SLICE_M, right half)
    acc_br1 = acc_br1.to(out_dtype)
    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + 2 * SLICE_M, c_base_n + HALF_N], smemC2)
    smemC3.store(acc_br1)

    gl.amd.gfx1250.tdm.async_store(c_desc, [c_base_m + 3 * SLICE_M, c_base_n + HALF_N], smemC3)

    gl.amd.gfx1250.tdm.async_wait(0)


_KERNEL_MAP = {
    "basic": _gemm_a16w16_basic_kernel,
    "warp_priority": _gemm_a16w16_warp_priority_kernel,
    "k_subtiling": _gemm_a16w16_k_subtiling_kernel,
    "lds_pipeline": _gemm_a16w16_lds_pipeline_kernel,
    "v9": _gemm_a16w16_v9_kernel,
}

# Minimum pipeline depth (NUM_BUFFERS) each kernel needs to be correct. The
# prologue/epilogue of the pipelined kernels assume the pipeline can be filled,
# i.e. num_k_tiles >= NUM_BUFFERS, so NUM_BUFFERS is clamped to num_k_tiles on
# the host. When that clamp drops below a kernel's minimum (e.g. a single K
# tile), there is nothing to pipeline and we fall back to "basic".
_KERNEL_MIN_BUFFERS = {
    "basic": 1,
    "warp_priority": 3,
    "k_subtiling": 2,
    "lds_pipeline": 2,
    # v9 manages its own double-buffered LDS pipeline and ignores NUM_BUFFERS.
    "v9": 1,
}

# Kernels that issue TDM loads with absolute K offsets against fixed descriptor
# bounds. Tiles past K are zero-filled by the TDM engine (descriptors use zero
# padding), so the pipeline can over-issue tiles harmlessly when
# num_k_tiles < NUM_BUFFERS. These kernels stay correct at the requested
# NUM_BUFFERS for any K, so they skip the host-side clamp/fallback and compile a
# single variant regardless of shape.
_ZERO_FILL_SAFE_KERNELS = {"lds_pipeline"}


def _gemm_a16w16_v9(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype,
    y: Optional[torch.Tensor],
):
    """Host wrapper for the v9 2x2-sliced kernel (large compute-bound shapes).

    Forces the kernel's fixed tiling (256x256x64, num_warps=4), pads K to a
    multiple of 2*BLOCK_K, and requires the standard AITER A16W16 layout: x
    row-major (M, K) and w row-major (N, K) so that, after ``w = w.T``, both A
    and B are K-contiguous. Bias and activation are not supported.
    """
    M, K = x.shape
    N, _ = w.shape

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4

    # The K loop is unrolled by 2 and the epilogue peels the last two K tiles,
    # so iterMax = K / BLOCK_K must be even (K divisible by 2*BLOCK_K) and > 4.
    K_mult = 2 * BLOCK_K
    K_padded = triton.cdiv(K, K_mult) * K_mult
    if K_padded != K:
        x = torch.nn.functional.pad(x, (0, K_padded - K))
        w = torch.nn.functional.pad(w, (0, K_padded - K))
        K = K_padded

    assert (
        K // BLOCK_K
    ) > 4, f"v9 kernel requires K/{BLOCK_K} > 4 (got K={K}); use kernel_type='basic' for small K"

    # Partial M and N tiles are handled inside the kernel: the TDM load
    # descriptors zero-fill OOB rows/columns and the store descriptor masks OOB
    # rows/columns against its (m_rem, n_rem) bounds, so neither M nor N needs to
    # be a multiple of the block size.
    w = w.T  # (K, N)

    assert (
        x.stride(1) == 1
    ), f"v9 kernel requires row-major x (M, K) with stride_ak == 1, got strides {x.stride()}"
    assert x.is_contiguous(), "v9 kernel requires contiguous x"
    assert w.stride(0) == 1, (
        "v9 kernel requires K-contiguous w^T (standard (N, K) row-major weights), "
        f"got w^T strides {w.stride()}"
    )

    out = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    _gemm_a16w16_v9_kernel[grid](
        x,
        w,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    if y is not None:
        y.copy_(out)
        return y
    return out


def gemm_a16w16_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[Dict] = None,
    activation: Optional[str] = None,
    kernel_type: str = "basic",
):
    """
    Compute 16 bit gemm y = x @ w^T + bias using gluon (gfx1250).

    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (N, K), internally transposed
        bias: Optional bias tensor of shape (N,)
        dtype: Output data type
        y: Optional pre-allocated output tensor
        config: Kernel tuning parameters:
            - BLOCK_M: Tile size in M dimension (default: 128)
            - BLOCK_N: Tile size in N dimension (default: 128)
            - BLOCK_K: Tile size in K dimension (default: 32)
            - NUM_BUFFERS: Pipeline stages (default: 2)
            - num_warps: Warps per block (default: 8)
        activation: Activation function ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu")
        kernel_type: Kernel variant to use:
            - "basic": Simple pipelining with async TDM loads (default)
            - "warp_priority": Warp priority pipelining (requires NUM_BUFFERS >= 3)
            - "k_subtiling": K-dimension subtiling for LDS latency hiding (requires BLOCK_K >= 64)
            - "lds_pipeline": Manually pipelines LDS loads across K-tiles; places
              load_shared_relaxed for tile i+1 before wmma for tile i so the
              hardware LDS unit and matrix unit run in parallel (requires NUM_BUFFERS >= 2,
              NUM_BUFFERS >= 3 recommended)
            - "v9": 2x2 tile-sliced kernel tuned for large compute-bound shapes.
              Uses a fixed 256x256x64 tiling (num_warps=4) with a double-buffered
              TDM/WMMA pipeline and a sliced, store-interleaved epilogue. Requires
              the standard layout (x row-major (M, K), w row-major (N, K)) and does
              NOT support bias or activation; ignores config/NUM_BUFFERS.

    Returns:
        Output tensor of shape (M, N)
    """

    assert (
        kernel_type in _KERNEL_MAP
    ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"

    _LOGGER.info(
        f"GEMM_A16W16 [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)} kernel={kernel_type}"
    )

    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    if kernel_type == "v9":
        assert bias is None, "v9 kernel does not support bias"
        assert activation is None, "v9 kernel does not support activation"
        return _gemm_a16w16_v9(x, w, dtype, y)

    M, K = x.shape
    N, _ = w.shape

    if config is None:
        config, _ = _get_config(M, N, K)

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
    num_warps = config["num_warps"]
    L2_PREFETCH_DISTANCE = config.get("L2_PREFETCH_DISTANCE", 0)

    # Pad K to be divisible by block k so tdm loads never read out of bounds
    K_padded = triton.cdiv(K, BLOCK_K) * BLOCK_K
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size))
        K = K_padded

    # Zero-fill-safe kernels (see _ZERO_FILL_SAFE_KERNELS) mask K via the TDM
    # descriptor bounds: tiles past K are zero-filled and contribute nothing, so
    # they stay correct at the requested NUM_BUFFERS for any K and compile a
    # single variant. The other kernels calibrate their prologue/epilogue waits
    # for a full pipeline and are only correct when num_k_tiles >= NUM_BUFFERS,
    # so clamp NUM_BUFFERS to the available K tiles and fall back to "basic"
    # (which works at any depth) when there are too few tiles.
    num_k_tiles = triton.cdiv(K, BLOCK_K)
    if kernel_type not in _ZERO_FILL_SAFE_KERNELS:
        NUM_BUFFERS = min(NUM_BUFFERS, num_k_tiles)
        if NUM_BUFFERS < _KERNEL_MIN_BUFFERS[kernel_type]:
            kernel_type = "basic"

    w = w.T

    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False
    else:
        raise ValueError(
            f"x must be contiguous in at least one dimension, got strides {x.stride()}"
        )

    if w.stride(1) == 1:
        physical_kn = True
    elif w.stride(0) == 1:
        physical_kn = False
    else:
        raise ValueError(
            f"w must be contiguous in at least one dimension, got strides {w.stride()}"
        )

    if y is None:
        y = torch.empty((M, N), dtype=dtype).cuda()

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    wmma_layout = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
    )

    operand_a = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)

    shared_layouts = create_shared_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, physical_mk, physical_kn
    )
    shared_a, shared_b = shared_layouts[0], shared_layouts[1]

    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n, 1)

    kernel_fn = _KERNEL_MAP[kernel_type]

    kernel_fn[grid](
        x,
        w,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        PHYSICAL_MK=physical_mk,
        PHYSICAL_KN=physical_kn,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        activation=_get_activation_from_str(activation) if activation else None,
        USE_ACTIVATION=activation is not None,
        ADD_BIAS=(bias is not None),
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
        num_warps=num_warps,
    )

    return y
