# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import functools
from aiter import dtypes
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.torch_guard import torch_compile_guard


@compile_ops("module_mhc")
def mhc_pre_gemm_sqrsum(
    out: Tensor,
    sqrsum: Tensor,
    x: Tensor,
    fn: Tensor,
    tile_k: int = 128,  # 64 or 128
) -> None: ...


@compile_ops("module_mhc")
def mhc_pre_big_fuse(
    post_mix: Tensor,
    comb_mix: Tensor,
    layer_input: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> None: ...


@compile_ops("module_mhc")
def mhc_pre_big_fuse_rmsnorm(
    post_mix: Tensor,
    comb_mix: Tensor,
    out: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> None: ...


@functools.lru_cache(maxsize=1024)
def get_mhc_pre_splitk(m: int, hc_hidden_size: int) -> tuple[int, int]:
    prefetch_stages = 2
    tile_m = 16 * 4
    num_cu = get_cu_num()
    tile_k_tg_dict = {
        128: 2 * num_cu,
        64: 4 * num_cu,
    }
    selected_splitk = 1
    selected_tile_k = 64
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg_dict[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, meanwhile_tg in tile_k_tg_dict.items():
        if (hc_hidden_size % tile_k) != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hc_hidden_size % (splitk * tile_k) != 0 or (hc_hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            if selected_score < score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            # print(f"{selected_score=} {selected_splitk=} {selected_tile_k=} {score=} {splitk=} {tile_k=}")
            if num_tg > meanwhile_tg * 2:
                break

    return selected_splitk, selected_tile_k


def mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    device = residual.device
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    return post_mix, comb_mix, layer_input


@torch_compile_guard(gen_fake=mhc_pre_fake)
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult or (
        hc_mult3 == hc_mult and sinkhorn_repeat == 0
    )
    hc_hidden_size = hc_mult * hidden_size
    selected_splitk, selected_tile_k = get_mhc_pre_splitk(m, hc_hidden_size)
    device = residual.device
    out_pad = torch.empty(
        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32, device=device)
    mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, selected_tile_k)
    # out = out.sum(0)
    # sqrsum = sqrsum.sum(0)

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    if norm_weight is not None:
        mhc_pre_big_fuse_rmsnorm(
            post_mix,
            comb_mix,
            layer_input,
            out,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            norm_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        mhc_pre_big_fuse(
            post_mix,
            comb_mix,
            layer_input,
            out,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    return post_mix, comb_mix, layer_input


@compile_ops("module_mhc")
def mhc_post(
    out: Tensor,
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> None: ...


def mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route B fused boundary: ``mhc_post`` + pre-norm GEMM/sqrsum, then the
    unchanged ``mhc_pre_big_fuse``.

    Replaces the unfused ``mhc_post -> mhc_pre_gemm_sqrsum -> mhc_pre_big_fuse``
    chain for the next layer's pre block. ``big_fuse`` is byte-for-byte unchanged;
    it still re-reads the bf16 residual that the fused kernel writes.

    Returns:
        residual_out: post-mapped residual, shape (m, hc_mult, hidden_size) bf16
        post_mix:     shape (m, hc_mult, 1) fp32
        comb_mix:     shape (m, hc_mult, hc_mult) fp32
        layer_input:  shape (m, hidden_size) bf16
    """
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    device = residual.device

    out = torch.empty_like(residual)
    gemm_out_mul, gemm_out_sqrsum = mhc_post_gemm_sqrsum(
        out, x, residual, post_layer_mix, comb_res_mix, fn
    )

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    mhc_pre_big_fuse(
        post_mix,
        comb_mix,
        layer_input,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    return out, post_mix, comb_mix, layer_input


@compile_ops("module_mhc")
def mhc_post_gemm_sqrsum(
    out: Tensor,
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    fn: Tensor,
) -> list[Tensor]:
    """Route B fused post + pre-norm GEMM/sqrsum.

    Writes the bf16 residual ``out`` byte-for-byte identically to ``mhc_post`` and
    returns ``[gemm_out_mul, gemm_out_sqrsum]`` where
    ``gemm_out_mul`` is ``(k_blocks, m, hc_mult3)`` and ``gemm_out_sqrsum`` is
    ``(k_blocks, m)`` -- exactly the split layout ``mhc_pre_big_fuse`` consumes.
    ``k_blocks`` is the post kernel's internal hidden split (== n_splits).
    """
    ...
