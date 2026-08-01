# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _get_config,
    _get_ptpc_config,
)


def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    splitK: int | None = None,
    YQ: torch.Tensor | None = None,
    transpose_bm: bool | None = False,
    transpose_bm_in: bool | None = False,
    config: dict | None = None,
    emit_ptpc: bool = False,
    YQ_ptpc: torch.Tensor | None = None,
    y_scale: torch.Tensor | None = None,
    row_amax: torch.Tensor | None = None,
    row_counter: torch.Tensor | None = None,
):
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to INT8 during computation using per-token grouped quantization.
    W is pre-quantized INT8 with per-batch-element scaling.

    Args:
        X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
            Quantized to INT8 on-the-fly during GEMM.
        WQ (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
        w_scale (torch.Tensor): Per-batch scale for WQ with shape (1,).
        group_size (int): Group size for per-token grouped quantization of X. Must be power of 2.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
        transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M).
        emit_ptpc (bool): Emit flattened FP8 output and one FP32 scale per M row.
        YQ_ptpc (Optional[torch.Tensor]): Pre-allocated FP8 output with shape (M, B*N).
        y_scale (Optional[torch.Tensor]): Pre-allocated FP32 scales with shape (M, 1).
        row_amax (Optional[torch.Tensor]): Zero-initialized FP32 workspace with shape (M,).
        row_counter (Optional[torch.Tensor]): Zero-initialized int32 workspace with shape (3*M,).

    Returns:
        torch.Tensor: BF16/FP16 output, or ``(YQ_ptpc, y_scale)`` when
        ``emit_ptpc=True``.
    """

    # Check constraints.
    if not transpose_bm_in:
        B = X.shape[0]
        M = X.shape[1]
    else:
        M = X.shape[0]
        B = X.shape[1]
    K = X.shape[2]
    N = WQ.shape[1]

    assert B == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert K == WQ.shape[2], "Incompatible K dimensions!!!"
    assert (
        triton.next_power_of_2(group_size) == group_size
    ), "group_size mush be power of 2"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    WQ = WQ.transpose(1, 2)

    has_bias = bias is not None
    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    else:
        if transpose_bm:
            assert (
                YQ.shape[0] == M and YQ.shape[1] == B and YQ.shape[2] == N
            ), "Output dimension error"
        else:
            assert (
                YQ.shape[0] == B and YQ.shape[1] == M and YQ.shape[2] == N
            ), "Output dimension error"

    if config is None:
        config, _ = (
            _get_ptpc_config(M, N, K) if emit_ptpc else _get_config(M, N, K)
        )
    else:
        config = dict(config)
    config["BLOCK_SIZE_K"] = group_size
    config["kpack"] = 1

    if emit_ptpc:
        assert transpose_bm, "PTPC output requires transpose_bm=True"
        assert M <= 64, "Fused PTPC output currently supports M <= 64"
        assert YQ.is_contiguous(), "PTPC BF16 scratch must be contiguous (M, B, N)"
        if YQ_ptpc is None:
            YQ_ptpc = torch.empty((M, B * N), dtype=WQ.dtype, device=X.device)
        if y_scale is None:
            y_scale = torch.empty((M, 1), dtype=torch.float32, device=X.device)
        if row_amax is None:
            row_amax = torch.zeros((M,), dtype=torch.float32, device=X.device)
        if row_counter is None:
            row_counter = torch.zeros((3 * M,), dtype=torch.int32, device=X.device)
        assert YQ_ptpc.shape == (M, B * N) and YQ_ptpc.is_contiguous()
        assert y_scale.shape == (M, 1) and y_scale.dtype == torch.float32
        assert row_amax.shape == (M,) and row_amax.dtype == torch.float32
        assert row_counter.shape == (3 * M,) and row_counter.dtype == torch.int32
    else:
        # Dead placeholders removed by the EMIT_PTPC constexpr specialization.
        YQ_ptpc = YQ
        y_scale = YQ
        row_amax = YQ
        row_counter = YQ

    grid = lambda META: (
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    DTYPE_MAX = (
        torch.finfo(WQ.dtype).max
        if torch.is_floating_point(WQ)
        else torch.iinfo(WQ.dtype).max
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[
        grid
    ](
        X,
        WQ,
        YQ,
        YQ_ptpc,
        y_scale,
        row_amax,
        row_counter,
        w_scale,
        bias,
        B,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        YQ_ptpc.stride(0),
        YQ_ptpc.stride(1),
        y_scale.stride(0),
        bias.stride(0) if has_bias else 0,
        has_bias,
        EMIT_PTPC=emit_ptpc,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    if emit_ptpc:
        return YQ_ptpc, y_scale
    return YQ
