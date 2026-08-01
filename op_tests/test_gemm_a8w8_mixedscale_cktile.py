# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mixedscale_bpreshuffle_cktile
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_weight


@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 64])
def test_gemm_a8w8_mixedscale_cktile(m):
    n, k = 6144, 4096
    torch.manual_seed(2026 + m)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    x_q, x_scale = per_group_quant_hip(
        x,
        quant_dtype=aiter.dtypes.fp8,
        group_size=128,
        transpose_scale=False,
    )
    w_q, w_scale = aiter.pertoken_quant(
        weight,
        quant_dtype=aiter.dtypes.fp8,
    )
    w_q_ref = w_q
    w_q = shuffle_weight(w_q_ref, (16, 16)).contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    actual = gemm_a8w8_mixedscale_bpreshuffle_cktile(
        x_q,
        w_q,
        x_scale,
        w_scale,
        out,
    )
    expected = F.linear(
        (
            x_q.view(m, k // 128, 128).float()
            * x_scale.float().unsqueeze(-1)
        ).view(m, k),
        w_q_ref.float() * w_scale.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.1)


def test_gemm_a8w8_mixedscale_cktile_cuda_graph():
    m, n, k = 4, 6144, 4096
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    x_q, x_scale = per_group_quant_hip(
        x,
        quant_dtype=aiter.dtypes.fp8,
        group_size=128,
        transpose_scale=False,
    )
    w_q, w_scale = aiter.pertoken_quant(
        weight,
        quant_dtype=aiter.dtypes.fp8,
    )
    w_q = shuffle_weight(w_q, (16, 16)).contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    for _ in range(3):
        gemm_a8w8_mixedscale_bpreshuffle_cktile(
            x_q, w_q, x_scale, w_scale, out
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_a8w8_mixedscale_bpreshuffle_cktile(
            x_q, w_q, x_scale, w_scale, out
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
