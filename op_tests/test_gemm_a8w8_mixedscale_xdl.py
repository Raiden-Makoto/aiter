# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_ck
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_weight


MIXED_KERNEL = (
    "a8w8_blockscale_bpreshuffle_1x1x128_"
    "256x16x128x128_8x16_16x16_16x16x1_8x32x1_"
    "1x16x1x16_8_1x2_intrawave_v1"
)


@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 64])
def test_gemm_a8w8_mixedscale_xdl(m):
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
    x_scale_ck = x_scale.T.contiguous().view_as(x_scale)
    w_q, w_scale = aiter.pertoken_quant(
        weight,
        quant_dtype=aiter.dtypes.fp8,
    )
    w_q_ref = w_q
    w_q = shuffle_weight(w_q_ref, (16, 16)).contiguous()
    w_scale_ck = w_scale.expand(n, k // 128).contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    actual = gemm_a8w8_blockscale_bpreshuffle_ck(
        x_q,
        w_q,
        x_scale_ck,
        w_scale_ck,
        out,
        MIXED_KERNEL,
    )
    expected = F.linear(
        (
            x_q.view(m, k // 128, 128).float()
            * x_scale.float().unsqueeze(-1)
        ).view(m, k),
        w_q_ref.float() * w_scale.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.1)


def test_gemm_a8w8_mixedscale_xdl_cuda_graph():
    m, n, k = 4, 6144, 4096
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    x_q, x_scale = per_group_quant_hip(
        x,
        quant_dtype=aiter.dtypes.fp8,
        group_size=128,
        transpose_scale=False,
    )
    x_scale = x_scale.T.contiguous().view_as(x_scale)
    w_q, w_scale = aiter.pertoken_quant(
        weight,
        quant_dtype=aiter.dtypes.fp8,
    )
    w_q = shuffle_weight(w_q, (16, 16)).contiguous()
    w_scale = w_scale.expand(n, k // 128).contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    for _ in range(3):
        gemm_a8w8_blockscale_bpreshuffle_ck(
            x_q, w_q, x_scale, w_scale, out, MIXED_KERNEL
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_a8w8_blockscale_bpreshuffle_ck(
            x_q, w_q, x_scale, w_scale, out, MIXED_KERNEL
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
