# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True):
    torch.manual_seed(0)
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, out_dtype, y


def get_x_vals():
    return [
        (1, 1, 1),
        (3, 5, 2),
        (1024, 1024, 1024),
        (32, 256, 7168),
        (64, 640, 2880),
        (128, 2880, 512),
    ]


def get_fewer_x_vals():
    return [
        (16, 1024, 1024),
        (128, 512, 7168),
    ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_gemm_a16_w16(M, N, K):
    x, w, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )
    torch_out = F.linear(x, w, bias=None)
    triton_out = gemm_a16w16(x, w)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_activation(M, N, K, dtype, output, activation):
    x, w, out_dtype, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=output)
    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu":
        torch_out = F.silu(torch_out)

    triton_out = gemm_a16w16(x, w, None, out_dtype, y, activation=activation)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_layout(M, N, K, layout):
    torch.cuda.empty_cache()
    x, w, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )
    torch_out = F.linear(x, w, bias=None)
    triton_out = gemm_a16w16(x, w, None, out_dtype, y)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M, N, K, output):
    torch.cuda.empty_cache()
    x, w, _, y = generate_gemm_a16w16_inputs(M, N, K, torch.bfloat16, output=output)
    torch_out = F.linear(x, w, bias=None)

    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_atomic_layout(M, N, K, layout):
    torch.cuda.empty_cache()
    x, w, _, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=True
    )
    torch_out = F.linear(x, w, bias=None)
    y = y.to(torch.float32).zero_()
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
