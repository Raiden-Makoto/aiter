# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.flydsl import gemm_kernels
from aiter.ops.triton import quant as quant_mod
from aiter.ops.shuffle import shuffle_weight


def _inputs(m=2):
    return (
        torch.empty((m, 2048), dtype=dtypes.fp8),
        torch.empty((4096, 2048), dtype=dtypes.fp8),
        torch.empty((m, 1), dtype=torch.float32),
        torch.empty((4096, 1), dtype=torch.float32),
    )


def test_glm52_qb_parses_committed_and_current_flydsl_names():
    assert gemm_mod._parse_flydsl_kernel_name(
        "flydsl_bpreshuflle_128x256x128_F8_F8_B16_2x1x1x2x4_default"
    ) == (128, 256, 128, 1, 2, 4, 2, "default")
    assert gemm_mod._parse_flydsl_kernel_name(
        "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default"
    ) == (128, 256, 128, 1, 2, 4, 2, "default")


def test_glm52_qb_fake_registration_shapes():
    with FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(t) for t in _inputs(3))
        q_nope, q_scale, q_pe = gemm_mod.gemm_a8w8_bpreshuffle_glm52_qb(*args)

    assert q_nope.shape == (3, 16, 96)
    assert q_nope.dtype == torch.uint8
    assert q_scale.shape == (3, 16, 6)
    assert q_scale.dtype == torch.uint8
    assert q_scale.stride() == (16, 1, 48)
    assert q_pe.shape == (3, 16, 64)
    assert q_pe.dtype == torch.bfloat16


def test_glm52_qb_dispatches_selected_flydsl_config(monkeypatch):
    args = _inputs()
    seen = {}
    kernel_name = "flydsl_bpreshuflle_64x128x256_F8_F8_B16_2x0x1x4x4_default"

    monkeypatch.setattr(
        gemm_mod,
        "get_GEMM_config_with_quant_type",
        lambda *a, **k: {"libtype": "flydsl", "kernelName": kernel_name},
    )
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_mod, "is_flydsl_available", lambda: True)

    def fake_flydsl(*call_args, **kwargs):
        seen["tile"] = call_args[7:13]
        seen["kwargs"] = kwargs
        return call_args[4], call_args[5], call_args[6]

    monkeypatch.setattr(
        gemm_kernels, "flydsl_preshuffle_gemm_glm52_qb", fake_flydsl
    )
    q_nope, q_scale, q_pe = gemm_mod.gemm_a8w8_bpreshuffle_glm52_qb(*args)

    assert (q_nope.shape, q_scale.shape, q_pe.shape) == (
        (2, 16, 96),
        (2, 16, 6),
        (2, 16, 64),
    )
    assert seen["tile"] == (64, 128, 256, 1, 4, 4)
    assert seen["kwargs"] == {"lds_stage": 2, "enable_scheduler": True}


def test_glm52_qb_guarded_fallback(monkeypatch):
    args = _inputs()
    q_bf16 = torch.arange(2 * 4096, dtype=torch.float32).to(torch.bfloat16).view(2, 4096)
    packed = torch.full((2 * 16, 96), 0x32, dtype=torch.uint8)
    scales = torch.full((2 * 16, 6), 127, dtype=torch.uint8)
    seen = {}

    monkeypatch.setattr(
        gemm_mod,
        "get_GEMM_config_with_quant_type",
        lambda *a, **k: {"libtype": "ck", "kernelName": "scalar-cde"},
    )
    monkeypatch.setattr(
        gemm_mod, "gemm_a8w8_bpreshuffle", lambda *a, **k: q_bf16
    )

    def fake_quant(x):
        seen["quant_input"] = x.clone()
        return packed, scales

    monkeypatch.setattr(quant_mod, "dynamic_mxfp4_quant", fake_quant)
    q_nope, q_scale, q_pe = gemm_mod.gemm_a8w8_bpreshuffle_glm52_qb(*args)

    expected = q_bf16.view(2, 16, 256)
    torch.testing.assert_close(seen["quant_input"], expected[..., :192].reshape(32, 192))
    assert q_nope.data_ptr() == packed.data_ptr()
    assert q_scale.data_ptr() == scales.data_ptr()
    torch.testing.assert_close(q_pe, expected[..., 192:])


def test_glm52_qb_flydsl_matches_materialized_reference():
    if (
        not torch.cuda.is_available()
        or gemm_mod.get_gfx() != "gfx950"
        or not gemm_mod.is_flydsl_available()
    ):
        pytest.skip("packed GLM-5.2 q_b numerical test requires gfx950 and FlyDSL")

    torch.manual_seed(7)
    m = 2
    xq = torch.randn((m, 2048), device="cuda").to(dtypes.fp8)
    wq = shuffle_weight(
        torch.randn((4096, 2048), device="cuda").to(dtypes.fp8),
        layout=(16, 16),
    )
    x_scale = torch.rand((m, 1), device="cuda", dtype=torch.float32) + 0.25
    w_scale = torch.rand((4096, 1), device="cuda", dtype=torch.float32) + 0.25

    q_bf16 = gemm_mod.gemm_a8w8_bpreshuffle(
        xq, wq, x_scale, w_scale, dtype=torch.bfloat16
    ).view(m, 16, 256)
    q_nope_ref, q_scale_ref = quant_mod.dynamic_mxfp4_quant(
        q_bf16[..., :192].contiguous().view(m * 16, 192)
    )
    q_nope, q_scale, q_pe = gemm_mod.gemm_a8w8_bpreshuffle_glm52_qb(
        xq, wq, x_scale, w_scale
    )

    torch.testing.assert_close(
        q_nope, q_nope_ref.view(m, 16, 96), rtol=0, atol=0
    )
    torch.testing.assert_close(
        q_scale, q_scale_ref.view(m, 16, 6), rtol=0, atol=0
    )
    torch.testing.assert_close(q_pe, q_bf16[..., 192:], rtol=0, atol=0)
