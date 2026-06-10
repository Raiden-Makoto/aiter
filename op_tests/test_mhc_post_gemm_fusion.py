# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline numeric gate for Route B: the fused ``mhc_post_gemm_sqrsum`` HIP kernel.

This is the *correctness gate* for the fused post + pre-norm GEMM/sqrsum kernel.
It checks, on the production dsv4 shape ([M, 4, 7168] bf16 residual, fn fp32
[24, 28672]):

  (a) the fused kernel's GEMM/sqrsum partials (summed over the post hidden split)
      match the unfused ``mhc_post`` -> ``mhc_pre_gemm_sqrsum`` outputs, and match
      a pure-torch reference that emulates the bf16 residual round-trip;
  (b) the bf16 residual written by the fused kernel is byte-identical to the
      unfused ``mhc_post`` output (the fused kernel must not change the residual);
  (c) the end-to-end ``layer_input`` after running the UNCHANGED ``big_fuse`` on
      the fused vs unfused gemm partials agrees within tolerance.

Divergence in (a)/(c) should come only from GEMM reduction order, so the
tolerances are loose on the GEMM (rtol ~2e-2) and tighter on layer_input.

Run as a script:   python op_tests/test_mhc_post_gemm_fusion.py
Run with pytest:   pytest op_tests/test_mhc_post_gemm_fusion.py
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.mhc import (
    get_mhc_pre_splitk,
    mhc_post_gemm_sqrsum,
    mhc_post_pre,
)

HC_MULT = 4
HIDDEN = 7168  # dsv4 hidden_size; 7168 % 512 == 0 and % 1024 == 0
HC_MULT3 = 2 * HC_MULT + HC_MULT * HC_MULT  # 24
HC_HIDDEN = HC_MULT * HIDDEN  # 28672

RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20


def _rel_err(ref: torch.Tensor, got: torch.Tensor):
    ref = ref.float()
    got = got.float()
    denom = ref.abs().clamp_min(1e-6)
    rel = (got - ref).abs() / denom
    return rel.max().item(), rel.mean().item()


def _make_inputs(m: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, dtype=dtypes.bf16, device="cuda", generator=g)
    residual = torch.randn(
        m, HC_MULT, HIDDEN, dtype=dtypes.bf16, device="cuda", generator=g
    )
    post_layer_mix = torch.randn(
        m, HC_MULT, dtype=dtypes.fp32, device="cuda", generator=g
    )
    comb_res_mix = torch.randn(
        m, HC_MULT, HC_MULT, dtype=dtypes.fp32, device="cuda", generator=g
    )
    fn = torch.randn(HC_MULT3, HC_HIDDEN, dtype=dtypes.fp32, device="cuda", generator=g)
    hc_scale = torch.randn(3, dtype=dtypes.fp32, device="cuda", generator=g) * 0.1
    hc_base = torch.randn(
        HC_MULT3, dtype=dtypes.fp32, device="cuda", generator=g
    ) * 0.1
    return x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base


def _post_ref(x, residual, post_layer_mix, comb_res_mix):
    """bf16 residual reference, matching aiter mhc_post_ref."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())  # (m, hc, hidden)
    res = x.float().unsqueeze(-2) * post_layer_mix.unsqueeze(-1) + term2
    return res.bfloat16()


def _unfused_gemm_partials(res_out_unfused, fn):
    """Run the prod-style split-k gemm/sqrsum, return summed (m,24) and (m,)."""
    m = res_out_unfused.shape[0]
    splitk, tile_k = get_mhc_pre_splitk(m, HC_HIDDEN)
    out_pad = torch.empty(splitk, m, 32, dtype=dtypes.fp32, device="cuda")
    out_u = out_pad[:, :, :HC_MULT3]
    sqrsum_u = torch.empty(splitk, m, dtype=dtypes.fp32, device="cuda")
    aiter.mhc_pre_gemm_sqrsum(
        out_u, sqrsum_u, res_out_unfused.view(m, HC_HIDDEN), fn, tile_k
    )
    return out_u.sum(0), sqrsum_u.sum(0)


def _run_case(m: int):
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base = _make_inputs(m)

    # ---- Unfused path: mhc_post -> mhc_pre_gemm_sqrsum -> big_fuse (via mhc_pre)
    res_out_unfused = torch.empty_like(residual)
    aiter.mhc_post(res_out_unfused, x, residual, post_layer_mix, comb_res_mix)
    mul_u, sqr_u = _unfused_gemm_partials(res_out_unfused, fn)
    post_u, comb_u, layer_input_u = aiter.mhc_pre(
        res_out_unfused,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
    )

    # ---- Fused kernel: raw partials
    res_out_fused = torch.empty_like(residual)
    mul_f_split, sqr_f_split = mhc_post_gemm_sqrsum(
        res_out_fused, x, residual, post_layer_mix, comb_res_mix, fn
    )
    mul_f = mul_f_split.sum(0)  # (m, 24)
    sqr_f = sqr_f_split.sum(0)  # (m,)

    # ---- Fused end-to-end (post+gemm fused, then unchanged big_fuse)
    res_out_fused2, post_f, comb_f, layer_input_f = mhc_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
    )

    # ---- Pure-torch references
    res_ref = _post_ref(x, residual, post_layer_mix, comb_res_mix)
    res_ref_flat = res_ref.view(m, HC_HIDDEN).float()
    mul_ref = res_ref_flat @ fn.T
    sqr_ref = res_ref_flat.square().sum(-1)

    # ---------- Assertions / reporting ----------
    results = {}

    # (b) residual must be byte-for-byte identical (same post math path).
    res_match = torch.equal(res_out_fused, res_out_unfused)
    res_match2 = torch.equal(res_out_fused2, res_out_unfused)
    results["residual_bytematch_vs_unfused"] = res_match
    results["residual_bytematch_endtoend"] = res_match2

    # (a) gemm partials.
    results["mul_fused_vs_unfused"] = _rel_err(mul_u, mul_f)
    results["mul_fused_vs_ref"] = _rel_err(mul_ref, mul_f)
    results["sqr_fused_vs_unfused"] = _rel_err(sqr_u, sqr_f)
    results["sqr_fused_vs_ref"] = _rel_err(sqr_ref, sqr_f)

    # (c) end-to-end layer_input + mixes.
    results["layer_input_fused_vs_unfused"] = _rel_err(
        layer_input_u.float(), layer_input_f.float()
    )
    results["post_mix_fused_vs_unfused"] = _rel_err(post_u, post_f.squeeze(-1))
    results["comb_mix_fused_vs_unfused"] = _rel_err(comb_u, comb_f)

    # layer_input atol check (bf16-scale).
    li_atol = (layer_input_f.float() - layer_input_u.float()).abs().max().item()
    results["layer_input_max_abs"] = li_atol

    return results


def _check(results, m):
    print(f"\n=== Route B fused mhc_post_gemm_sqrsum numeric gate, M={m} ===")
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"  {k:38s} max_rel={v[0]:.3e}  mean_rel={v[1]:.3e}")
        else:
            print(f"  {k:38s} {v}")

    # Residual must be identical (the fused kernel must not change the residual).
    assert results["residual_bytematch_vs_unfused"], "fused residual != unfused residual"
    assert results["residual_bytematch_endtoend"], "end-to-end fused residual != unfused"

    # GEMM partials: divergence only from reduction order.
    GEMM_RTOL = 2e-2
    for key in ("mul_fused_vs_unfused", "mul_fused_vs_ref",
                "sqr_fused_vs_unfused", "sqr_fused_vs_ref"):
        max_rel, mean_rel = results[key]
        assert mean_rel < GEMM_RTOL, f"{key} mean_rel {mean_rel:.3e} >= {GEMM_RTOL}"

    # layer_input: tighter.
    LI_RTOL = 1e-2
    LI_ATOL = 5e-2
    max_rel, mean_rel = results["layer_input_fused_vs_unfused"]
    assert mean_rel < LI_RTOL, f"layer_input mean_rel {mean_rel:.3e} >= {LI_RTOL}"
    assert results["layer_input_max_abs"] < LI_ATOL or max_rel < LI_RTOL, (
        f"layer_input max_abs {results['layer_input_max_abs']:.3e} too large"
    )


@pytest.mark.parametrize("m", [8192, 4096, 1024, 512, 7, 1])
def test_mhc_post_gemm_fusion(m):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")
    results = _run_case(m)
    _check(results, m)


if __name__ == "__main__":
    for m in [8192, 4096, 1024, 512, 33, 7, 1]:
        results = _run_case(m)
        _check(results, m)
    print("\nAll Route B numeric-gate cases passed.")
