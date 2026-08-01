// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <string_view>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_bpreshuffle_common.cuh"
#include "gemm_a8w8_blockscale_bpreshuffle_lookup.h"
#include "gemm_a8w8_blockscale_bpreshuffle_manifest.h"

using BlockwiseKernel = torch::Tensor (*)(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&);

// Name-keyed dispatch table; see gemm_a8w8_blockscale.cu for the rationale
// behind std::string_view keys + raw fn-ptr values (constant-init into
// .data.rel.ro, matching PR #3255's GemmDispatchMap style).
using BlockwiseKernelMap = std::unordered_map<std::string_view, BlockwiseKernel>;

// Python-driven name-keyed dispatch (see gemm_a8w8_blockscale.cu for the
// rationale).  Empty kernelName -> default heuristic; non-empty but unknown
// kernelName -> hard error.
template <typename DDataType, typename EDataType = DDataType>
BlockwiseKernel blockscale_bpreshuffle_dispatch(const std::string& kernelName)
{
    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, F16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, F16)};
        }
        else if constexpr(std::is_same_v<EDataType, B16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, B16)};
        }
        else
        {
            static_assert(false, "blockscale_bpreshuffle_dispatch used with unsupported dtype!");
        }
    }();

    if(!kernelName.empty())
    {
        auto it = lookup.find(std::string_view{kernelName});
        if(it != lookup.end())
        {
            return it->second;
        }
        TORCH_CHECK(false,
                    "gemm_a8w8_blockscale_bpreshuffle kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv) and try "
                    "again.");
    }

    // Default heuristic kernel (used when Python had no tuned row).
    return a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale_bpreshuffle(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& w_scale,
                                               torch::Tensor& Y,
                                               std::string kernelName)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_bpreshuffle_dispatch<F32, F16>(kernelName)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_bpreshuffle_dispatch<F32, B16>(kernelName)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}

torch::Tensor gemm_a8w8_mixedscale_bpreshuffle(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& b_block_scale,
                                               torch::Tensor& w_channel_scale,
                                               torch::Tensor& Y)
{
    const int M = XQ.size(0);
    const int N = WQ.size(0);
    const int K = XQ.size(1);

    TORCH_CHECK(K == 4096 && WQ.size(1) == K,
                "mixed-scale XDL GEMM currently supports K=4096");
    TORCH_CHECK(x_scale.numel() == M * (K / 128),
                "x_scale must contain M*K/128 elements");
    TORCH_CHECK(b_block_scale.sizes() == torch::IntArrayRef({N / 128, K / 128}),
                "b_block_scale must have shape [N/128, K/128]");
    TORCH_CHECK(w_channel_scale.sizes() == torch::IntArrayRef({N, 1}),
                "w_channel_scale must have shape [N, 1]");
    TORCH_CHECK(Y.sizes() == torch::IntArrayRef({M, N}), "Y must have shape [M, N]");

    using MixedGemm = DeviceGemmHelperF8MixedScaleBPreshuffle<
        F32,
        B16,
        512,
        16,
        64,
        256,
        16,
        16,
        16,
        16,
        1,
        1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        1,
        1,
        S<1, 16, 1, 16>,
        S<4>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        ck::tensor_operation::device::GemmSpecialization::MNKPadding>;

    return gemm_a8w8_mixedscale_bpreshuffle_impl<F32, B16, MixedGemm>(
        XQ, WQ, x_scale, b_block_scale, w_channel_scale, Y);
}
