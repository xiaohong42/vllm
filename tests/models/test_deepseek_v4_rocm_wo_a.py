# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.amd.rocm import _wo_a_block_scale_to_e8m0
from vllm.platforms import current_platform


def test_wo_a_block_scale_to_e8m0_from_float():
    scale = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)

    encoded = _wo_a_block_scale_to_e8m0(scale)

    assert encoded is not None
    torch.testing.assert_close(
        encoded,
        torch.tensor([[126, 127, 128, 129]], dtype=torch.uint8),
    )
    assert encoded.is_contiguous()


def test_wo_a_block_scale_to_e8m0_preserves_encoded_scales():
    raw = torch.tensor([[125, 127, 131]], dtype=torch.uint8)
    encoded = raw.view(torch.float8_e8m0fnu)

    converted = _wo_a_block_scale_to_e8m0(encoded)

    assert converted is not None
    torch.testing.assert_close(converted, raw)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm DeepSeek V4 model")
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.uint8])
def test_quantize_block_fp8_matches_v4_checkpoint_layout(scale_dtype: torch.dtype):
    from vllm.models.deepseek_v4.amd.model import _quantize_block_fp8

    torch.manual_seed(0)
    weight = torch.randn(256, 384, dtype=torch.bfloat16)
    scale_param = torch.empty(2, 3, dtype=scale_dtype)

    fp8, scale = _quantize_block_fp8(weight, [128, 128], scale_param)

    assert fp8.dtype == current_platform.fp8_dtype()
    assert fp8.shape == weight.shape
    assert scale.dtype == scale_dtype and scale.shape == (2, 3)
    if scale_dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.float()
    # ue8m0: every block scale is an exact power of two.
    torch.testing.assert_close(scale, torch.exp2(torch.log2(scale).round()))
    dequant = fp8.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    torch.testing.assert_close(
        dequant, weight.float().to(dequant.device), rtol=0.07, atol=0.02
    )


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[-1.0, 1.0]]),
        torch.tensor([[0.75, 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.ones(1, dtype=torch.int32),
    ],
)
def test_wo_a_block_scale_to_e8m0_rejects_invalid_scales(scale: torch.Tensor):
    assert _wo_a_block_scale_to_e8m0(scale) is None
