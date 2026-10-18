# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.models.config import DeepseekV4ForCausalLMConfig

EXPERT = "layers.3.ffn.experts.0.w1.weight"


def _write_checkpoint(path, tensors):
    save_file(tensors, str(path / "model-00001-of-00001.safetensors"))
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float8_e4m3fn, "fp8"),
        (torch.int8, "fp4"),
        (torch.bfloat16, None),
    ],
)
def test_infer_expert_dtype_from_routed_expert_weight(tmp_path, dtype, expected):
    _write_checkpoint(
        tmp_path,
        {
            "layers.3.attn.wq_a.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            EXPERT: torch.zeros(8, 8).to(dtype),
        },
    )

    inferred = DeepseekV4ForCausalLMConfig._infer_expert_dtype(str(tmp_path))

    assert inferred == expected


def test_infer_expert_dtype_without_routed_experts(tmp_path):
    _write_checkpoint(
        tmp_path, {"layers.0.attn.wq_a.weight": torch.zeros(8, 8, dtype=torch.bfloat16)}
    )

    assert DeepseekV4ForCausalLMConfig._infer_expert_dtype(str(tmp_path)) is None


def test_infer_expert_dtype_without_index(tmp_path):
    assert DeepseekV4ForCausalLMConfig._infer_expert_dtype(str(tmp_path)) is None
