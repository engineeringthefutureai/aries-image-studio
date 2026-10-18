"""Stand-ins for torch/diffusers/transformers, shared by the stage tests.

Stage modules import these at module level; installing one set of stand-ins (once, before
the first stage import) keeps every test file patching the same objects.
"""
import sys
from unittest.mock import MagicMock

STUBBED = (
    "diffusers",
    "diffusers.models",
    "diffusers.models.transformers",
    "diffusers.models.transformers.transformer_qwenimage21",
    "diffusers.pipelines",
    "diffusers.pipelines.qwenimage21",
    "diffusers.pipelines.qwenimage21.pipeline_qwenimage21",
    "diffusers.utils",
    "diffusers.utils.torch_utils",
    "transformers",
    "transformers.generation",
    "transformers.generation.streamers",
)


def install() -> MagicMock:
    torch = sys.modules.get("torch")
    if not isinstance(torch, MagicMock):
        torch = MagicMock()
        torch.xpu.is_available.return_value = False
        torch.xpu.device_count.return_value = 0
        sys.modules["torch"] = torch
        for name in STUBBED:
            sys.modules[name] = MagicMock()
    return torch
