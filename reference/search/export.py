"""Exports SigLIP 2 for Studio's search (docs/SEARCH.md): the image and text encoders as
ONNX, the tokenizer, a manifest, and the ONNX Runtime library Studio loads them with.

    python export.py OUT_DIR

The image encoder stays f32 (int8 moved some image embeddings to 0.57 cosine of f32); the
text encoder's 256k-token embedding table is stored as int8 (it is only looked up: query
embeddings stay within 0.997 cosine), which halves the text encoder. Studio's Dockerfile
runs this in a build stage; the model downloads at build time, never at runtime.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnxruntime
import torch
from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel

MODEL = "google/siglip2-base-patch16-256"
REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
IMAGE_SIZE = 256
MAX_TOKENS = 64


def features(out):
    # transformers 5 returns a model output; older versions the tensor.
    return out if torch.is_tensor(out) else out.pooler_output


class Vision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return features(self.model.get_image_features(pixel_values=pixel_values))


class Text(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return features(self.model.get_text_features(input_ids=input_ids))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.model"]))
    model = AutoModel.from_pretrained(src).eval()

    pixels = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    tokens = torch.zeros(1, MAX_TOKENS, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            Vision(model), (pixels,), out / "vision.onnx",
            input_names=["pixel_values"], output_names=["embeds"],
            dynamic_axes={"pixel_values": {0: "n"}, "embeds": {0: "n"}}, opset_version=17, dynamo=False,
        )
        torch.onnx.export(
            Text(model), (tokens,), out / "text.f32.onnx",
            input_names=["input_ids"], output_names=["embeds"],
            dynamic_axes={"input_ids": {0: "n"}, "embeds": {0: "n"}}, opset_version=17, dynamo=False,
        )
    quantize_dynamic(out / "text.f32.onnx", out / "text.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    (out / "text.f32.onnx").unlink()
    shutil.copy(src / "tokenizer.json", out / "tokenizer.json")

    lib = next((Path(onnxruntime.__file__).parent / "capi").glob("libonnxruntime.so.*"))
    shutil.copy(lib, out / "libonnxruntime.so")

    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "image_size": IMAGE_SIZE,
        "max_tokens": MAX_TOKENS,
        "dim": model.config.vision_config.hidden_size,
        "logit_scale": model.logit_scale.exp().item(),
        "logit_bias": model.logit_bias.item(),
        "onnxruntime": onnxruntime.__version__,
        "files": {f: sha256(out / f) for f in ("vision.onnx", "text.onnx", "tokenizer.json")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
