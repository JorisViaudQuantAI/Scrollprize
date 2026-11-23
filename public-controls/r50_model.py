"""Verified weights and single-tile inference shared by both R50 entry points."""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


MODEL = "scrollprize/PHerc.1667-iteration-0"
REVISION = "06c306449b39df42c745608ceade243498d24243"
GIT_BLOBS = {
    "config.json": "32accd0facbc9424458e5d268460de0f07e6bd7d",
    "configuration_inkdetection.py": "24ac6d12668bfb5cf4b448c8f61723e97264ee25",
    "modeling_inkdetection.py": "c75143f25652f1e5fe9f7f7dc7b5183f00d37fb5",
}
WEIGHT_SHA = "62c43b98f55c169ae874c684a59953b5ce1cb45140129107f861501cce3edaef"
WIDTH = 256
DEPTH = 62


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_model_file(path, name):
    if not path.is_file():
        return False
    if name == "model.safetensors":
        return path.stat().st_size == 333531996 and file_sha256(path) == WEIGHT_SHA
    data = path.read_bytes()
    header = f"blob {len(data)}".encode() + bytes([0])
    return hashlib.sha1(header + data).hexdigest() == GIT_BLOBS[name]


def verified_cache(directory, *, download=False):
    directory = Path(directory)
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    for name in (*GIT_BLOBS, "model.safetensors"):
        final = directory / name
        if valid_model_file(final, name):
            continue
        if not download:
            raise ValueError(f"Incomplete or invalid model cache at {final}")
        import requests

        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as fp:
                temp = Path(fp.name)
                with requests.get(url, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        fp.write(chunk)
                fp.flush()
                os.fsync(fp.fileno())
            if not valid_model_file(temp, name):
                raise ValueError(f"Checksum mismatch for public model file {name}")
            temp.replace(final)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return directory.resolve()


def load_model(directory, *, download=False):
    cache = verified_cache(directory, download=download)
    return AutoModel.from_pretrained(str(cache), local_files_only=True,
                                     trust_remote_code=True).cpu().eval()


@torch.inference_mode()
def predict_tile(model, raw):
    if raw.shape != (DEPTH, WIDTH, WIDTH) or raw.dtype != np.uint8:
        raise ValueError(f"Expected uint8 {(DEPTH, WIDTH, WIDTH)}, got {raw.shape}/{raw.dtype}")
    tensor = torch.from_numpy(np.minimum(raw, 200).astype(np.float32)/255.0)[None, None]
    logits = model(tensor).logits
    if tuple(logits.shape) != (1, 1, 64, 64):
        raise ValueError(f"Unexpected output shape {tuple(logits.shape)}")
    probability = F.interpolate(logits.sigmoid(), size=(WIDTH, WIDTH), mode="bilinear",
                                align_corners=False)[0, 0].numpy()
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite model output")
    return probability
