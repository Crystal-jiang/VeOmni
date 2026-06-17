"""Bagel example dataset adapter.

Reads the upstream Bagel example data (t2i parquet + vlm jsonl) and yields samples
in the ``{task, prompt, response, image}`` format consumed by
``BagelSampleTransform``.

Register as ``bagel_unified`` in the VeOmni dataset registry so the training
config can reference ``train_path`` as a directory containing the example data.
"""

import io
import json
import os
from typing import Callable, Optional

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset

from ..dataset import DATASET_REGISTRY


class BagelUnifiedDataset(IterableDataset):
    """Interleaves VLM (understanding) and T2I (generation) samples from the
    upstream ``bagel_example`` directory layout.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed

    def _iter_vlm(self):
        vlm_dir = os.path.join(self.data_dir, "vlm")
        jsonl_path = os.path.join(vlm_dir, "llava_ov_si.jsonl")
        if not os.path.isfile(jsonl_path):
            return
        images_dir = os.path.join(vlm_dir, "images")
        with open(jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line)
                conversations = record.get("conversations", [])
                image_file = record.get("image", "")
                if not image_file or not conversations:
                    continue
                image_path = os.path.join(images_dir, image_file)
                if not os.path.isfile(image_path):
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    continue
                prompt_parts = []
                response = ""
                for turn in conversations:
                    role = turn.get("from", "")
                    value = turn.get("value", "").replace("<image>", "").strip()
                    if role == "human":
                        prompt_parts.append(value)
                    elif role == "gpt":
                        response = value
                        break
                if not response:
                    continue
                sample = {
                    "task": "understanding",
                    "prompt": " ".join(prompt_parts) if prompt_parts else "Describe this image.",
                    "response": response,
                    "image": image,
                }
                if self.transform is not None:
                    result = self.transform(sample)
                    if result is not None:
                        yield result
                else:
                    yield sample

    def _iter_t2i(self):
        t2i_dir = os.path.join(self.data_dir, "t2i")
        if not os.path.isdir(t2i_dir):
            return
        for fname in sorted(os.listdir(t2i_dir)):
            if not fname.endswith(".parquet"):
                continue
            table = pq.read_table(os.path.join(t2i_dir, fname))
            columns = table.column_names
            for i in range(len(table)):
                row = {col: table.column(col)[i].as_py() for col in columns}
                image_bytes = row.get("image")
                captions = row.get("captions")
                if image_bytes is None or captions is None:
                    continue
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    continue
                if isinstance(captions, str):
                    try:
                        parsed = json.loads(captions)
                        prompt = parsed.get("caption", captions) if isinstance(parsed, dict) else captions
                    except (json.JSONDecodeError, TypeError):
                        prompt = captions
                elif isinstance(captions, list):
                    prompt = captions[0] if captions else ""
                else:
                    prompt = str(captions)
                if not prompt:
                    continue
                sample = {
                    "task": "generation",
                    "prompt": prompt,
                    "image": image,
                }
                if self.transform is not None:
                    result = self.transform(sample)
                    if result is not None:
                        yield result
                else:
                    yield sample

    def __iter__(self):
        import random

        vlm_samples = list(self._iter_vlm())
        t2i_samples = list(self._iter_t2i())

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(vlm_samples)
            rng.shuffle(t2i_samples)

        combined = []
        vi, ti = 0, 0
        while vi < len(vlm_samples) or ti < len(t2i_samples):
            if vi < len(vlm_samples):
                combined.append(vlm_samples[vi])
                vi += 1
            if ti < len(t2i_samples):
                combined.append(t2i_samples[ti])
                ti += 1

        if self.shuffle:
            buffer = []
            for sample in combined:
                buffer.append(sample)
                if len(buffer) >= 64:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            if buffer:
                rng.shuffle(buffer)
                yield from buffer
        else:
            yield from combined


@DATASET_REGISTRY.register("bagel_unified")
def build_bagel_unified_dataset(
    train_path: str,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    seed: int = 42,
    **kwargs,
):
    return BagelUnifiedDataset(data_dir=train_path, transform=transform, shuffle=shuffle, seed=seed)
