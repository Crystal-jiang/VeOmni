#!/usr/bin/env python3
"""Convert ImageNet-1K HuggingFace parquet to VeOmni multi-shard parquet.

Usage:
    python scripts/prepare_imagenet1k.py \
        --input /home/usr/data/imagenet-1k/data \
        --classes /home/usr/data/imagenet-1k/classes.py \
        --output /home/dataset/imagenet1k_train \
        --num_shards 10

Input HF parquet schema:
    image: struct{bytes: binary, path: string}
    label: int64

Output parquet schema:
    conversations: str  (comma-separated class labels, e.g. "tabby, tabby cat")
    images: list[str]  (absolute paths to saved JPEG files)
"""

import argparse
import importlib.util
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def load_classes(classes_path):
    spec = importlib.util.spec_from_file_location("classes", classes_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = list(mod.IMAGENET2012_CLASSES.values())
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing train-*.parquet files")
    parser.add_argument("--classes", required=True, help="Path to classes.py with IMAGENET2012_CLASSES")
    parser.add_argument("--output", required=True, help="Output directory for converted parquet shards")
    parser.add_argument("--num_shards", type=int, default=10, help="Number of output shards")
    args = parser.parse_args()

    classes = load_classes(args.classes)
    print(f"Loaded {len(classes)} classes")

    input_files = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.startswith("train-") and f.endswith(".parquet")
    ])
    print(f"Found {len(input_files)} input shards")

    image_dir = os.path.join(os.path.dirname(args.output.rstrip("/")), "imagenet1k_images")
    os.makedirs(image_dir, exist_ok=True)

    all_conversations = []
    all_images = []
    for i, fpath in enumerate(input_files):
        table = pq.read_table(fpath)
        df = table.to_pandas()
        for idx, row in df.iterrows():
            label_name = classes[row["label"]]
            img_bytes = row["image"]["bytes"]
            img_path = row["image"]["path"]
            save_name = img_path.replace("/", "_")
            if not save_name.lower().endswith((".jpg", ".jpeg", ".png")):
                save_name += ".JPEG"
            save_path = os.path.join(image_dir, save_name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
            all_conversations.append(label_name)
            all_images.append([save_path])
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(input_files)} shards ({len(all_conversations)} samples)")

    print(f"Total samples: {len(all_conversations)}")

    os.makedirs(args.output, exist_ok=True)
    for f in os.listdir(args.output):
        if f.endswith(".parquet"):
            os.remove(os.path.join(args.output, f))

    shard_size = len(all_conversations) // args.num_shards
    for shard_idx in range(args.num_shards):
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < args.num_shards - 1 else len(all_conversations)

        table = pa.table({
            "conversations": all_conversations[start:end],
            "images": all_images[start:end],
        })

        output_path = os.path.join(args.output, f"train-{shard_idx:05d}-of-{args.num_shards:05d}.parquet")
        pq.write_table(table, output_path)
        print(f"Wrote {output_path}: {end - start} samples")

    print(f"Done. {args.num_shards} shards in {args.output}")


if __name__ == "__main__":
    main()
