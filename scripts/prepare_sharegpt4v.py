#!/usr/bin/env python3
"""Convert ShareGPT4V JSON to VeOmni multi-shard parquet.

Usage:
    python scripts/prepare_sharegpt4v.py \
        --input /home/dataset/sharegpt4v_instruct_gpt4-vision_cap100k.json \
        --image_root /home/dataset \
        --output /home/dataset/sharegpt4v_cap_100k \
        --num_shards 6

Input JSON schema:
    {"id": str, "image": "coco/train2017/xxx.jpg", "conversations": [{"from": "human"|"gpt", "value": str}]}

Output parquet schema:
    conversations: list[struct{from: str, value: str}]
    images: list[str]  (absolute paths)
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--image_root", required=True, help="Root directory for resolving relative image paths")
    parser.add_argument("--output", required=True, help="Output directory for parquet shards")
    parser.add_argument("--num_shards", type=int, default=6, help="Number of output shards")
    parser.add_argument("--filter", default="coco", help="Only keep samples whose image path starts with this prefix (empty = keep all)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.input}")

    filtered = []
    skipped = 0
    for item in data:
        img = item.get("image", "")
        if args.filter and not img.startswith(args.filter):
            continue
        abs_path = os.path.join(args.image_root, img)
        if not os.path.exists(abs_path):
            skipped += 1
            continue
        filtered.append({
            "conversations": item["conversations"],
            "images": [abs_path],
        })

    print(f"Kept {len(filtered)} samples, skipped {skipped} (missing images)")

    os.makedirs(args.output, exist_ok=True)

    for f in os.listdir(args.output):
        if f.endswith(".parquet") or f.endswith(".json"):
            os.remove(os.path.join(args.output, f))

    shard_size = len(filtered) // args.num_shards
    for shard_idx in range(args.num_shards):
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < args.num_shards - 1 else len(filtered)
        shard_data = filtered[start:end]

        table = pa.table({
            "conversations": [d["conversations"] for d in shard_data],
            "images": [d["images"] for d in shard_data],
        })

        output_path = os.path.join(args.output, f"train-{shard_idx:05d}-of-{args.num_shards:05d}.parquet")
        pq.write_table(table, output_path)
        print(f"Wrote {output_path}: {len(shard_data)} samples")

    print(f"Done. {args.num_shards} shards in {args.output}")


if __name__ == "__main__":
    main()
