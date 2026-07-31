#!/usr/bin/env python3
"""Pack LoRA adapter safetensors into flat BF16 binary files for the runtime.

Usage: python pack_lora_adapter.py <adapter_dir> <output_dir>

Produces: output_dir/layer_00.bf16.bin ... layer_11.bf16.bin

Each file contains 14 matrices in canonical order:
  q_proj_A, q_proj_B, k_proj_A, k_proj_B, v_proj_A, v_proj_B,
  o_proj_A, o_proj_B, gate_proj_A, gate_proj_B,
  up_proj_A, up_proj_B, down_proj_A, down_proj_B

Scale (lora_alpha / rank) is folded into B before packing.
"""

import json
import os
import struct
import sys
from collections import defaultdict

import ml_dtypes
import numpy as np
from safetensors import safe_open

CANONICAL_ORDER = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_key(key):
    """Extract (layer_id, target, ab_kind) from a safetensors key."""
    parts = key.split(".")
    ab = None
    if "lora_A" in parts:
        ab = "A"
    elif "lora_B" in parts:
        ab = "B"
    else:
        return None

    layer_id = None
    target = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_id = int(parts[i + 1])
        if p in CANONICAL_ORDER:
            target = p
    if layer_id is None or target is None:
        return None
    return layer_id, target, ab


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <adapter_dir> <output_dir>")
        sys.exit(1)

    adapter_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")

    with open(config_path) as f:
        config = json.load(f)

    rank = int(config["r"])
    lora_alpha = float(config["lora_alpha"])
    scale = lora_alpha / rank
    print(f"rank={rank} alpha={lora_alpha} scale={scale}")

    # Index all tensors
    tensors = defaultdict(dict)  # (layer, target) -> {"A": np, "B": np}
    num_layers = 0
    bf16 = ml_dtypes.bfloat16

    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            parsed = parse_key(key)
            if parsed is None:
                continue
            layer_id, target, ab = parsed
            num_layers = max(num_layers, layer_id + 1)
            arr = f.get_tensor(key).float().numpy()
            tensors[(layer_id, target)][ab] = arr

    print(f"num_layers={num_layers}")

    for layer_id in range(num_layers):
        out_path = os.path.join(output_dir, f"layer_{layer_id:02d}.bf16.bin")
        chunks = []
        total_bytes = 0

        for target in CANONICAL_ORDER:
            ab = tensors.get((layer_id, target))
            if ab is None:
                print(f"  WARNING: layer {layer_id} target {target} missing A/B, skipping")
                continue
            A = ab["A"].astype(np.float32)
            B = ab["B"].astype(np.float32)
            # Fold scale into B
            B_scaled = (scale * B).astype(bf16)
            A_bf16 = A.astype(bf16)

            chunks.append(A_bf16.tobytes())
            chunks.append(B_scaled.tobytes())
            total_bytes += A_bf16.nbytes + B_scaled.nbytes

        with open(out_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
        print(f"  layer_{layer_id:02d}.bf16.bin: {total_bytes} bytes ({len(chunks)} tensors)")


if __name__ == "__main__":
    main()
