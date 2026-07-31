#!/usr/bin/env python3
"""Pack a validated PEFT LoRA adapter into per-layer BF16 runtime files.

Usage: python pack_lora_adapter.py <adapter_dir> <output_dir>

The packer rejects every tensor or PEFT behavior outside the first-version
contract.  It folds ``lora_alpha / rank`` into B, writes exactly 14 tensors per
layer in canonical order, and emits ``manifest.json`` only after all packed
files have been produced successfully.
"""

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import ml_dtypes
import numpy as np
from safetensors import safe_open


PACKER_VERSION = "0.2"
CANONICAL_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
TENSOR_ORDER = [f"{target}_{ab}" for target in CANONICAL_ORDER for ab in ("A", "B")]
REQUIRED_FALSE = ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias", "use_qalora")
REQUIRED_EMPTY = (
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
    "exclude_modules",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "megatron_config",
    "trainable_token_indices",
    "eva_config",
    "corda_config",
    "alora_invocation_tokens",
    "target_parameters",
)


class PackError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key(key):
    """Extract (layer_id, target, ab_kind) from a supported tensor key."""
    parts = key.split(".")
    markers = [ab for ab in ("A", "B") if f"lora_{ab}" in parts]
    if len(markers) != 1:
        return None

    layer_ids = []
    targets = []
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_ids.append(int(parts[i + 1]))
        if part in CANONICAL_ORDER:
            targets.append(part)
    if len(layer_ids) != 1 or len(targets) != 1:
        return None
    return layer_ids[0], targets[0], markers[0]


def validate_config(config):
    if config.get("peft_type") != "LORA":
        raise PackError(f"unsupported peft_type={config.get('peft_type')!r}")
    if config.get("bias", "none") != "none":
        raise PackError(f"unsupported bias={config.get('bias')!r}")
    for key in REQUIRED_FALSE:
        if config.get(key, False):
            raise PackError(f"unsupported {key}={config.get(key)!r}; expected false")
    for key in REQUIRED_EMPTY:
        value = config.get(key)
        if value not in (None, [], {}):
            raise PackError(f"unsupported {key}={value!r}; expected null or empty")
    if config.get("qalora_group_size") is not None:
        raise PackError("qalora_group_size requires unsupported QALoRA behavior")

    targets = config.get("target_modules")
    if not isinstance(targets, (list, tuple, set)) or set(targets) != set(CANONICAL_ORDER):
        raise PackError(
            f"target_modules must be exactly {CANONICAL_ORDER}; got {targets!r}"
        )

    try:
        rank = int(config["r"])
        alpha = float(config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PackError(f"invalid r/lora_alpha: {exc}") from exc
    if rank <= 0:
        raise PackError(f"rank must be positive, got {rank}")
    if not math.isfinite(alpha) or alpha <= 0:
        raise PackError(f"lora_alpha must be finite and positive, got {alpha}")
    return rank, alpha


def load_and_validate_tensors(weights_path, rank):
    tensors = {}
    layers = set()
    shapes = {}

    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            parsed = parse_key(key)
            if parsed is None:
                raise PackError(f"unrecognized or unsupported adapter tensor: {key}")
            if parsed in tensors:
                raise PackError(f"duplicate tensor slot {parsed}: {tensors[parsed][0]} and {key}")

            tensor = f.get_tensor(key)
            if str(tensor.dtype) != "torch.bfloat16":
                raise PackError(f"tensor {key} must be BF16, got {tensor.dtype}")
            if tensor.ndim != 2:
                raise PackError(f"tensor {key} must be 2-D, got shape {tuple(tensor.shape)}")

            layer, target, ab = parsed
            shape = tuple(int(v) for v in tensor.shape)
            if ab == "A" and shape[0] != rank:
                raise PackError(f"tensor {key} A rows {shape[0]} != rank {rank}")
            if ab == "B" and shape[1] != rank:
                raise PackError(f"tensor {key} B cols {shape[1]} != rank {rank}")

            tensors[parsed] = (key, tensor.float().numpy())
            layers.add(layer)

    if not layers:
        raise PackError("adapter contains no supported LoRA tensors")
    expected_layers = set(range(max(layers) + 1))
    if layers != expected_layers:
        raise PackError(f"layer IDs must be contiguous from 0; got {sorted(layers)}")

    for layer in sorted(expected_layers):
        for target in CANONICAL_ORDER:
            for ab in ("A", "B"):
                slot = (layer, target, ab)
                if slot not in tensors:
                    raise PackError(f"missing tensor for layer={layer} target={target} {ab}")
            a_shape = tuple(tensors[(layer, target, "A")][1].shape)
            b_shape = tuple(tensors[(layer, target, "B")][1].shape)
            if target not in shapes:
                shapes[target] = (a_shape, b_shape)
            elif shapes[target] != (a_shape, b_shape):
                raise PackError(
                    f"shape mismatch for target={target} layer={layer}: "
                    f"A/B={(a_shape, b_shape)} expected={shapes[target]}"
                )

    expected_count = len(expected_layers) * len(CANONICAL_ORDER) * 2
    if len(tensors) != expected_count:
        raise PackError(f"expected exactly {expected_count} tensors, got {len(tensors)}")
    return tensors, len(expected_layers), shapes


def write_atomic(path, chunks):
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def pack(adapter_dir, output_dir):
    adapter_dir = os.path.abspath(adapter_dir)
    output_dir = os.path.abspath(output_dir)
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(config_path):
        raise PackError(f"missing {config_path}")
    if not os.path.isfile(weights_path):
        raise PackError(f"missing {weights_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    rank, alpha = validate_config(config)
    scale = alpha / rank
    tensors, num_layers, shapes = load_and_validate_tensors(weights_path, rank)

    parent = os.path.dirname(output_dir) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".lora-pack-", dir=parent)
    files = {}
    bf16 = ml_dtypes.bfloat16
    try:
        for layer in range(num_layers):
            filename = f"layer_{layer:02d}.bf16.bin"
            out_path = os.path.join(staging, filename)
            chunks = []
            for target in CANONICAL_ORDER:
                a = tensors[(layer, target, "A")][1].astype(np.float32)
                b = tensors[(layer, target, "B")][1].astype(np.float32)
                chunks.append(a.astype(bf16).tobytes())
                chunks.append((scale * b).astype(bf16).tobytes())
            write_atomic(out_path, chunks)
            files[filename] = {
                "size": os.path.getsize(out_path),
                "sha256": sha256_file(out_path),
            }

        manifest = {
            "format_version": 1,
            "scale_folded_into_b": True,
            "rank": rank,
            "lora_alpha": alpha,
            "scale": scale,
            "num_layers": num_layers,
            "target_modules": CANONICAL_ORDER,
            "tensor_order": TENSOR_ORDER,
            "source_dtype": "BF16",
            "packed_dtype": "BF16",
            "ab_shapes": {
                target: [list(a_shape), list(b_shape)]
                for target, (a_shape, b_shape) in shapes.items()
            },
            "adapter_id": os.path.basename(os.path.normpath(adapter_dir)),
            "base_model_id": config.get("base_model_name_or_path", ""),
            "adapter_config_sha256": sha256_file(config_path),
            "source_safetensors_sha256": sha256_file(weights_path),
            "packer_version": PACKER_VERSION,
            "files": files,
        }
        manifest_path = os.path.join(staging, "manifest.json")
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        write_atomic(manifest_path, [manifest_bytes])

        os.makedirs(output_dir, exist_ok=True)
        for filename in sorted(files):
            os.replace(os.path.join(staging, filename), os.path.join(output_dir, filename))
        os.replace(manifest_path, os.path.join(output_dir, "manifest.json"))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"rank={rank} alpha={alpha} scale={scale} num_layers={num_layers}")
    for filename in sorted(files):
        info = files[filename]
        print(f"  {filename}: {info['size']} bytes (14 tensors) sha256={info['sha256']}")
    print(f"  manifest.json: {os.path.join(output_dir, 'manifest.json')}")
    return manifest


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <adapter_dir> <output_dir>", file=sys.stderr)
        return 2
    try:
        pack(sys.argv[1], sys.argv[2])
    except (PackError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
