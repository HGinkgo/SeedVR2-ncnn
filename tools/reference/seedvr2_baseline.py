"""Small, reproducible reference artifacts for the SeedVR2 port.

The AWA implementation mirrors the frozen upstream window ordering. The VAE
command intentionally loads the upstream implementation from a separate
checkout selected by ``SEEDVR2_UPSTREAM_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

Window = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _triple(value: Sequence[int], name: str) -> Tuple[int, int, int]:
    if len(value) != 3 or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} dimensions must be positive triples")
    return tuple(int(item) for item in value)


def make_windows(
    size: Sequence[int], num_windows: Sequence[int], shifted: bool = False
) -> List[Window]:
    """Return upstream-compatible clipped AWA windows in partition order."""

    t, h, w = _triple(size, "size")
    resized_nt, resized_nh, resized_nw = _triple(num_windows, "num_windows")
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh = math.ceil(resized_h / resized_nh)
    ww = math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)

    if not shifted:
        st = sh = sw = 0.0
        nt = math.ceil(t / wt)
        nh = math.ceil(h / wh)
        nw = math.ceil(w / ww)
    else:
        st = 0.5 if wt < t else 0.0
        sh = 0.5 if wh < h else 0.0
        sw = 0.5 if ww < w else 0.0
        nt = math.ceil((t - st) / wt) + (1 if st > 0 else 0)
        nh = math.ceil((h - sh) / wh) + (1 if sh > 0 else 0)
        nw = math.ceil((w - sw) / ww) + (1 if sw > 0 else 0)

    windows: List[Window] = []
    for iw in range(nw):
        for ih in range(nh):
            for it in range(nt):
                starts = (
                    max(int((it - st) * wt), 0),
                    max(int((ih - sh) * wh), 0),
                    max(int((iw - sw) * ww), 0),
                )
                ends = (
                    min(int((it - st + 1) * wt), t),
                    min(int((ih - sh + 1) * wh), h),
                    min(int((iw - sw + 1) * ww), w),
                )
                if all(end > start for start, end in zip(starts, ends)):
                    windows.append(tuple(zip(starts, ends)))  # type: ignore[arg-type]
    return windows


def _index_maps(size: Sequence[int], windows: Iterable[Window]) -> Tuple[torch.Tensor, torch.Tensor]:
    t, h, w = _triple(size, "size")
    source = torch.arange(t * h * w, dtype=torch.long).reshape(t, h, w)
    target = torch.cat(
        [source[s0[0] : s0[1], s1[0] : s1[1], s2[0] : s2[1]].reshape(-1) for s0, s1, s2 in windows]
    )
    reverse = torch.argsort(target)
    return target, reverse


def awa_round_trip(
    source: torch.Tensor, num_windows: Sequence[int], shifted: bool = False
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Partition and reverse a ``(T,H,W,C)`` tensor using upstream AWA order."""

    if source.ndim != 4:
        raise ValueError("source must have shape (T,H,W,C)")
    windows = make_windows(source.shape[:3], num_windows, shifted=shifted)
    target, reverse = _index_maps(source.shape[:3], windows)
    flat = source.contiguous().reshape(-1, source.shape[-1])
    partitioned = flat.index_select(0, target)
    restored = partitioned.index_select(0, reverse).reshape_as(source)
    metadata = {
        "source_shape": list(source.shape),
        "num_windows": [int(item) for item in num_windows],
        "shifted": bool(shifted),
        "window_count": len(windows),
        "windows": [[list(axis) for axis in window] for window in windows],
        "partitioned_shape": list(partitioned.shape),
        "target_index_sha256": _tensor_sha256(target),
        "reverse_index_sha256": _tensor_sha256(reverse),
    }
    return restored, metadata


def make_reference_input(size: Sequence[int], seed: int = 666) -> torch.Tensor:
    """Create a deterministic normalized ``(C,T,H,W)`` input for VAE tests."""

    t, h, w = _triple(size, "size")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand((3, t, h, w), generator=generator, dtype=torch.float32).mul(2.0).sub(1.0)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_tensor(path: Path, tensor: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().contiguous(), path)
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(paths: Mapping[str, Path], metadata: Mapping[str, object]) -> Dict[str, object]:
    artifacts: Dict[str, object] = {}
    for name, path in paths.items():
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        artifacts[name] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return {"metadata": dict(metadata), "artifacts": artifacts}


def _parse_triple(value: str) -> Tuple[int, int, int]:
    try:
        return _triple([int(item.strip()) for item in value.split(",")], "argument")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _run_awa(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    source = torch.arange(args.channels * math.prod(args.size), dtype=torch.float32).reshape(
        *args.size, args.channels
    )
    windows = make_windows(args.size, args.windows, shifted=args.shifted)
    target, _ = _index_maps(args.size, windows)
    partitioned = source.reshape(-1, args.channels).index_select(0, target)
    restored, metadata = awa_round_trip(source, args.windows, shifted=args.shifted)
    paths = {
        "source": output_dir / "awa_source.pt",
        "partitioned": output_dir / "awa_partitioned.pt",
        "restored": output_dir / "awa_restored.pt",
    }
    save_tensor(paths["source"], source)
    save_tensor(paths["partitioned"], partitioned)
    save_tensor(paths["restored"], restored)
    manifest = build_manifest(paths, {"kind": "awa", "seed": 0, **metadata})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _load_reference_vae(upstream_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(upstream_root))
    previous_cwd = Path.cwd()
    os.chdir(upstream_root)
    try:
        from common.config import create_object, load_config

        config = load_config("configs_3b/main.yaml")
        config.vae.checkpoint = str(checkpoint)
        vae = create_object(config.vae.model)
        vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        state = torch.load(checkpoint, map_location=device, mmap=True, weights_only=False)
        loading_info = vae.load_state_dict(state, strict=True)
        if hasattr(vae, "set_causal_slicing"):
            vae.set_causal_slicing(**config.vae.slicing)
        if hasattr(vae, "set_memory_limit"):
            vae.set_memory_limit(**config.vae.memory_limit)
        return vae, config, str(loading_info)
    finally:
        os.chdir(previous_cwd)


def _run_vae(args: argparse.Namespace) -> None:
    upstream_root = Path(os.environ.get("SEEDVR2_UPSTREAM_ROOT", "")).expanduser()
    checkpoint_dir = Path(os.environ.get("SEEDVR2_CKPT_DIR", "ckpts")).expanduser()
    if not upstream_root.is_dir():
        raise SystemExit("SEEDVR2_UPSTREAM_ROOT must point to the frozen SeedVR source checkout")
    checkpoint = checkpoint_dir / "ema_vae.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"missing VAE checkpoint: {checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("VAE baseline requires a CUDA-enabled PyTorch environment")

    owns_process_group = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
        owns_process_group = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = make_reference_input(args.size, seed=args.seed)
    device = torch.device("cuda")
    vae, config, loading_info = _load_reference_vae(upstream_root, checkpoint, device)
    dtype = getattr(torch, config.vae.dtype)
    scale = float(config.vae.scaling_factor)
    sample = source.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        encoded = vae.encode(vae.preprocess(sample))
        latent = encoded.posterior.mode().squeeze(2)
        latent = (latent - float(config.vae.get("shifting_factor", 0.0))) * scale
        latent_cthw = latent.unsqueeze(2) if latent.ndim == 4 else latent
        latent_thwc = latent_cthw.permute(0, 2, 3, 4, 1).squeeze(0).float().cpu()

        decode_latent = latent / scale + float(config.vae.get("shifting_factor", 0.0))
        reconstruction = vae.decode(decode_latent).sample
        reconstruction_cthw = reconstruction.unsqueeze(2) if reconstruction.ndim == 4 else reconstruction
        reconstruction_cthw = reconstruction_cthw.float().cpu().squeeze(0)

    paths = {
        "input": output_dir / "vae_input.pt",
        "latent": output_dir / "vae_latent.pt",
        "reconstruction": output_dir / "vae_reconstruction.pt",
    }
    save_tensor(paths["input"], source)
    save_tensor(paths["latent"], latent_thwc)
    save_tensor(paths["reconstruction"], reconstruction_cthw)
    metadata = {
        "kind": "vae",
        "seed": int(args.seed),
        "input_shape": list(source.shape),
        "dtype": str(dtype),
        "scaling_factor": scale,
        "upstream_root": str(upstream_root),
        "upstream_commit": os.environ.get("SEEDVR2_UPSTREAM_COMMIT", "unknown"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "loading_info": loading_info,
    }
    manifest = build_manifest(paths, metadata)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if owns_process_group:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    awa = subparsers.add_parser("awa", help="write a deterministic AWA partition baseline")
    awa.add_argument("--size", type=_parse_triple, default=(1, 45, 80))
    awa.add_argument("--windows", type=_parse_triple, default=(4, 3, 3))
    awa.add_argument("--channels", type=int, default=4)
    awa.add_argument("--shifted", action="store_true")
    awa.add_argument("--output-dir", required=True)
    vae = subparsers.add_parser("vae", help="write a deterministic VAE encode/decode baseline")
    vae.add_argument("--size", type=_parse_triple, default=(1, 128, 128))
    vae.add_argument("--seed", type=int, default=666)
    vae.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "awa":
        if args.channels <= 0:
            parser.error("--channels must be positive")
        _run_awa(args)
    elif args.command == "vae":
        _run_vae(args)


if __name__ == "__main__":
    main()
