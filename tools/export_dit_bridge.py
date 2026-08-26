"""Export the fixed VAE -> DiT block -> VAE bridge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.export_dit_block import rewrite_ncnn_param
from tools.reference.dit_block import FixedDitBlock
from tools.reference.dit_bridge import FixedDitInputProjection, FixedDitOutputProjection
from tools.reference.dit_pipeline import DIT_PIPELINE_CONTRACT


def _trace_save(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], path: Path) -> None:
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs, check_trace=True)
    traced.save(str(path))


def _run_pnnx(pnnx: Path, model_path: Path, output_dir: Path, input_shape: str) -> None:
    subprocess.run(
        [str(pnnx.resolve()), str(model_path.resolve()), f"inputshape={input_shape}"],
        cwd=output_dir.resolve(),
        check=True,
    )


def export_bridge(checkpoint: Path, output_dir: Path, pnnx: Path | None, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    contract = DIT_PIPELINE_CONTRACT

    patch_in = FixedDitInputProjection(checkpoint).float().eval()
    patches = torch.randn(contract.video_tokens, 132)
    text = torch.randn(contract.text_tokens, contract.text_input_dim)
    _trace_save(patch_in, (patches, text), output_dir / "dit_patch_in.pt")

    block = FixedDitBlock(source_shape=contract.source_shape, text_tokens=contract.text_tokens).float().eval()
    from tools.reference.dit_block import load_official_block_weights

    load_official_block_weights(block, checkpoint, block_index=0)
    block_input = torch.randn(contract.video_tokens, 2560)
    block_text = torch.randn(contract.text_tokens, 2560)
    embedding = torch.randn(1, contract.embedding_dim)
    _trace_save(block, (block_input, block_text, embedding), output_dir / "dit_block0.pt")

    patch_out = FixedDitOutputProjection(checkpoint).float().eval()
    output_input = torch.randn(contract.video_tokens, 2560)
    _trace_save(patch_out, (output_input,), output_dir / "dit_patch_out.pt")

    manifest_path = output_dir / "dit_bridge_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "fixed_vae_dit_block0_bridge",
                "source_shape": list(contract.source_shape),
                "window_shape": [4, 3, 3],
                "text_tokens": contract.text_tokens,
                "artifacts": {
                    "patch_in": {
                        "model": str((output_dir / "dit_patch_in.pt").resolve()),
                        "inputs": [[contract.video_tokens, 132], list(contract.text_shape)],
                        "outputs": [[contract.video_tokens, 2560], [contract.text_tokens, 2560]],
                    },
                    "block0": {
                        "model": str((output_dir / "dit_block0.pt").resolve()),
                        "inputs": [[contract.video_tokens, 2560], [contract.text_tokens, 2560], [1, contract.embedding_dim]],
                        "outputs": [[contract.video_tokens, 2560], [contract.text_tokens, 2560]],
                    },
                    "patch_out": {
                        "model": str((output_dir / "dit_patch_out.pt").resolve()),
                        "inputs": [[contract.video_tokens, 2560]],
                        "outputs": [[contract.video_tokens, 64]],
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if pnnx is None:
        return

    _run_pnnx(pnnx, output_dir / "dit_patch_in.pt", output_dir, "[64,132],[58,5120]")
    _run_pnnx(pnnx, output_dir / "dit_block0.pt", output_dir, "[64,2560],[58,2560],[1,15360]")
    _run_pnnx(pnnx, output_dir / "dit_patch_out.pt", output_dir, "[64,2560]")
    rewrite_ncnn_param(
        output_dir / "dit_block0.ncnn.param",
        size=contract.source_shape,
        windows=(4, 3, 3),
        text_tokens=contract.text_tokens,
    )
    print(f"rewritten {output_dir / 'dit_block0.ncnn.param'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts") / "seedvr2_ema_3b.pth")
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_bridge(args.checkpoint, args.output_dir, args.pnnx, args.seed)


if __name__ == "__main__":
    main()
