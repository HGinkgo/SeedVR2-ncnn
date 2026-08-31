"""Create a self-contained dynamic-resolution SeedVR2 ncnn model package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.export_dit_block import normalize_dynamic_awa_template, normalize_dynamic_dit_gemm_rows
from tools.export_vae import migrate_legacy_depth_to_space_params, normalize_dynamic_vae_template


_VAE_GRAPHS = ("vae_encode", "vae_decode")
_DIT_BASE_GRAPHS = ("dit_input", "dit_embedding", "dit_output")
_DIT_BLOCK_GRAPHS = tuple(f"dit_block_{index:02d}" for index in range(32))
_GRAPH_NAMES = _VAE_GRAPHS + _DIT_BASE_GRAPHS + _DIT_BLOCK_GRAPHS
_CONDITIONING_NAME = Path("conditioning") / "pos_emb.f32"


def _require_regular_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing model artifact: {path}")


def _copy_graph(source_dir: Path, destination_dir: Path, name: str) -> None:
    for suffix in (".ncnn.param", ".ncnn.bin"):
        source = source_dir / f"{name}{suffix}"
        _require_regular_file(source)
        shutil.copy2(source, destination_dir / source.name, follow_symlinks=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(destination_dir: Path) -> None:
    files = sorted(
        path
        for path in destination_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(destination_dir).as_posix()}" for path in files]
    (destination_dir / "manifest.sha256").write_text("\n".join(lines) + "\n")


def materialize_dynamic_model_package(
    source_variant_dir: Path, source_conditioning_dir: Path, destination_dir: Path
) -> None:
    """Copy one exported variant into a flat, dynamic-resolution package.

    The source graph binaries are resolution-independent.  Their parameter
    templates are normalized while copying, yielding a package with ordinary
    files only, suitable for archives and release distribution.
    """

    source_variant_dir = Path(source_variant_dir).resolve()
    source_conditioning_dir = Path(source_conditioning_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    if not source_variant_dir.is_dir():
        raise NotADirectoryError(f"source variant directory does not exist: {source_variant_dir}")
    if not source_conditioning_dir.is_dir():
        raise NotADirectoryError(f"source conditioning directory does not exist: {source_conditioning_dir}")
    if destination_dir.exists():
        raise FileExistsError(f"destination already exists: {destination_dir}")

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{destination_dir.name}.tmp-", dir=destination_dir.parent))
    try:
        for graph_name in _GRAPH_NAMES:
            _copy_graph(source_variant_dir, staging_dir, graph_name)

        conditioning = source_conditioning_dir / _CONDITIONING_NAME.name
        _require_regular_file(conditioning)
        conditioning_output_dir = staging_dir / _CONDITIONING_NAME.parent
        conditioning_output_dir.mkdir()
        shutil.copy2(conditioning, conditioning_output_dir / conditioning.name, follow_symlinks=True)

        for graph_name in _VAE_GRAPHS:
            param_path = staging_dir / f"{graph_name}.ncnn.param"
            migrate_legacy_depth_to_space_params(param_path)
            normalize_dynamic_vae_template(param_path)
        for graph_name in ("dit_input", "dit_output"):
            normalize_dynamic_dit_gemm_rows(staging_dir / f"{graph_name}.ncnn.param")
        for graph_name in _DIT_BLOCK_GRAPHS:
            normalize_dynamic_awa_template(staging_dir / f"{graph_name}.ncnn.param")
        _write_manifest(staging_dir)
        staging_dir.replace(destination_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-variant-dir", type=Path, required=True)
    parser.add_argument("--source-conditioning-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_dynamic_model_package(args.source_variant_dir, args.source_conditioning_dir, args.output_dir)


if __name__ == "__main__":
    main()
