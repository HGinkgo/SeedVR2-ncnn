# SeedVR2-ncnn

Native C++ / ncnn / Vulkan SeedVR2 image and video enhancement CLI for Linux x86_64.

[中文](README.md) · [Build from source](#build) · [Model](https://modelscope.cn/models/HGinkgo/SeedVR2-ncnn)

Production defaults to the mixed-precision configuration that passed all six paired fixed-256 quality cases: DiT input, timestep embedding, and the first 9 blocks use FP32, followed by BF16 blocks. Peak VRAM was around 22.9 GiB on an RTX 3090. `--precision fp32` remains available for full-FP32 comparison. The fixed-256 ExportedProgram/PT2 → PNNX → ncnn path remains opt-in.

## Showcase

An official example frame is downsampled to `64x64` and restored with the fixed `256x256` path:

| Input | Output |
| --- | --- |
| <img src="assets/showcase-image-input-64.png" alt="64x64 input" width="256"> | <img src="assets/showcase-image-output-256.png" alt="256x256 output" width="256"> |

The input comes from the [official SeedVR2 demo](https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B).

Video inference result (36 frames, `64x64` input restored to `256x256`, RGB24 AVI):

The comparison GIF shows the `64x64` input on the left and the ncnn Vulkan `256x256` enhanced output on the right:

![Official video input and SeedVR2-ncnn output comparison](assets/showcase-video-comparison-bf16.gif)

<video controls width="256" preload="metadata">
  <source src="assets/showcase-video-output-256.avi" type="video/x-msvideo">
</video>

The complete [input video](assets/showcase-video-input-64.avi) and ncnn Vulkan [output video](assets/showcase-video-output-256.avi) are included. The input is made by bilinearly downsampling the first 36 frames of the official `256x256` example video to `64x64`. This file showcases the ncnn Vulkan inference output and is not an independent video-quality acceptance result.

## Features

- Native C++ inference without Python, PyTorch, or a CUDA runtime.
- Linux NVIDIA Vulkan with BF16 storage and FP32 accumulation where required.
- PNG/JPEG images and RGB24 AVI video.
- Validated fixed targets: `128x128`, `128x256`, and `256x256`.
- Current optimization acceptance targets RTX 3090, one GPU, fixed `256x256` BF16 Vulkan.
- One-step Euler is the default; multi-step sampling and VAE tiling are experimental.
- Optional `--seed 666` selects reproducible standard-normal initial noise; omitting it retains the legacy fixture.
- Output uses reference-guided color reconstruction, retaining generated high-frequency detail while rebuilding low-frequency color from the input.

The validated fixed-`256x256` mixed-precision path is now the default:

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

This runs the DiT input, timestep embedding, and first 9 DiT blocks in FP32, then uses BF16 for the remaining blocks. All six fixed quality comparisons passed; the one-step peak was about 22.9 GiB VRAM.

## Quick Start

Follow [Build](#build) to compile from source, then run the commands below from the repository root.

Download the model:

```bash
modelscope download HGinkgo/SeedVR2-ncnn --local-dir models/seedvr2-3b
```

Image:

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.png \
  --output output.png \
  --width 256 --height 256 \
  --gpu-id 0
```

Video:

```bash
build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b \
  --input input.avi \
  --output output.avi \
  --width 256 --height 256 \
  --frames 36 \
  --gpu-id 0
```

This processes the first 36 input frames and writes an RGB24 AVI. Omit `--frames` to process the whole video. The default build supports AVI; compressed video input requires a build with `-DSEEDVR2_ENABLE_FFMPEG=ON`. The runtime does not require Python, PyTorch, or CUDA. Use `--help` for all options.

Set `SEEDVR2_PIPELINE_CACHE_PATH` to enable persistent Vulkan pipeline caching. A matching cache helps repeated image startup; a missing or incompatible cache is rebuilt:

```bash
mkdir -p "$HOME/.cache/seedvr2-ncnn"
SEEDVR2_PIPELINE_CACHE_PATH="$HOME/.cache/seedvr2-ncnn/pipeline.cache" \
  build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png --gpu-id 0
```

## Validation

| Target | Image | AVI |
| --- | --- | --- |
| `128x128` | verified | verified, 2 frames |
| `128x256` | verified | - |
| `256x256` | verified | verified, 36 frames |

The fixed `256x256` path retains three optimizations: Vulkan pointwise Conv3D
for five decoder nodes, OC8 output-channel reuse for two causal Conv3D nodes,
and DiT intermediate tensor lifetime and in-place reuse.

The latest runtime comparison measures `343.65 s -> 337.07 s` for one 36-frame
pair, with `3.55%` lower DiT stage latency and byte-identical output. Per-block
peak live temporary memory falls by `84.6%`; whole-video peak GPU memory is
unchanged. These gains cannot be added or directly compared with older
end-to-end timings.

## Build

Use a C++17 compiler, CMake, Vulkan development files, and the pinned ncnn
submodule:

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DSEEDVR2_ENABLE_VULKAN=ON \
  -DNCNN_BATCH=ON \
  -DNCNN_BF16=ON \
  -DNCNN_BENCHMARK=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target seedvr2-ncnn --parallel
```

The model directory must be the complete dynamic package, containing
`manifest.sha256` with 75 entries and no symbolic links. Pass that directory
to `--model-dir`.

Inside this source workspace, use the prepared NVIDIA runtime:

```bash
tools/with-nvidia-vulkan-runtime.sh build/seedvr2-ncnn \
  --model-dir models/seedvr2-3b --input input.png --output output.png \
  --width 256 --height 256 --gpu-id 0
```

The helper selects NVIDIA user-space libraries from the ignored `runtime/`
directory. Environments with a configured NVIDIA Vulkan driver can run
`build/seedvr2-ncnn` directly.
Compressed video input additionally requires FFmpeg development files and
`-DSEEDVR2_ENABLE_FFMPEG=ON`. The ModelScope CLI is only a download tool, not an
inference dependency.

## PT2 / PNNX experimental path

The 2026-10-19 audit imports original NaDiT and compares every layer. It corrected the final
Ada embedding slice and the last block's incorrect text Ada/MLP execution. New exports select
the first triplet per feature from `[2560,6]`; block 31 retains text attention and skips text
Ada/MLP. Resume and packaging reject artifacts without the new contracts.
The pinned official source has an output-Ada dimension mismatch; this project uses an explicit
compatibility contract. Historical consistency measurements below do not establish official sampling or image quality.

The repository-local [`pnnx/`](pnnx/) module provides a native Torch Dynamo ExportedProgram reader and supports `pnnx model.pt2`, including PT2 graph, weight, and input/output signature loading. Use [`tools/prepare_pnnx_pt2.sh`](tools/prepare_pnnx_pt2.sh) to build the module and run the two basic PT2 tests. The broader operator matrix and full SeedVR2 conversion remain separate integration work.

Static/bounded-token fixtures, checkpoint MLP/Ada/attention, complete fixed blocks 0/1/9, and fixed single-frame PT2 VAE encoder/decoder have CPU/Vulkan evidence. **Provenance correction:** earlier 35-graph stack measurements used TorchScript `.pt` exports. The old `model-package-pt2-full-stack-256` contains PT2 VAE plus TorchScript full-matrix DiT, despite its name. Those measurements do not establish PT2 DiT stack support.

On 2026-10-19, genuine `.pt2` exports passed fixed-256 acceptance for both 58-token positive and 64-token negative stacks: all 35 graphs in each stack passed CPU/Vulkan checks, with zero maximum difference against TorchScript on identical inputs. Both serial Vulkan stacks reached a final maximum absolute error of `1.26362e-5`.

The genuine PT2 VAE + DiT candidate passed four fixed-256 image CLI cases on an RTX 3090: CFG=1/1.5 with one/two steps. The production TorchScript one/two-step regressions also passed. Releasing both CFG DiT weight sets before VAE decode resolves an observed allocation failure. PNG differences are recorded separately from FP32 graph acceptance; these runs establish neither image-quality equivalence nor a performance gain.

BF16 validation exposed an AWA int32 mapping upload being converted as reduced-precision weights. The fix preserves integer bytes and bounds padded RoPE dispatch by token count. With identical PT2 VAE and conditioning, the corrected TS/PT2 DiT paths match all three tensor handoff fingerprints and the final PNG SHA256 for two-step CFG=1 and 1.5. An independent PT2 process also reproduces the handoffs and PNG exactly; BF16 repeats match all 68 stages of each stack, and the two-frame CFG video path passes. The video CLI currently runs source_t=1 models independently per frame; temporal VAE and cross-frame attention have not been implemented and validated, so this is not yet a temporal-video port.

Five of six fixed-input comparisons between the original default BF16 path and Vulkan FP32 missed the project quality gates. The default now keeps the DiT input, timestep embedding and first nine blocks in FP32, with BF16 for the remaining blocks. It passed all six fixed-256 comparisons against the same gates, with peak GPU memory around 22.9 GiB. Full FP32 comparison remains available with `--precision fp32`.

Optional standard-normal noise via `--seed` passed five image and two video Vulkan runs. Matched TS/PT2 and an independent PT2 repeat with seed=666 match all four handoff fingerprints and PNG exactly; the second video frame also reproduces when processed separately with its absolute frame index. Omitting seed preserves the previous output. Complete original preprocessing and temporal model paths remain unverified; the precision result above is limited to the fixed 256 comparison set.

[`tools/export_dit_stack.py`](tools/export_dit_stack.py) now accepts explicit `--format pt2` and retains each source hash and native conversion record; its default remains `torchscript`. CFG uses separate 58-token positive and 64-token negative stacks, enabled with `--cfg-scale` (default 1). [`tools/prepare_pt2_cfg_package.py`](tools/prepare_pt2_cfg_package.py) assembles the fixed-256 experimental package. The production model package remains TorchScript/ncnn and defaults to the validated mixed precision path; replacing it end to end with PT2 remains unfinished. Dynamic/multi-frame VAE, text encoding and CFG rescale are unverified.

See [`pnnx/README.md`](pnnx/README.md) for the PNNX module and basic validation command.

Run the basic project regressions with `ctest --test-dir build --output-on-failure`. Run the basic PNNX `.pt2` conversion checks with `tools/prepare_pnnx_pt2.sh --python <Python-with-PyTorch>`.

## Scope

- Current product line: Linux x86_64 NVIDIA Vulkan.
- Output target is capped at `256x256`; 720p and long video are not validated.
- Model weights and third-party dependencies retain their own licenses.

## License

ncnn uses BSD-3-Clause. SeedVR2 models, weights, and other third-party dependencies retain their respective licenses.
New PT2 tools and patches use [BSD-3-Clause](LICENSES/BSD-3-Clause.txt). The existing
project has no root LICENSE; this does not retroactively license all existing code.
