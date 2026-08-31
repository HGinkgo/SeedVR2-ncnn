# Video Batch Reuse Design

## Goal

Reduce repeated model loading during AVI enhancement without changing the CLI,
the FP32 inference mode, or the Vulkan memory peak validated at dynamic 256x256.

The current video loop invokes the complete image pipeline for every frame. Each
frame therefore loads and releases the VAE encoder, the 32-block DiT stack, and
the VAE decoder. A two-frame 128x128 baseline completed in 110.19 seconds and
loaded the DiT stack twice.

## Decision

Process video input in fixed internal batches of two frames. There is no new
CLI option in this stage. A one-frame final batch uses the same path.

Within each batch, execute stages in this order:

1. Load the VAE encoder once, encode every input frame, and download each
   condition latent to CPU memory.
2. Destroy the encoder and clear its allocators. Load the DiT stack once,
   process every condition latent, and download each output latent to CPU
   memory.
3. Destroy the DiT stack and clear its allocators. Load the VAE decoder once,
   decode every output latent, then return RGB frames in their original order.

The CLI continues to own AVI read/write. It collects up to two frames, calls a
batch method on `ImageInferenceSession`, and immediately writes the returned
frames in order.

## Why This Shape

Keeping encoder, DiT, and decoder alive together is not safe: the earlier
dynamic 256x256 persistent-session experiment failed at VAE decode with Vulkan
out-of-memory. Releasing a cached `DitStackSession` before decode is also not a
solution, because its destructor calls `ncnn::Net::clear()` on all 32 blocks;
the next frame must load them again.

The selected design keeps only one large model group resident at a time, while
reducing a two-frame batch from six model loads to three. CPU latent storage is
bounded by two frames and is released after each batch. This follows the
low-peak-memory staging principle used by Wan ncnn Vulkan, without adopting its
separate FP16, packing, or ReBAR optimizations.

## Interfaces

`ImageInferenceSession` gains a batch method accepting ordered RGB inputs and
returning ordered RGB outputs. `run_frame()` delegates to the same implementation
with a single-element batch so image inference behavior remains unchanged.

The Vulkan implementation is factored into three internal phase helpers:

- encode RGB images to CPU condition latents;
- run the DiT pipeline from CPU condition latents to CPU output latents;
- decode CPU output latents to RGB images.

Each helper creates its own local `ncnn::Net` or `DitStackSession`; destruction
and allocator clearing happen before the next helper begins. CPU/GPU transfers
are explicit `VkCompute` upload or download operations.

## Observability And Errors

Video logs add `stage=video-batch start=<n> frames=<count>` and retain one
`stage=video-frame index=<n>` line for every decoded input frame. Per-frame
compute stages retain their existing names and include the input-frame index
where a batch error is reported. Existing top-level video errors remain
non-zero and the writer stays incremental: frames from earlier completed
batches remain in the output file if a later batch fails.

## Acceptance

- A two-frame 128x128 AVI smoke test exits zero, writes two output frames in
  order, and logs `stage=load-dit-stack` exactly once.
- The same log contains two executions each of `vae-encode`, `dit-stack`, and
  `vae-decode`.
- Peak GPU residency never includes encoder, DiT, and decoder weights at the
  same time; dynamic 256x256 one-frame inference remains successful.
- Existing image inference and non-Vulkan error behavior remain covered by the
  focused C++ test suite.

## Non-Goals

This stage does not change model precision, enable packing, implement ReBAR,
add multi-GPU execution, add a batch-size flag, or change AVI/FFmpeg behavior.
