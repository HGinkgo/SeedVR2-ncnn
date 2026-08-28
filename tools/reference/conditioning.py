"""Frozen SeedVR2 conditioning and sampling contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SeedVR2ConditioningContract:
    text_width: int = 5120
    positive_tokens: int = 58
    negative_tokens: int = 64
    cfg_scale: float = 1.0
    cfg_rescale: float = 0.0
    schedule_t: float = 1000.0
    sampling_steps: int = 1
    sampling_shift: float = 1.0


SEEDVR2_CONDITIONING_CONTRACT = SeedVR2ConditioningContract()


def _load_condition(
    path: Path, token_count: int, contract: SeedVR2ConditioningContract, label: str
) -> torch.Tensor:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"condition file {path} does not contain a tensor")
    if value.ndim != 2 or value.shape[1] != contract.text_width:
        raise ValueError(f"{label} condition {path} expected width {contract.text_width}, got shape {tuple(value.shape)}")
    if value.shape[0] != token_count:
        raise ValueError(f"{label} condition {path} expected {token_count} tokens, got {value.shape[0]}")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def load_conditioning_pair(
    positive_path: Path,
    negative_path: Path,
    *,
    contract: SeedVR2ConditioningContract = SEEDVR2_CONDITIONING_CONTRACT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load official positive and negative embeddings without padding or truncation."""

    positive = _load_condition(positive_path, contract.positive_tokens, contract, "positive")
    negative = _load_condition(negative_path, contract.negative_tokens, contract, "negative")
    return positive, negative


def classifier_free_guidance(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    scale: float,
    rescale: float = 0.0,
) -> torch.Tensor:
    """Match the official CFG and optional standard-deviation rescale."""

    if positive.shape != negative.shape:
        raise ValueError(f"CFG predictions must have equal shapes, got {positive.shape} and {negative.shape}")
    guided = negative + float(scale) * (positive - negative)
    if float(rescale) != 0.0:
        reduce_dims = list(range(1, guided.ndim))
        positive_std = positive.std(dim=reduce_dims, keepdim=True)
        guided_std = guided.std(dim=reduce_dims, keepdim=True)
        factor = positive_std / guided_std
        factor = float(rescale) * factor + (1.0 - float(rescale))
        guided = guided * factor
    return guided


def uniform_trailing_timesteps(
    *,
    schedule_t: float = SEEDVR2_CONDITIONING_CONTRACT.schedule_t,
    steps: int = SEEDVR2_CONDITIONING_CONTRACT.sampling_steps,
    shift: float = SEEDVR2_CONDITIONING_CONTRACT.sampling_shift,
) -> torch.Tensor:
    """Match ``UniformTrailingSamplingTimesteps`` for a continuous schedule."""

    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if float(schedule_t) <= 0.0 or float(shift) <= 0.0:
        raise ValueError("schedule_t and shift must be positive")
    timesteps = torch.arange(1.0, 0.0, -1.0 / int(steps), dtype=torch.float32)
    timesteps = float(shift) * timesteps / (1.0 + (float(shift) - 1.0) * timesteps)
    return timesteps * float(schedule_t)


def v_lerp_endpoint(
    sample: torch.Tensor,
    prediction: torch.Tensor,
    *,
    timestep: float,
    schedule_t: float = SEEDVR2_CONDITIONING_CONTRACT.schedule_t,
) -> torch.Tensor:
    """Resolve the official one-step ``v_lerp`` prediction at ``t=T`` to ``x_0``."""

    if sample.shape != prediction.shape:
        raise ValueError(f"sample and prediction must have equal shapes, got {sample.shape} and {prediction.shape}")
    if float(schedule_t) <= 0.0 or float(timestep) != float(schedule_t):
        raise ValueError("v_lerp endpoint requires timestep equal to schedule_t")
    return sample - prediction


def transform_timestep(
    timesteps: torch.Tensor,
    latent_shapes: torch.Tensor,
    *,
    schedule_t: float = SEEDVR2_CONDITIONING_CONTRACT.schedule_t,
    temporal_downsample_factor: int = 4,
    spatial_downsample_factor: int = 8,
) -> torch.Tensor:
    """Match the official resolution-dependent timestep transform."""

    shape = torch.as_tensor(latent_shapes)
    if shape.ndim != 2 or shape.shape[1] != 3:
        raise ValueError(f"latent_shapes must have shape (batch, 3), got {tuple(shape.shape)}")
    values = torch.as_tensor(timesteps, dtype=torch.float32)
    if values.ndim == 0:
        values = values[None]
    if values.ndim != 1 or values.numel() not in (1, shape.shape[0]):
        raise ValueError(f"timesteps must have one value or one value per batch, got {tuple(values.shape)}")
    if values.numel() == 1 and shape.shape[0] != 1:
        values = values.expand(shape.shape[0])

    shape = shape.to(dtype=torch.float32, device=values.device)
    frames = (shape[:, 0] - 1.0) * float(temporal_downsample_factor) + 1.0
    heights = shape[:, 1] * float(spatial_downsample_factor)
    widths = shape[:, 2] * float(spatial_downsample_factor)

    image_m = (3.2 - 1.0) / (1024.0 * 1024.0 - 256.0 * 256.0)
    image_b = 1.0 - image_m * 256.0 * 256.0
    video_m = (5.0 - 1.0) / (1280.0 * 720.0 * 145.0 - 256.0 * 256.0 * 37.0)
    video_b = 1.0 - video_m * 256.0 * 256.0 * 37.0
    image_shift = image_m * heights * widths + image_b
    video_shift = video_m * heights * widths * frames + video_b
    shift = torch.where(frames > 1.0, video_shift, image_shift)

    normalized = values / float(schedule_t)
    transformed = shift * normalized / (1.0 + (shift - 1.0) * normalized)
    return transformed * float(schedule_t)


__all__ = [
    "SEEDVR2_CONDITIONING_CONTRACT",
    "SeedVR2ConditioningContract",
    "classifier_free_guidance",
    "load_conditioning_pair",
    "transform_timestep",
    "uniform_trailing_timesteps",
    "v_lerp_endpoint",
]
