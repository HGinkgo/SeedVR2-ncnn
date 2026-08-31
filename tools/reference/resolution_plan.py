"""SeedVR2 image-resolution planning contract."""

from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_AREA = 1280 * 720


@dataclass(frozen=True)
class ResolutionPlan:
    image_height: int
    image_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    latent_height: int
    latent_width: int
    source_shape: tuple[int, int, int]
    video_tokens: int


def round_half_even(value: float) -> int:
    """Return the same non-negative half-even rounding used by Python/upstream."""

    return round(value)


def _make_plan(image_height: int, image_width: int, resized_height: int, resized_width: int,
               crop_top: int, crop_left: int) -> ResolutionPlan:
    if image_height <= 0 or image_width <= 0 or image_height % 16 or image_width % 16:
        raise ValueError(f"image dimensions must be positive multiples of 16, got {image_height}x{image_width}")
    if image_height * image_width > DEFAULT_AREA:
        raise ValueError(f"image area must not exceed {DEFAULT_AREA}, got {image_height}x{image_width}")
    source_shape = (1, image_height // 16, image_width // 16)
    return ResolutionPlan(
        image_height=image_height,
        image_width=image_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        latent_height=image_height // 8,
        latent_width=image_width // 8,
        source_shape=source_shape,
        video_tokens=source_shape[0] * source_shape[1] * source_shape[2],
    )


def plan_from_explicit(image_height: int, image_width: int) -> ResolutionPlan:
    return _make_plan(image_height, image_width, image_height, image_width, 0, 0)


def plan_from_input_area(input_height: int, input_width: int, max_area: int = DEFAULT_AREA) -> ResolutionPlan:
    if input_height <= 0 or input_width <= 0 or max_area <= 0:
        raise ValueError("input dimensions and max area must be positive")
    scale = math.sqrt(max_area / (input_height * input_width))
    resized_height = round_half_even(input_height * scale)
    resized_width = round_half_even(input_width * scale)
    image_height = resized_height - resized_height % 16
    image_width = resized_width - resized_width % 16
    if image_height <= 0 or image_width <= 0:
        raise ValueError("area-resized dimensions are smaller than the 16-pixel alignment")
    crop_top = round_half_even((resized_height - image_height) / 2)
    crop_left = round_half_even((resized_width - image_width) / 2)
    return _make_plan(image_height, image_width, resized_height, resized_width, crop_top, crop_left)
