import pytest

from tools.reference.resolution_plan import plan_from_explicit, plan_from_input_area, round_half_even


def test_explicit_plan_derives_latent_and_dit_grid():
    plan = plan_from_explicit(720, 1280)

    assert (plan.image_height, plan.image_width) == (720, 1280)
    assert (plan.latent_height, plan.latent_width) == (90, 160)
    assert (plan.source_shape, plan.video_tokens) == ((1, 45, 80), 3600)


def test_auto_plan_matches_seedvr2_area_resize_and_divisible_crop():
    plan = plan_from_input_area(720, 1280)

    assert (plan.resized_height, plan.resized_width) == (720, 1280)
    assert (plan.image_height, plan.image_width) == (720, 1280)
    assert (plan.crop_top, plan.crop_left) == (0, 0)


def test_auto_plan_upsamples_small_image_before_alignment():
    plan = plan_from_input_area(100, 100)

    assert (plan.resized_height, plan.resized_width) == (960, 960)
    assert (plan.image_height, plan.image_width) == (960, 960)
    assert (plan.latent_height, plan.latent_width) == (120, 120)


@pytest.mark.parametrize("height,width", [(0, 128), (-1, 128), (128, 127), (736, 1280)])
def test_invalid_dimensions_are_rejected(height, width):
    with pytest.raises(ValueError):
        plan_from_explicit(height, width)


def test_auto_plan_uses_half_even_rounding_for_resize():
    assert round_half_even(2.5) == 2
    assert round_half_even(3.5) == 4
