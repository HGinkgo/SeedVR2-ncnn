#include <cstdio>
#include <cstdlib>
#include <string>

#include "resolution/resolution_plan.h"

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

} // namespace

int main()
{
    seedvr2::ResolutionPlan square;
    require(seedvr2::ResolutionPlan::from_explicit(128, 128, square), "explicit square plan");
    require(square.image_height == 128 && square.image_width == 128, "square image shape");
    require(square.latent_height == 16 && square.latent_width == 16, "square latent shape");
    require(square.source_t == 1 && square.source_height == 8 && square.source_width == 8,
            "square dit shape");
    require(square.video_tokens == 64, "square token count");

    seedvr2::ResolutionPlan widescreen;
    require(seedvr2::ResolutionPlan::from_explicit(720, 1280, widescreen), "explicit widescreen plan");
    require(widescreen.latent_height == 90 && widescreen.latent_width == 160, "widescreen latent shape");
    require(widescreen.source_t == 1 && widescreen.source_height == 45 && widescreen.source_width == 80,
            "widescreen dit shape");
    require(widescreen.video_tokens == 3600, "widescreen token count");

    seedvr2::ResolutionPlan automatic;
    require(seedvr2::ResolutionPlan::from_input_area(720, 1280, automatic), "official auto plan");
    require(automatic.image_height == 720 && automatic.image_width == 1280, "official auto dimensions");
    require(automatic.crop_top == 0 && automatic.crop_left == 0, "official auto crop");

    seedvr2::ResolutionPlan upsampled;
    require(seedvr2::ResolutionPlan::from_input_area(100, 100, upsampled), "small auto plan");
    require(upsampled.image_height == 960 && upsampled.image_width == 960, "small image area resize");
    require(upsampled.latent_height == 120 && upsampled.latent_width == 120, "small image latent shape");

    seedvr2::ResolutionPlan invalid;
    require(!seedvr2::ResolutionPlan::from_explicit(0, 128, invalid), "zero height rejected");
    require(!seedvr2::ResolutionPlan::from_explicit(128, 127, invalid), "unaligned width rejected");
    require(!seedvr2::ResolutionPlan::from_input_area(-1, 128, invalid), "negative input rejected");

    std::puts("seedvr2-resolution: ok");
    return 0;
}
