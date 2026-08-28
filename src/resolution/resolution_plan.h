#pragma once

#include <string>

namespace seedvr2
{

struct ResolutionPlan final
{
    static constexpr int kDefaultArea = 1280 * 720;

    int image_height = 0;
    int image_width = 0;
    int resized_height = 0;
    int resized_width = 0;
    int crop_top = 0;
    int crop_left = 0;
    int latent_height = 0;
    int latent_width = 0;
    int source_t = 1;
    int source_height = 0;
    int source_width = 0;
    int video_tokens = 0;

    static bool from_explicit(int image_height, int image_width, ResolutionPlan& plan,
                              std::string* error = nullptr);
    static bool from_input_area(int input_height, int input_width, ResolutionPlan& plan,
                                std::string* error = nullptr, int max_area = kDefaultArea);
};

} // namespace seedvr2
