#include "resolution_plan.h"

#include <cmath>
#include <sstream>

namespace seedvr2
{
namespace
{

int round_half_even(double value)
{
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5 || std::fabs(fraction - 0.5) <= 1.0e-12 && static_cast<long long>(lower) % 2 == 0)
        return static_cast<int>(lower);
    return static_cast<int>(lower + 1.0);
}

void set_error(std::string* error, const std::string& message)
{
    if (error)
        *error = message;
}

bool fill_plan(int image_height, int image_width, int resized_height, int resized_width, int crop_top, int crop_left,
               ResolutionPlan& plan, std::string* error)
{
    if (image_height <= 0 || image_width <= 0 || image_height % 16 != 0 || image_width % 16 != 0)
    {
        std::ostringstream message;
        message << "image dimensions must be positive multiples of 16, got " << image_height << "x" << image_width;
        set_error(error, message.str());
        return false;
    }
    plan.image_height = image_height;
    plan.image_width = image_width;
    plan.resized_height = resized_height;
    plan.resized_width = resized_width;
    plan.crop_top = crop_top;
    plan.crop_left = crop_left;
    plan.latent_height = image_height / 8;
    plan.latent_width = image_width / 8;
    plan.source_t = 1;
    plan.source_height = image_height / 16;
    plan.source_width = image_width / 16;
    plan.video_tokens = plan.source_t * plan.source_height * plan.source_width;
    return true;
}

} // namespace

bool ResolutionPlan::from_explicit(int image_height, int image_width, ResolutionPlan& plan, std::string* error)
{
    return fill_plan(image_height, image_width, image_height, image_width, 0, 0, plan, error);
}

bool ResolutionPlan::from_input_area(int input_height, int input_width, ResolutionPlan& plan, std::string* error,
                                     int max_area)
{
    if (input_height <= 0 || input_width <= 0 || max_area <= 0)
    {
        set_error(error, "input dimensions and max area must be positive");
        return false;
    }

    const double scale = std::sqrt(static_cast<double>(max_area) /
                                   (static_cast<double>(input_height) * static_cast<double>(input_width)));
    const int resized_height = round_half_even(static_cast<double>(input_height) * scale);
    const int resized_width = round_half_even(static_cast<double>(input_width) * scale);
    const int image_height = resized_height - resized_height % 16;
    const int image_width = resized_width - resized_width % 16;
    if (image_height <= 0 || image_width <= 0)
    {
        set_error(error, "area-resized dimensions are smaller than the 16-pixel alignment");
        return false;
    }

    const int crop_top = round_half_even(static_cast<double>(resized_height - image_height) / 2.0);
    const int crop_left = round_half_even(static_cast<double>(resized_width - image_width) / 2.0);
    return fill_plan(image_height, image_width, resized_height, resized_width, crop_top, crop_left, plan, error);
}

} // namespace seedvr2
