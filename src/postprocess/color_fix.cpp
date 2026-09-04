#include "postprocess/color_fix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace seedvr2
{
namespace
{

constexpr int kWaveletLevels = 5;

bool valid_rgb(const RgbImage& image)
{
    return image.width > 0 && image.height > 0 &&
           image.pixels.size() == static_cast<std::size_t>(image.width) * image.height * 3u;
}

std::size_t pixel_offset(int width, int x, int y, int channel)
{
    return (static_cast<std::size_t>(y) * width + x) * 3u + channel;
}

float clamp_unit(float value)
{
    return std::max(0.f, std::min(1.f, value));
}

using Plane = std::vector<float>;

Plane blur_plane(const Plane& input, int width, int height, int radius)
{
    Plane output(input.size());
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float sum = 0.f;
            for (int kernel_y = -1; kernel_y <= 1; kernel_y++)
                for (int kernel_x = -1; kernel_x <= 1; kernel_x++)
                {
                    const int sample_x = std::max(0, std::min(width - 1, x + kernel_x * radius));
                    const int sample_y = std::max(0, std::min(height - 1, y + kernel_y * radius));
                    const float weight = (kernel_x == 0 ? 2.f : 1.f) * (kernel_y == 0 ? 2.f : 1.f) / 16.f;
                    sum += input[static_cast<std::size_t>(sample_y) * width + sample_x] * weight;
                }
            output[static_cast<std::size_t>(y) * width + x] = sum;
        }
    return output;
}

void decompose(const Plane& input, int width, int height, Plane& high, Plane& low)
{
    Plane current = input;
    high.assign(input.size(), 0.f);
    for (int level = 0; level < kWaveletLevels; level++)
    {
        const Plane blurred = blur_plane(current, width, height, 1 << level);
        for (std::size_t index = 0; index < input.size(); index++)
            high[index] += current[index] - blurred[index];
        current = blurred;
    }
    low = std::move(current);
}

} // namespace

bool prepare_color_reference(const RgbImage& input, const ResolutionPlan& plan, RgbImage& reference,
                             std::string& error)
{
    error.clear();
    reference = RgbImage();
    if (!valid_rgb(input) || plan.image_width <= 0 || plan.image_height <= 0 || plan.resized_width <= 0 ||
        plan.resized_height <= 0 || plan.crop_left < 0 || plan.crop_top < 0 ||
        plan.crop_left + plan.image_width > plan.resized_width || plan.crop_top + plan.image_height > plan.resized_height)
    {
        error = "input image or resolution plan is invalid";
        return false;
    }

    reference.width = plan.image_width;
    reference.height = plan.image_height;
    reference.pixels.resize(static_cast<std::size_t>(reference.width) * reference.height * 3u);
    for (int target_y = 0; target_y < plan.image_height; target_y++)
    {
        const int resized_y = target_y + plan.crop_top;
        const float source_y0 = static_cast<float>(resized_y) * input.height / plan.resized_height;
        const float source_y1 = static_cast<float>(resized_y + 1) * input.height / plan.resized_height;
        const int source_y_begin = std::max(0, static_cast<int>(std::floor(source_y0)));
        const int source_y_end = std::min(input.height, static_cast<int>(std::ceil(source_y1)));
        for (int target_x = 0; target_x < plan.image_width; target_x++)
        {
            const int resized_x = target_x + plan.crop_left;
            const float source_x0 = static_cast<float>(resized_x) * input.width / plan.resized_width;
            const float source_x1 = static_cast<float>(resized_x + 1) * input.width / plan.resized_width;
            const int source_x_begin = std::max(0, static_cast<int>(std::floor(source_x0)));
            const int source_x_end = std::min(input.width, static_cast<int>(std::ceil(source_x1)));
            const float area = (source_x1 - source_x0) * (source_y1 - source_y0);
            if (area <= 0.f)
            {
                error = "color reference resize area is invalid";
                reference = RgbImage();
                return false;
            }

            float channels[3] = {0.f, 0.f, 0.f};
            for (int source_y = source_y_begin; source_y < source_y_end; source_y++)
            {
                const float overlap_y = std::max(0.f, std::min(source_y1, static_cast<float>(source_y + 1)) -
                                                          std::max(source_y0, static_cast<float>(source_y)));
                for (int source_x = source_x_begin; source_x < source_x_end; source_x++)
                {
                    const float overlap_x = std::max(0.f, std::min(source_x1, static_cast<float>(source_x + 1)) -
                                                              std::max(source_x0, static_cast<float>(source_x)));
                    const float weight = overlap_x * overlap_y / area;
                    const std::size_t offset = pixel_offset(input.width, source_x, source_y, 0);
                    for (int channel = 0; channel < 3; channel++)
                        channels[channel] += input.pixels[offset + channel] * weight;
                }
            }
            const std::size_t output_offset = pixel_offset(reference.width, target_x, target_y, 0);
            for (int channel = 0; channel < 3; channel++)
                reference.pixels[output_offset + channel] = static_cast<unsigned char>(std::lround(channels[channel]));
        }
    }
    return true;
}

bool apply_wavelet_color_fix(const RgbImage& content, const RgbImage& reference, RgbImage& output,
                             std::string& error)
{
    error.clear();
    if (!valid_rgb(content) || !valid_rgb(reference) || content.width != reference.width ||
        content.height != reference.height)
    {
        error = "color fix inputs must be valid RGB images with the same dimensions";
        return false;
    }

    RgbImage result;
    result.width = content.width;
    result.height = content.height;
    result.pixels.resize(content.pixels.size());
    for (int channel = 0; channel < 3; channel++)
    {
        Plane content_plane(content.pixels.size() / 3u);
        Plane reference_plane(content.pixels.size() / 3u);
        for (int y = 0; y < content.height; y++)
            for (int x = 0; x < content.width; x++)
            {
                const std::size_t index = static_cast<std::size_t>(y) * content.width + x;
                content_plane[index] = content.pixels[pixel_offset(content.width, x, y, channel)] / 255.f;
                reference_plane[index] = reference.pixels[pixel_offset(reference.width, x, y, channel)] / 255.f;
            }

        Plane content_high;
        Plane content_low;
        Plane reference_high;
        Plane reference_low;
        decompose(content_plane, content.width, content.height, content_high, content_low);
        decompose(reference_plane, reference.width, reference.height, reference_high, reference_low);
        (void)reference_high;
        for (int y = 0; y < content.height; y++)
            for (int x = 0; x < content.width; x++)
            {
                const std::size_t index = static_cast<std::size_t>(y) * content.width + x;
                result.pixels[pixel_offset(result.width, x, y, channel)] =
                    static_cast<unsigned char>(std::lround(clamp_unit(content_high[index] + reference_low[index]) * 255.f));
            }
    }
    output = std::move(result);
    return true;
}

} // namespace seedvr2
