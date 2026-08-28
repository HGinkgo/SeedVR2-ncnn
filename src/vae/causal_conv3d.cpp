#include "causal_conv3d.h"

#include <algorithm>
#include <vector>

#include "layer/fused_activation.h"

namespace
{

struct Padding
{
    int before = 0;
    int after = 0;
};

Padding resolve_padding(int size, int kernel_extent, int stride, int before, int after)
{
    if (before == -233 && after == -233)
    {
        const int total = kernel_extent + (size - 1) / stride * stride - size;
        return {total / 2, total - total / 2};
    }
    if (before == -234 && after == -234)
    {
        const int total = kernel_extent + (size - 1) / stride * stride - size;
        return {total - total / 2, total / 2};
    }
    return {std::max(before, 0), std::max(after, 0)};
}

} // namespace

SeedVR2CausalConv3D::SeedVR2CausalConv3D()
{
    support_vulkan = false;
}

int SeedVR2CausalConv3D::load_param(const ncnn::ParamDict& pd)
{
    const int result = ncnn::Convolution3D::load_param(pd);
    prepend_ = pd.get(31, 0);
    return result == 0 && prepend_ > 0 ? 0 : -1;
}

int SeedVR2CausalConv3D::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                                  const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.elempack != 1 || bottom_blob.elemsize != 4u ||
        bottom_blob.empty() || prepend_ <= 0)
        return -1;

    const int input_width = bottom_blob.w;
    const int input_height = bottom_blob.h;
    const int input_depth = bottom_blob.d;
    const int input_channels = bottom_blob.c;
    const int virtual_depth = input_depth + prepend_;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;
    const Padding padding_w = resolve_padding(input_width, kernel_extent_w, stride_w, pad_left, pad_right);
    const Padding padding_h = resolve_padding(input_height, kernel_extent_h, stride_h, pad_top, pad_bottom);
    const Padding padding_d = resolve_padding(virtual_depth, kernel_extent_d, stride_d, pad_front, pad_behind);
    const int padded_width = input_width + padding_w.before + padding_w.after;
    const int padded_height = input_height + padding_h.before + padding_h.after;
    const int padded_depth = virtual_depth + padding_d.before + padding_d.after;
    const int output_width = (padded_width - kernel_extent_w) / stride_w + 1;
    const int output_height = (padded_height - kernel_extent_h) / stride_h + 1;
    const int output_depth = (padded_depth - kernel_extent_d) / stride_d + 1;
    if (output_width <= 0 || output_height <= 0 || output_depth <= 0)
        return -1;

    top_blob.create(output_width, output_height, output_depth, num_output, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int kernel_size = kernel_w * kernel_h * kernel_d;
    const float* weights = static_cast<const float*>(weight_data.data);
    const float* bias = bias_term ? static_cast<const float*>(bias_data.data) : 0;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int output_channel = 0; output_channel < num_output; output_channel++)
    {
        float* output = top_blob.channel(output_channel);
        const float* channel_weights = weights + static_cast<size_t>(output_channel) * input_channels * kernel_size;
        for (int output_depth_index = 0; output_depth_index < output_depth; output_depth_index++)
        {
            for (int output_row = 0; output_row < output_height; output_row++)
            {
                for (int output_column = 0; output_column < output_width; output_column++)
                {
                    float sum = bias ? bias[output_channel] : 0.f;
                    const float* kernel = channel_weights;
                    for (int input_channel = 0; input_channel < input_channels; input_channel++)
                    {
                        const ncnn::Mat input = bottom_blob.channel(input_channel);
                        for (int kernel_depth_index = 0; kernel_depth_index < kernel_d; kernel_depth_index++)
                        {
                            const int virtual_depth_index = output_depth_index * stride_d +
                                                            kernel_depth_index * dilation_d - padding_d.before;
                            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
                            {
                                const int input_row = output_row * stride_h + kernel_row * dilation_h - padding_h.before;
                                for (int kernel_column = 0; kernel_column < kernel_w; kernel_column++, kernel++)
                                {
                                    const int input_column = output_column * stride_w +
                                                             kernel_column * dilation_w - padding_w.before;
                                    float value = pad_value;
                                    if (virtual_depth_index >= 0 && virtual_depth_index < virtual_depth &&
                                        input_row >= 0 && input_row < input_height &&
                                        input_column >= 0 && input_column < input_width)
                                    {
                                        const int source_depth = std::max(virtual_depth_index - prepend_, 0);
                                        value = input.depth(source_depth).row(input_row)[input_column];
                                    }
                                    sum += value * *kernel;
                                }
                            }
                        }
                    }
                    *output++ = activation_ss(sum, activation_type, activation_params);
                }
            }
        }
    }

    return 0;
}

DEFINE_LAYER_CREATOR(SeedVR2CausalConv3D)
