#include <cmath>
#include <cstdio>

#include "net.h"
#include "layer/convolution3d.h"
#include "vae/causal_conv3d.h"
#include "vae/temporal_pad.h"

namespace
{

bool causal_conv3d_matches_temporal_pad()
{
    ncnn::Mat input(2, 2, 2, 3);
    for (int channel = 0; channel < input.c; channel++)
        for (int depth = 0; depth < input.d; depth++)
            for (int row = 0; row < input.h; row++)
                for (int column = 0; column < input.w; column++)
                    input.channel(channel).depth(depth).row(row)[column] =
                        static_cast<float>(100 * channel + 10 * depth + 2 * row + column);

    ncnn::ParamDict convolution_params;
    convolution_params.set(0, 2);
    convolution_params.set(1, 3);
    convolution_params.set(11, 3);
    convolution_params.set(21, 3);
    convolution_params.set(2, 1);
    convolution_params.set(12, 1);
    convolution_params.set(22, 1);
    convolution_params.set(3, 1);
    convolution_params.set(13, 1);
    convolution_params.set(23, 1);
    convolution_params.set(4, 1);
    convolution_params.set(14, 1);
    convolution_params.set(24, 0);
    convolution_params.set(5, 1);
    convolution_params.set(6, 162);
    convolution_params.set(9, 1);

    ncnn::Convolution3D convolution;
    if (convolution.load_param(convolution_params) != 0)
        return false;

    ncnn::ParamDict causal_params = convolution_params;
    causal_params.set(31, 2);
    SeedVR2CausalConv3D causal_convolution;
    if (causal_convolution.load_param(causal_params) != 0)
        return false;

    ncnn::Mat weights(162);
    for (int index = 0; index < 162; index++)
        weights[index] = static_cast<float>((index % 7) - 3) * 0.03125f;
    ncnn::Mat bias(2);
    bias[0] = 0.25f;
    bias[1] = -0.5f;
    convolution.weight_data = weights;
    convolution.bias_data = bias;
    causal_convolution.weight_data = weights;
    causal_convolution.bias_data = bias;

    ncnn::ParamDict temporal_pad_params;
    temporal_pad_params.set(0, 2);
    SeedVR2TemporalPad temporal_pad;
    if (temporal_pad.load_param(temporal_pad_params) != 0)
        return false;

    ncnn::Option options;
    ncnn::Mat padded;
    ncnn::Mat expected;
    ncnn::Mat actual;
    if (temporal_pad.forward(input, padded, options) != 0 ||
        convolution.forward(padded, expected, options) != 0 ||
        causal_convolution.forward(input, actual, options) != 0 ||
        expected.total() != actual.total())
        return false;

    for (size_t index = 0; index < expected.total(); index++)
        if (std::fabs(expected[index] - actual[index]) > 1e-5f)
            return false;

    return true;
}

} // namespace

int main()
{
    ncnn::Mat input(2, 1, 2, 3);
    for (int channel = 0; channel < input.c; channel++)
        for (int depth = 0; depth < input.d; depth++)
            for (int width = 0; width < input.w; width++)
                input.channel(channel).depth(depth)[width] = static_cast<float>(100 * channel + 10 * depth + width);

    ncnn::Net net;
    register_seedvr2_vae_layers(net);
    const char* param =
        "7767517\n"
        "2 2\n"
        "Input in0 0 1 in0\n"
        "SeedVR2TemporalPad temporal_pad 1 1 in0 out0 0=2\n";
    if (net.load_param_mem(param) != 0)
        return 1;

    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("in0", input) != 0)
        return 1;
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0)
        return 1;
    if (output.dims != 4 || output.w != 2 || output.h != 1 || output.d != 4 || output.c != 3)
        return 1;

    for (int channel = 0; channel < output.c; channel++)
        for (int width = 0; width < output.w; width++)
        {
            const float expected = static_cast<float>(100 * channel + width);
            if (std::fabs(output.channel(channel).depth(0)[width] - expected) > 1e-6f ||
                std::fabs(output.channel(channel).depth(1)[width] - expected) > 1e-6f ||
                std::fabs(output.channel(channel).depth(2)[width] - expected) > 1e-6f ||
                std::fabs(output.channel(channel).depth(3)[width] - (expected + 10.f)) > 1e-6f)
                return 1;
        }

    if (!causal_conv3d_matches_temporal_pad())
        return 1;

    std::puts("seedvr2-temporal-pad: ok");
    return 0;
}
