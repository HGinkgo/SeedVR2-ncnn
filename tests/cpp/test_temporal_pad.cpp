#include <cmath>
#include <cstdio>

#include "net.h"
#include "vae/temporal_pad.h"

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

    std::puts("seedvr2-temporal-pad: ok");
    return 0;
}
