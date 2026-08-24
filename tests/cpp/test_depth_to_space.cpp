#include <cmath>
#include <cstdio>

#include "net.h"
#include "vae/depth_to_space.h"

int main()
{
    ncnn::Mat input(2, 2, 1, 8);
    for (int channel = 0; channel < input.c; channel++)
        for (int row = 0; row < input.h; row++)
            for (int column = 0; column < input.w; column++)
                input.channel(channel).row(row)[column] = static_cast<float>(100 * channel + 10 * row + column);

    ncnn::Net net;
    register_seedvr2_vae_layers(net);
    const char* param =
        "7767517\n"
        "2 2\n"
        "Input in0 0 1 in0\n"
        "SeedVR2DepthToSpace depth_to_space 1 1 in0 out0 0=2 1=2 2=2\n";
    if (net.load_param_mem(param) != 0)
        return 1;

    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("in0", input) != 0)
        return 1;
    ncnn::Mat output;
    if (extractor.extract("out0", output) != 0)
        return 1;
    if (output.dims != 4 || output.w != 4 || output.h != 4 || output.d != 2 || output.c != 1)
    {
        std::fprintf(stderr, "shape dims=%d w=%d h=%d d=%d c=%d n=%d\n", output.dims, output.w, output.h, output.d, output.c, output.n);
        return 1;
    }

    for (int depth = 0; depth < output.d; depth++)
        for (int row = 0; row < output.h; row++)
            for (int column = 0; column < output.w; column++)
            {
                const int source_channel = ((row % 2) * 2 + (column % 2)) * 2 + (depth % 2);
                const int source_row = row / 2;
                const int source_column = column / 2;
                const float expected = static_cast<float>(100 * source_channel + 10 * source_row + source_column);
                if (std::fabs(output.channel(0).depth(depth).row(row)[column] - expected) > 1e-6f)
                {
                    std::fprintf(stderr, "mismatch d=%d y=%d x=%d got=%f expected=%f source=%d\n", depth, row, column,
                                 output.channel(0).depth(depth).row(row)[column], expected, source_channel);
                    return 1;
                }
            }

    std::puts("seedvr2-depth-to-space: ok");
    return 0;
}
