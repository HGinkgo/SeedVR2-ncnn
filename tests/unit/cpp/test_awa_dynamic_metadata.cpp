#include <cstdio>
#include <cstdlib>

#include "awa/awa_layers.h"

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
    if (!seedvr2::make_awa_windows(0, 45, 80, 4, 3, 3, false).empty() ||
        !seedvr2::make_awa_windows(1, 45, 80, 0, 3, 3, false).empty())
    {
        std::fprintf(stderr, "invalid AWA metadata inputs must return empty windows\n");
        return 1;
    }

    const std::vector<seedvr2::AwaWindow> unshifted = seedvr2::make_awa_windows(1, 45, 80, 4, 3, 3, false);
    require(unshifted.size() == 9, "unshifted window count");
    require(unshifted.front().h0 == 0 && unshifted.front().h1 == 15 && unshifted.front().w0 == 0 &&
                unshifted.front().w1 == 27,
            "unshifted first window");
    require(unshifted.back().h0 == 30 && unshifted.back().h1 == 45 && unshifted.back().w0 == 54 &&
                unshifted.back().w1 == 80,
            "unshifted last window");

    const std::vector<seedvr2::AwaWindow> shifted = seedvr2::make_awa_windows(1, 45, 80, 4, 3, 3, true);
    require(shifted.size() == 16, "shifted window count");
    require(shifted.front().h0 == 0 && shifted.front().h1 == 7 && shifted.front().w0 == 0 &&
                shifted.front().w1 == 13,
            "shifted first window");

    ncnn::ParamDict params;
    params.set(0, 1);
    params.set(1, 45);
    params.set(2, 80);
    params.set(3, 4);
    params.set(4, 3);
    params.set(5, 3);
    params.set(6, 5);
    params.set(7, 0);
    SeedVR2AWAPack pack;
    require(pack.load_param(params) == 0, "dynamic pack parameters");
    ncnn::Mat video(3, 2, 3, 3600, 4u, 1, 1);
    ncnn::Mat text(3, 2, 3, 5, 4u, 1, 1);
    std::vector<ncnn::Mat> outputs;
    require(pack.forward({video, text}, outputs, ncnn::Option()) == 0, "dynamic pack forward");
    require(outputs.size() == 2 && outputs[0].c == 3645 && outputs[1].w == 10,
            "dynamic pack output shape");
    require(static_cast<int>(outputs[1][outputs[1].w - 1]) == 3645, "dynamic cumulative sequence length");

    std::puts("seedvr2-awa-dynamic: ok");
    return 0;
}
