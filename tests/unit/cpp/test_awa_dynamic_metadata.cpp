#include <cstdio>
#include <cstdlib>
#include <string>

#include "awa/awa_layers.h"
#include "net.h"

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

void require_runtime_awa_pack(int source_h, int source_w, int expected_sequence_tokens)
{
    const seedvr2::AwaRuntimeSpec runtime_spec = {1, source_h, source_w, source_h * source_w};
    ncnn::Net net;
    register_seedvr2_awa_layers(net, &runtime_spec);

    static const char kDynamicPackParam[] =
        "7767517\n"
        "3 4\n"
        "Input video 0 1 video\n"
        "Input text 0 1 text\n"
        "SeedVR2AWAPack pack 2 2 video text packed lengths 0=-1 1=-1 2=-1 3=4 4=3 5=3 6=5 7=0\n";
    require(net.load_param_mem(kDynamicPackParam) == 0, "runtime AWA pack graph load");

    ncnn::Mat video(3, 2, 3, runtime_spec.video_tokens, 4u, 1, 1);
    ncnn::Mat text(3, 2, 3, 5, 4u, 1, 1);
    ncnn::Extractor extractor = net.create_extractor();
    require(extractor.input("video", video) == 0 && extractor.input("text", text) == 0,
            "runtime AWA pack inputs");

    ncnn::Mat packed;
    ncnn::Mat lengths;
    require(extractor.extract("packed", packed) == 0 && extractor.extract("lengths", lengths) == 0,
            "runtime AWA pack outputs");
    require(packed.dims == 3 && packed.c == expected_sequence_tokens,
            "runtime AWA packed sequence length");
    require(lengths.w > 0 && static_cast<int>(lengths[lengths.w - 1]) == expected_sequence_tokens,
            "runtime AWA cumulative sequence length");
}

void require_runtime_awa_layer_load(const char* layer_type, int top_count)
{
    const seedvr2::AwaRuntimeSpec runtime_spec = {1, 45, 80, 3600};
    ncnn::Net net;
    register_seedvr2_awa_layers(net, &runtime_spec);

    const int blob_count = 1 + top_count;
    std::string param = "7767517\n2 " + std::to_string(blob_count) + "\n";
    param += "Input input 0 1 input\n";
    param += layer_type;
    param += " dynamic 1 ";
    param += std::to_string(top_count);
    param += " input output0";
    if (top_count == 2)
        param += " output1";
    param += " 0=-1 1=-1 2=-1 3=4 4=3 5=3 6=5 7=0 8=126\n";
    require(net.load_param_mem(param.c_str()) == 0, "runtime AWA layer graph load");
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

    require_runtime_awa_pack(8, 8, 69);
    require_runtime_awa_pack(16, 16, 261);
    require_runtime_awa_pack(45, 80, 3645);
    require_runtime_awa_layer_load("SeedVR2AWAUnpack", 2);
    require_runtime_awa_layer_load("SeedVR2MMRoPE", 1);
    require_runtime_awa_layer_load("SeedVR2WindowAttention", 1);

    std::puts("seedvr2-awa-dynamic: ok");
    return 0;
}
