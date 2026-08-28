#include <cstdio>
#include <vector>

#include "awa/awa_layers.h"
#include "net.h"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: test_awa_graph_load <param> <bin>\n");
        return 2;
    }

    ncnn::Net net;
    register_seedvr2_awa_layers(net);
    if (net.load_param(argv[1]) != 0)
    {
        std::fprintf(stderr, "load_param failed\n");
        return 1;
    }
    if (net.load_model(argv[2]) != 0)
    {
        std::fprintf(stderr, "load_model failed\n");
        return 1;
    }

    constexpr int source_tokens = 2 * 19 * 23;
    constexpr int text_tokens = 5;
    constexpr int heads = 2;
    constexpr int head_dim = 3;
    ncnn::Mat video(head_dim, heads, 3, source_tokens);
    ncnn::Mat text(head_dim, heads, 3, text_tokens);
    for (int token = 0; token < source_tokens; token++)
    {
        float* data = static_cast<float*>(video.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 17 + value) % 101) / 100.f;
    }
    for (int token = 0; token < text_tokens; token++)
    {
        float* data = static_cast<float*>(text.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 23 + value) % 79) / 79.f;
    }
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", video) != 0 || extractor.input("in1", text) != 0)
    {
        std::fprintf(stderr, "extractor input failed\n");
        return 1;
    }
    ncnn::Mat output_video;
    ncnn::Mat output_text;
    const int output_video_ret = extractor.extract("out0", output_video);
    const int output_text_ret = extractor.extract("out1", output_text);
    if (output_video_ret != 0 || output_text_ret != 0)
    {
        std::fprintf(stderr, "extractor output failed: out0=%d out1=%d\n", output_video_ret, output_text_ret);
        return 1;
    }
    if (output_video.empty() || output_text.empty() || output_video.dims != 4 || output_video.w != head_dim ||
        output_video.h != heads || output_video.d != 23 || output_video.c != 19 || output_video.n != 2 ||
        output_text.dims != 3 || output_text.w != head_dim || output_text.h != heads || output_text.c != text_tokens)
    {
        std::fprintf(stderr, "extractor returned unexpected output shape\n");
        return 1;
    }
    std::puts("seedvr2-awa-graph-load: ok");
    return 0;
}
