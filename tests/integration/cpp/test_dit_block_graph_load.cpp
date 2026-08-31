#include <cstdio>
#include <cstdlib>
#include <limits>

#include "awa/awa_layers.h"
#include "net.h"

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 6)
    {
        std::fprintf(stderr, "usage: test_dit_block_graph_load <param> <bin> [source-t source-h source-w]\n");
        return 2;
    }

    ncnn::Net net;
    seedvr2::AwaRuntimeSpec runtime_spec;
    if (argc == 6)
    {
        runtime_spec.source_t = std::atoi(argv[3]);
        runtime_spec.source_h = std::atoi(argv[4]);
        runtime_spec.source_w = std::atoi(argv[5]);
        const long long video_tokens =
            static_cast<long long>(runtime_spec.source_t) * runtime_spec.source_h * runtime_spec.source_w;
        if (video_tokens > std::numeric_limits<int>::max())
        {
            std::fprintf(stderr, "source shape has too many video tokens\n");
            return 2;
        }
        runtime_spec.video_tokens = static_cast<int>(video_tokens);
        if (!runtime_spec.valid())
        {
            std::fprintf(stderr, "source shape must be positive\n");
            return 2;
        }
        register_seedvr2_awa_layers(net, &runtime_spec);
    }
    else
    {
        register_seedvr2_awa_layers(net);
    }
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

    std::puts("seedvr2-dit-block-graph-load: ok");
    return 0;
}
