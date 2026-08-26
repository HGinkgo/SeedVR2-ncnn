#include <cstdio>

#include "awa/awa_layers.h"
#include "net.h"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: test_dit_block_graph_load <param> <bin>\n");
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

    std::puts("seedvr2-dit-block-graph-load: ok");
    return 0;
}
