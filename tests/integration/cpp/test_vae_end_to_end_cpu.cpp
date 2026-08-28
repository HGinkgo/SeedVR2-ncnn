#include <cstdio>

#include "net.h"
#include "vae/temporal_pad.h"

namespace
{

bool load_graph(ncnn::Net& net, const char* param_path, const char* model_path)
{
    register_seedvr2_vae_layers(net);
    return net.load_param(param_path) == 0 && net.load_model(model_path) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: test_vae_end_to_end_cpu <encode.param> <encode.bin> <decode.param> <decode.bin>\n");
        return 2;
    }

    ncnn::Net encode;
    ncnn::Net decode;
    if (!load_graph(encode, argv[1], argv[2]) || !load_graph(decode, argv[3], argv[4]))
    {
        std::fprintf(stderr, "load VAE graph failed\n");
        return 1;
    }

    ncnn::Mat sample(128, 128, 1, 3);
    sample.fill(0.f);
    ncnn::Extractor encode_extractor = encode.create_extractor();
    if (encode_extractor.input("in0", sample) != 0)
        return 1;
    ncnn::Mat latent;
    if (encode_extractor.extract("out0", latent) != 0 || latent.empty() || latent.total() != 16 * 16 * 16)
    {
        std::fprintf(stderr, "encode output has an unexpected shape\n");
        return 1;
    }

    ncnn::Extractor decode_extractor = decode.create_extractor();
    if (decode_extractor.input("in0", latent) != 0)
        return 1;
    ncnn::Mat reconstruction;
    if (decode_extractor.extract("out0", reconstruction) != 0 || reconstruction.empty() ||
        reconstruction.total() != 3 * 128 * 128)
    {
        ncnn::Extractor debug_extractor = decode.create_extractor();
        ncnn::Mat pre_squeeze;
        if (debug_extractor.input("in0", latent) == 0 && debug_extractor.extract("453", pre_squeeze) == 0)
            std::fprintf(stderr, "pre-squeeze dims=%d w=%d h=%d d=%d c=%d n=%d total=%zu\n",
                         pre_squeeze.dims, pre_squeeze.w, pre_squeeze.h, pre_squeeze.d,
                         pre_squeeze.c, pre_squeeze.n, pre_squeeze.total());
        std::fprintf(stderr, "decode output has an unexpected shape dims=%d w=%d h=%d d=%d c=%d n=%d total=%zu\n",
                     reconstruction.dims, reconstruction.w, reconstruction.h, reconstruction.d,
                     reconstruction.c, reconstruction.n, reconstruction.total());
        return 1;
    }

    std::puts("seedvr2-vae-end-to-end-cpu: ok");
    return 0;
}
