#include <cstdio>
#include <cstring>
#include <string>

#include "net.h"
#include "platform.h"
#include "awa/awa_layers.h"
#include "cli/cli.h"
#include "inference/image_inference.h"
#include "io/image_io.h"
#include "model/model_registry.h"
#include "resolution/resolution_plan.h"
#include "vae/temporal_pad.h"

namespace
{

void print_usage()
{
    std::puts("Usage: seedvr2-ncnn [options]");
    std::puts("");
    std::puts("Options:");
    std::puts("  --help       Show this help message");
    std::puts("  --version    Show SeedVR2-ncnn and ncnn versions");
    std::puts("  --model-dir  Model directory (default: models/seedvr2-3b)");
    std::puts("  --input      Input image path");
    std::puts("  --output     Output image path (default: out.png)");
    std::puts("  --width      Explicit output width (optional)");
    std::puts("  --height     Explicit output height (optional)");
    std::puts("  --gpu-id     Vulkan GPU id, -1 selects automatically (default: -1)");
}

void print_version()
{
    // Constructing Net keeps the CLI smoke test tied to the linked ncnn API.
    ncnn::Net net;
    register_seedvr2_awa_layers(net);
    register_seedvr2_vae_layers(net);
    std::printf("SeedVR2-ncnn %s\n", SEEDVR2_VERSION);
    std::printf("ncnn %s\n", NCNN_VERSION_STRING);
}

} // namespace

int main(int argc, char** argv)
{
    seedvr2::CliOptions options;
    std::string error;
    if (!seedvr2::parse_cli(argc, argv, options, error))
    {
        std::fprintf(stderr, "error: %s\n\n", error.c_str());
        print_usage();
        return 2;
    }

    if (options.action == seedvr2::CliAction::Help)
    {
        print_usage();
        return 0;
    }

    if (options.action == seedvr2::CliAction::Version)
    {
        print_version();
        return 0;
    }

    seedvr2::ModelRegistry registry;
    if (!seedvr2::ModelRegistry::open(options.model_dir, registry, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    seedvr2::RgbImage input_image;
    if (!seedvr2::load_rgb_image(options.input, input_image, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    seedvr2::ResolutionPlan resolution_plan;
    if (!seedvr2::make_image_resolution_plan(options, input_image.width, input_image.height,
                                             resolution_plan, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "input=%dx%d target=%dx%d\n", input_image.width, input_image.height,
                 resolution_plan.image_width, resolution_plan.image_height);

    seedvr2::ModelGraphSet graphs;
    if (!registry.resolve(resolution_plan, graphs, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    seedvr2::RgbImage output_image;
    if (!seedvr2::run_image_inference(graphs, input_image, resolution_plan, options.gpu_id, output_image, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (!seedvr2::save_rgb_image(options.output, output_image, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stderr, "output=%s\n", options.output.string().c_str());
    std::puts("seedvr2-image-inference: ok");
    return 0;
}
