#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "net.h"
#include "platform.h"
#include "awa/awa_layers.h"
#include "cli/cli.h"
#include "inference/image_inference.h"
#include "inference/performance_profile.h"
#include "io/image_io.h"
#include "model/model_registry.h"
#include "resolution/resolution_plan.h"
#include "vae/temporal_pad.h"
#include "video/video_io.h"

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
    std::puts("  --input      Input image path (repeat up to 2 times)");
    std::puts("  --output     Output image path (repeat up to 2 times; default: out.png for one image)");
    std::puts("  --width      Explicit output width (optional; target area <= 256x256)");
    std::puts("  --height     Explicit output height (optional; target area <= 256x256)");
    std::puts("  --gpu-id     Vulkan GPU id, -1 selects automatically (default: -1)");
    std::puts("  --memory-budget-mib  Minimum Vulkan heap budget for preflight (default: 0, disabled)");
    std::puts("  --profile     Print opt-in stage timings to stderr (default: off)");
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

bool has_video_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    for (char& value : extension)
        if (value >= 'A' && value <= 'Z')
            value = static_cast<char>(value - 'A' + 'a');
    return extension == ".avi" || extension == ".mp4" || extension == ".m4v" || extension == ".mkv" ||
           extension == ".mov" || extension == ".webm";
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

    // Profiling is opt-in. When it is off the profile object stays disabled and
    // every measurement below becomes a no-op.
    const seedvr2::PerformanceProfile profile(options.profile);
    const auto run_start = seedvr2::PerformanceProfile::Clock::now();

    seedvr2::ModelRegistry registry;
    if (!seedvr2::ModelRegistry::open(options.model_dir, registry, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    if (has_video_extension(options.input))
    {
        if (options.inputs.size() != 1 || options.outputs.size() != 1)
        {
            std::fprintf(stderr, "error: video input accepts exactly one input and output path\n");
            return 1;
        }
        seedvr2::VideoReader reader;
        if (!seedvr2::VideoReader::open(options.input, reader, error))
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }

        seedvr2::ResolutionPlan resolution_plan;
        if (!seedvr2::make_image_resolution_plan(options, reader.info().width, reader.info().height,
                                                 resolution_plan, error))
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        if (!has_video_extension(options.output) || options.output.extension() != ".avi")
        {
            std::fprintf(stderr, "error: AVI input requires an AVI output path\n");
            return 1;
        }
        std::fprintf(stderr, "input-video=%dx%d frames=%d target=%dx%d\n", reader.info().width,
                     reader.info().height, reader.info().frame_count, resolution_plan.image_width,
                     resolution_plan.image_height);

        seedvr2::ModelGraphSet graphs;
        if (!registry.resolve(resolution_plan, graphs, error))
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        seedvr2::VideoInfo output_info = reader.info();
        output_info.width = resolution_plan.image_width;
        output_info.height = resolution_plan.image_height;
        seedvr2::AviVideoWriter writer;
        if (!seedvr2::AviVideoWriter::open(options.output, output_info, writer, error))
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }

        seedvr2::ImageInferenceSession session;
        if (!seedvr2::ImageInferenceSession::open(graphs, resolution_plan, options.gpu_id, session, error,
                                                  options.memory_budget_mib, &profile))
        {
            std::fprintf(stderr, "error: stage=video-inference-init: %s\n", error.c_str());
            return 1;
        }

        std::fprintf(stderr, "stage=video-batch start=0 frames=%d\n", reader.info().frame_count);
        std::size_t processed_frames = 0;
        const bool video_ok = session.run_video(
            [&](seedvr2::RgbImage& frame, std::string& read_error) {
                return reader.read_next(frame, read_error);
            },
            [&](const seedvr2::RgbImage& frame, std::string& write_error) {
                return writer.write_frame(frame, write_error);
            }, processed_frames, error);
        if (!video_ok)
        {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        if (!writer.close(error))
        {
            std::fprintf(stderr, "error: stage=video-encode failed: %s\n", error.c_str());
            return 1;
        }
        if (processed_frames == 0)
        {
            std::fprintf(stderr, "error: stage=video-decode failed: video contains no decodable frames\n");
            return 1;
        }
        std::fprintf(stderr, "output=%s frames=%zu\n", options.output.string().c_str(), processed_frames);
        profile.report_total(profile.elapsed_ms(run_start));
        std::puts("seedvr2-video-inference: ok");
        return 0;
    }

    std::vector<seedvr2::RgbImage> input_images;
    {
        const seedvr2::ProfileScope read_scope(profile, "image-read");
        input_images.reserve(options.inputs.size());
        for (const auto& input_path : options.inputs)
        {
            seedvr2::RgbImage image;
            if (!seedvr2::load_rgb_image(input_path, image, error))
            {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                return 1;
            }
            input_images.push_back(std::move(image));
        }
    }

    seedvr2::ResolutionPlan resolution_plan;
    if (!seedvr2::make_image_resolution_plan(options, input_images.front().width, input_images.front().height,
                                             resolution_plan, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (input_images.size() == 1)
        std::fprintf(stderr, "input=%dx%d target=%dx%d\n", input_images.front().width, input_images.front().height,
                     resolution_plan.image_width, resolution_plan.image_height);
    else
        std::fprintf(stderr, "input=%dx%d target=%dx%d frames=%zu\n", input_images.front().width,
                     input_images.front().height, resolution_plan.image_width, resolution_plan.image_height,
                     input_images.size());

    seedvr2::ModelGraphSet graphs;
    if (!registry.resolve(resolution_plan, graphs, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    seedvr2::ImageInferenceSession session;
    if (!seedvr2::ImageInferenceSession::open(graphs, resolution_plan, options.gpu_id, session, error,
                                              options.memory_budget_mib, &profile))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::vector<seedvr2::RgbImage> output_images;
    if (!session.run_batch(input_images, output_images, error))
    {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (output_images.size() != options.outputs.size())
    {
        std::fprintf(stderr, "error: inference returned %zu outputs for %zu inputs\n", output_images.size(),
                     options.outputs.size());
        return 1;
    }
    {
        const seedvr2::ProfileScope write_scope(profile, "image-write");
        for (std::size_t index = 0; index < output_images.size(); ++index)
        {
            if (!seedvr2::save_rgb_image(options.outputs[index], output_images[index], error))
            {
                std::fprintf(stderr, "error: %s\n", error.c_str());
                return 1;
            }
        }
    }

    for (const auto& output_path : options.outputs)
        std::fprintf(stderr, "output=%s\n", output_path.string().c_str());
    profile.report_total(profile.elapsed_ms(run_start));
    std::puts("seedvr2-image-inference: ok");
    return 0;
}
