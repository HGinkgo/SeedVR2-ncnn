#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

#include "cli/cli.h"
#include "inference/memory_diagnostics.h"
#include "inference/performance_profile.h"
#include "resolution/resolution_plan.h"

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

seedvr2::CliOptions parse(std::initializer_list<const char*> arguments, std::string& error)
{
    std::vector<const char*> argv(arguments);
    seedvr2::CliOptions options;
    require(seedvr2::parse_cli(static_cast<int>(argv.size()), argv.data(), options, error), error.c_str());
    return options;
}

} // namespace

int main()
{
    std::string error;
    const seedvr2::CliOptions help = parse({"seedvr2-ncnn", "--help"}, error);
    require(help.action == seedvr2::CliAction::Help, "help action");

    const seedvr2::CliOptions options = parse(
        {"seedvr2-ncnn", "--model-dir", "models/seedvr2-3b", "--input", "input.png", "--output", "result.png"},
        error);
    require(options.action == seedvr2::CliAction::Run, "run action");
    require(options.model_dir == std::filesystem::path("models/seedvr2-3b"), "model directory");
    require(options.input == std::filesystem::path("input.png"), "input path");
    require(options.output == std::filesystem::path("result.png"), "output path");
    require(options.inputs == std::vector<std::filesystem::path>{"input.png"}, "single input list");
    require(options.outputs == std::vector<std::filesystem::path>{"result.png"}, "single output list");
    require(options.width == 0 && options.height == 0, "automatic resolution defaults");
    require(options.gpu_id == -1, "automatic GPU default");
    require(options.memory_budget_mib == 0, "automatic memory budget default");

    const seedvr2::CliOptions default_output = parse({"seedvr2-ncnn", "--input", "input.png"}, error);
    require(default_output.outputs == std::vector<std::filesystem::path>{"out.png"}, "default output list");

    const seedvr2::CliOptions image_batch = parse(
        {"seedvr2-ncnn", "--input", "first.png", "--input", "second.png", "--output", "first-out.png",
         "--output", "second-out.png"},
        error);
    require(image_batch.inputs == std::vector<std::filesystem::path>{"first.png", "second.png"},
            "two image inputs");
    require(image_batch.outputs == std::vector<std::filesystem::path>{"first-out.png", "second-out.png"},
            "two image outputs");
    require(image_batch.input == std::filesystem::path("first.png"), "batch keeps first input alias");
    require(image_batch.output == std::filesystem::path("first-out.png"), "batch keeps first output alias");

    const seedvr2::CliOptions explicit_size = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--width", "256", "--height", "256", "--gpu-id", "1"},
        error);
    require(explicit_size.width == 256 && explicit_size.height == 256, "explicit dimensions");
    require(explicit_size.gpu_id == 1, "explicit GPU id");

    const seedvr2::CliOptions automatic_gpu = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--gpu-id", "-1"}, error);
    require(automatic_gpu.gpu_id == -1, "explicit automatic GPU id");

    const seedvr2::CliOptions memory_budget = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--memory-budget-mib", "4096"}, error);
    require(memory_budget.memory_budget_mib == 4096, "explicit memory budget");

    // --profile is opt-in and must not perturb any other default.
    require(!options.profile && !explicit_size.profile, "profile disabled by default");

    const seedvr2::CliOptions profiling = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--profile"}, error);
    require(profiling.action == seedvr2::CliAction::Run, "profile keeps the run action");
    require(profiling.profile, "profile enabled on request");
    require(profiling.model_dir == options.model_dir, "profile keeps the default model directory");
    require(profiling.input == options.input, "profile keeps the input path");
    require(profiling.width == 0 && profiling.height == 0, "profile keeps automatic resolution");
    require(profiling.gpu_id == options.gpu_id, "profile keeps the automatic GPU default");
    require(profiling.memory_budget_mib == 0, "profile keeps the automatic memory budget");

    const seedvr2::CliOptions profile_with_resolution = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--width", "128", "--height", "128", "--profile"},
        error);
    require(profile_with_resolution.profile, "profile enabled next to explicit dimensions");
    require(profile_with_resolution.width == 128 && profile_with_resolution.height == 128,
            "profile keeps explicit dimensions");

    seedvr2::ResolutionPlan automatic_plan;
    require(seedvr2::make_image_resolution_plan(options, 1280, 720, automatic_plan, error), error.c_str());
    require(automatic_plan.image_width == 336 && automatic_plan.image_height == 192,
            "automatic CLI low-resolution plan");

    seedvr2::ResolutionPlan small_automatic_plan;
    require(seedvr2::make_image_resolution_plan(options, 128, 128, small_automatic_plan, error), error.c_str());
    require(small_automatic_plan.image_width == 128 && small_automatic_plan.image_height == 128,
            "automatic CLI does not upscale small input");

    seedvr2::ResolutionPlan tiny_automatic_plan;
    require(seedvr2::make_image_resolution_plan(options, 2, 1, tiny_automatic_plan, error), error.c_str());
    require(tiny_automatic_plan.image_width == 32 && tiny_automatic_plan.image_height == 16,
            "automatic CLI preserves legal alignment for tiny input");

    seedvr2::ResolutionPlan explicit_plan;
    require(seedvr2::make_image_resolution_plan(explicit_size, 100, 100, explicit_plan, error), error.c_str());
    require(explicit_plan.image_width == 256 && explicit_plan.image_height == 256,
            "explicit CLI resolution plan");

    const seedvr2::CliOptions over_limit = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--width", "320", "--height", "256"}, error);
    seedvr2::ResolutionPlan rejected_plan;
    require(!seedvr2::make_image_resolution_plan(over_limit, 100, 100, rejected_plan, error),
            "explicit low-resolution limit");
    require(error.find("65536") != std::string::npos, "low-resolution limit error");

    seedvr2::CliOptions rejected;
    const char* missing_height[] = {"seedvr2-ncnn", "--input", "input.png", "--width", "720"};
    require(!seedvr2::parse_cli(5, missing_height, rejected, error), "width without height rejected");
    require(error.find("height") != std::string::npos, "missing height error");

    const char* zero_width[] = {"seedvr2-ncnn", "--input", "input.png", "--width", "0", "--height", "128"};
    require(!seedvr2::parse_cli(7, zero_width, rejected, error), "zero width rejected");

    const char* missing_input[] = {"seedvr2-ncnn", "--model-dir", "models"};
    require(!seedvr2::parse_cli(3, missing_input, rejected, error), "missing input rejected");

    const char* unknown_option[] = {"seedvr2-ncnn", "--bogus", "value"};
    require(!seedvr2::parse_cli(3, unknown_option, rejected, error), "unknown option rejected");

    const char* too_many_inputs[] = {"seedvr2-ncnn", "--input", "one.png", "--input", "two.png",
                                     "--input", "three.png", "--output", "out.png"};
    require(!seedvr2::parse_cli(9, too_many_inputs, rejected, error), "more than two image inputs rejected");
    require(error.find("at most 2") != std::string::npos, "image input limit error");

    const char* mismatched_outputs[] = {"seedvr2-ncnn", "--input", "one.png", "--input", "two.png",
                                        "--output", "out.png"};
    require(!seedvr2::parse_cli(7, mismatched_outputs, rejected, error), "image output count mismatch rejected");
    require(error.find("same number") != std::string::npos, "image output count error");

    const char* negative_budget[] = {"seedvr2-ncnn", "--input", "input.png", "--memory-budget-mib", "-1"};
    require(!seedvr2::parse_cli(5, negative_budget, rejected, error), "negative memory budget rejected");
    require(error.find("memory-budget-mib") != std::string::npos, "memory budget error");

    const char* profile_with_value[] = {"seedvr2-ncnn", "--input", "input.png", "--profile", "yes"};
    require(!seedvr2::parse_cli(5, profile_with_value, rejected, error), "--profile takes no value");

    seedvr2::VulkanMemoryDiagnostics diagnostics;
    diagnostics.gpu_id = 0;
    diagnostics.device_name = "Test GPU";
    diagnostics.heap_budget_mib = 2048;
    diagnostics.max_allocation_mib = 4094;
    diagnostics.target_width = 128;
    diagnostics.target_height = 256;
    const std::string formatted = seedvr2::format_vulkan_stage_error(
        "vae-decode", diagnostics, "ncnn operation returned failure");
    require(formatted.find("stage=vae-decode failed") != std::string::npos, "diagnostic stage");
    require(formatted.find("gpu=0 Test GPU") != std::string::npos, "diagnostic GPU");
    require(formatted.find("heap-budget-mib=2048") != std::string::npos, "diagnostic heap budget");
    require(formatted.find("max-allocation-mib=4094") != std::string::npos, "diagnostic allocation limit");
    require(formatted.find("target=128x256") != std::string::npos, "diagnostic target");
    require(formatted.find("reduce --width/--height") != std::string::npos, "diagnostic advice");

    const std::string preflight = seedvr2::format_vulkan_memory_preflight_error(diagnostics, 4096);
    require(preflight.find("requested minimum 4096 MiB") != std::string::npos, "preflight request");
    require(preflight.find("device heap budget 2048 MiB") != std::string::npos, "preflight budget");
    require(preflight.find("lower --memory-budget-mib") != std::string::npos, "preflight advice");

    // The profile lines use `name=` rather than `stage=` so that they can never be
    // mistaken for the existing `stage=` progress lines of the product path.
    const std::string stage_line = seedvr2::format_profile_line("vae-encode", 123.5);
    require(stage_line == "profile name=vae-encode ms=123.5", "profile stage line");
    require(stage_line.find("stage=") == std::string::npos, "profile line avoids the stage= keyword");

    // Times print with one decimal place ("%.1f"); 45.25 rounds to 45.2.
    const std::string frame_line = seedvr2::format_profile_line("vae-encode", "frame", 7, 45.25);
    require(frame_line == "profile name=vae-encode frame=7 ms=45.2", "profile frame line");

    const std::string batch_line = seedvr2::format_profile_line("video-batch", "frames", 2, 9876.5);
    require(batch_line == "profile name=video-batch frames=2 ms=9876.5", "profile batch line");

    const std::string total_line = seedvr2::format_profile_total_line(13579.0, 2913);
    require(total_line == "profile name=total ms=13579.0 peak-rss-mib=2913", "profile total line");

    const std::string dit_param_line = seedvr2::format_profile_dit_load_line("param", 12.25);
    require(dit_param_line == "profile name=dit-param-load ms=12.2", "DiT param load profile line");
    const std::string dit_bin_line = seedvr2::format_profile_dit_load_line("bin", 987.66);
    require(dit_bin_line == "profile name=dit-bin-load ms=987.7", "DiT bin load profile line");

    seedvr2::PerformanceProfile disabled;
    require(!disabled.enabled(), "profile construct disabled by default");

    const seedvr2::PerformanceProfile enabled(true);
    require(enabled.enabled(), "profile construct enabled on request");
    require(enabled.elapsed_ms(seedvr2::PerformanceProfile::Clock::now()) >= 0.0,
            "profile elapsed is non-negative");

    require(seedvr2::validate_model_directory(std::filesystem::current_path(), error), "existing model directory");
    const std::filesystem::path missing = std::filesystem::temp_directory_path() / "seedvr2-ncnn-missing-model-dir";
    std::filesystem::remove_all(missing);
    require(!seedvr2::validate_model_directory(missing, error), "missing model directory rejected");

    std::puts("seedvr2-cli: ok");
    return 0;
}
