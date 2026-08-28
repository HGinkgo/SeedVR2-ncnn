#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

#include "cli/cli.h"
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
    require(options.width == 0 && options.height == 0, "automatic resolution defaults");
    require(options.gpu_id == -1, "automatic GPU default");

    const seedvr2::CliOptions explicit_size = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--width", "720", "--height", "1280", "--gpu-id", "1"},
        error);
    require(explicit_size.width == 720 && explicit_size.height == 1280, "explicit dimensions");
    require(explicit_size.gpu_id == 1, "explicit GPU id");

    const seedvr2::CliOptions automatic_gpu = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--gpu-id", "-1"}, error);
    require(automatic_gpu.gpu_id == -1, "explicit automatic GPU id");

    seedvr2::ResolutionPlan automatic_plan;
    require(seedvr2::make_image_resolution_plan(options, 1280, 720, automatic_plan, error), error.c_str());
    require(automatic_plan.image_width == 1280 && automatic_plan.image_height == 720,
            "automatic CLI resolution plan");

    seedvr2::ResolutionPlan explicit_plan;
    require(seedvr2::make_image_resolution_plan(explicit_size, 100, 100, explicit_plan, error), error.c_str());
    require(explicit_plan.image_width == 720 && explicit_plan.image_height == 1280,
            "explicit CLI resolution plan");

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

    require(seedvr2::validate_model_directory(std::filesystem::current_path(), error), "existing model directory");
    const std::filesystem::path missing = std::filesystem::temp_directory_path() / "seedvr2-ncnn-missing-model-dir";
    std::filesystem::remove_all(missing);
    require(!seedvr2::validate_model_directory(missing, error), "missing model directory rejected");

    std::puts("seedvr2-cli: ok");
    return 0;
}
