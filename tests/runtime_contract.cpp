#include "cli/cli.h"
#include "resolution/resolution_plan.h"
#include "video/video_io.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

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

void check_cli_contract()
{
    std::string error;
    const seedvr2::CliOptions help = parse({"seedvr2-ncnn", "--help"}, error);
    require(help.action == seedvr2::CliAction::Help, "help action");

    const seedvr2::CliOptions options = parse(
        {"seedvr2-ncnn", "--model-dir", "models/seedvr2-3b", "--input", "input.png", "--output",
         "result.png", "--width", "256", "--height", "256", "--gpu-id", "0", "--memory-budget-mib",
         "4096"},
        error);
    require(options.action == seedvr2::CliAction::Run, "run action");
    require(options.model_dir == std::filesystem::path("models/seedvr2-3b"), "model directory");
    require(options.input == std::filesystem::path("input.png"), "input path");
    require(options.output == std::filesystem::path("result.png"), "output path");
    require(options.width == 256 && options.height == 256, "supported fixed shape");
    require(options.gpu_id == 0 && options.memory_budget_mib == 4096, "runtime resource options");

    seedvr2::ResolutionPlan plan;
    require(seedvr2::make_image_resolution_plan(options, 100, 100, plan, error), error.c_str());
    require(plan.image_width == 256 && plan.image_height == 256, "explicit resolution plan");

    const seedvr2::CliOptions automatic = parse({"seedvr2-ncnn", "--input", "input.png"}, error);
    require(seedvr2::make_image_resolution_plan(automatic, 1280, 720, plan, error), error.c_str());
    require(plan.image_width == 336 && plan.image_height == 192, "automatic low-resolution plan");

    seedvr2::CliOptions rejected;
    const char* missing_height[] = {"seedvr2-ncnn", "--input", "input.png", "--width", "256"};
    require(!seedvr2::parse_cli(5, missing_height, rejected, error), "width without height rejected");

    const char* invalid_tile[] = {"seedvr2-ncnn", "--input", "input.png", "--vae-tile-size", "50"};
    require(!seedvr2::parse_cli(5, invalid_tile, rejected, error), "misaligned VAE tile rejected");

    const seedvr2::CliOptions over_limit = parse(
        {"seedvr2-ncnn", "--input", "input.png", "--width", "320", "--height", "256"}, error);
    require(!seedvr2::make_image_resolution_plan(over_limit, 100, 100, plan, error), "product area limit");
    require(error.find("65536") != std::string::npos, "product area limit message");
}

void check_avi_contract()
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "seedvr2-runtime-contract.avi";
    std::string error;
    seedvr2::VideoInfo info;
    info.width = 2;
    info.height = 2;
    info.fps_num = 24;

    seedvr2::AviVideoWriter writer;
    require(seedvr2::AviVideoWriter::open(path, info, writer, error), "open AVI writer");
    seedvr2::RgbImage first;
    first.width = 2;
    first.height = 2;
    first.pixels = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    seedvr2::RgbImage second = first;
    second.pixels[0] = 12;
    require(writer.write_frame(first, error), "write first frame");
    require(writer.write_frame(second, error), "write second frame");
    require(writer.close(error), "close AVI writer");

    seedvr2::VideoReader reader;
    require(seedvr2::VideoReader::open(path, reader, error), "open AVI reader");
    require(reader.info().width == 2 && reader.info().height == 2, "AVI dimensions");
    require(reader.info().frame_count == 2 && reader.info().fps_num == 24, "AVI metadata");
    seedvr2::RgbImage decoded;
    require(reader.read_next(decoded, error) && decoded.pixels == first.pixels, "first AVI frame");
    require(reader.read_next(decoded, error) && decoded.pixels == second.pixels, "second AVI frame");
    require(!reader.read_next(decoded, error) && error.empty(), "AVI end of stream");
    std::remove(path.string().c_str());
}

} // namespace

int main()
{
    check_cli_contract();
    check_avi_contract();
    std::puts("seedvr2-runtime-contract: ok");
    return 0;
}
