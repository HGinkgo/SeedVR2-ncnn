#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace seedvr2
{

struct ResolutionPlan;

enum class CliAction
{
    Run,
    Help,
    Version,
};

struct CliOptions final
{
    CliAction action = CliAction::Run;
    std::filesystem::path model_dir = "models/seedvr2-3b";
    std::filesystem::path input;
    std::filesystem::path output = "out.png";
    std::vector<std::filesystem::path> inputs;
    std::vector<std::filesystem::path> outputs;
    int width = 0;
    int height = 0;
    int gpu_id = -1;
    std::uint32_t memory_budget_mib = 0;
    bool profile = false;
    bool color_fix = false;
};

bool parse_cli(int argc, const char* const argv[], CliOptions& options, std::string& error);
bool validate_model_directory(const std::filesystem::path& model_dir, std::string& error);
bool make_image_resolution_plan(const CliOptions& options, int input_width, int input_height,
                                ResolutionPlan& plan, std::string& error);

} // namespace seedvr2
