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
    CheckModel,
};

struct CliOptions final
{
    CliAction action = CliAction::Run;
    std::filesystem::path model_dir = "models/seedvr2-3b";
    std::filesystem::path input;
    std::filesystem::path output = "out.png";
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> inputs;
    std::vector<std::filesystem::path> outputs;
    int width = 0;
    int height = 0;
    int scale = 0;
    int start_frame = 0;
    int frame_count = 0;
    int vae_tile_size = 0;
    int gpu_id = -1;
    std::uint32_t memory_budget_mib = 0;
    bool profile = false;
};

bool parse_cli(int argc, const char* const argv[], CliOptions& options, std::string& error);
bool validate_model_directory(const std::filesystem::path& model_dir, std::string& error);
bool make_image_resolution_plan(const CliOptions& options, int input_width, int input_height,
                                ResolutionPlan& plan, std::string& error);

// Return the selected frame count, or -1 when an unknown-length source is
// intentionally left unbounded.
int select_video_frame_count(int source_frame_count, int start_frame, int requested_frame_count);

} // namespace seedvr2
