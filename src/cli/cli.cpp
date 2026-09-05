#include "cli.h"

#include "resolution/resolution_plan.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace seedvr2
{
namespace
{

constexpr int kProductMaxArea = 256 * 256;

bool parse_integer(const char* text, int& value)
{
    if (!text || *text == '\0')
        return false;

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX)
        return false;

    value = static_cast<int>(parsed);
    return true;
}

bool next_value(int argc, const char* const argv[], int& index, const char*& value, std::string& error,
                bool allow_leading_dash = false)
{
    if (index + 1 >= argc || (!allow_leading_dash && argv[index + 1][0] == '-'))
    {
        error = std::string(argv[index]) + " requires a value";
        return false;
    }
    value = argv[++index];
    return true;
}

} // namespace

bool parse_cli(int argc, const char* const argv[], CliOptions& options, std::string& error)
{
    options = CliOptions();
    error.clear();
    if (argc <= 0 || !argv)
    {
        error = "invalid argument vector";
        return false;
    }
    if (argc == 1)
    {
        options.action = CliAction::Help;
        return true;
    }

    bool width_set = false;
    bool height_set = false;
    bool scale_set = false;
    bool output_set = false;
    for (int index = 1; index < argc; ++index)
    {
        const char* argument = argv[index];
        if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0)
        {
            options.action = CliAction::Help;
            continue;
        }
        if (std::strcmp(argument, "--version") == 0)
        {
            options.action = CliAction::Version;
            continue;
        }
        if (std::strcmp(argument, "--check-model") == 0)
        {
            options.action = CliAction::CheckModel;
            continue;
        }
        if (std::strcmp(argument, "--profile") == 0)
        {
            options.profile = true;
            continue;
        }
        const char* value = nullptr;
        if (std::strcmp(argument, "--model-dir") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            options.model_dir = value;
        }
        else if (std::strcmp(argument, "--input") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            if (options.inputs.size() >= 2)
            {
                error = "--input accepts at most 2 paths";
                return false;
            }
            options.inputs.emplace_back(value);
            if (options.input.empty())
                options.input = value;
        }
        else if (std::strcmp(argument, "--input-dir") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            options.input_dir = value;
        }
        else if (std::strcmp(argument, "--output") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            if (options.outputs.size() >= 2)
            {
                error = "--output accepts at most 2 paths";
                return false;
            }
            options.outputs.emplace_back(value);
            if (!output_set)
            {
                options.output = value;
                output_set = true;
            }
        }
        else if (std::strcmp(argument, "--output-dir") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            options.output_dir = value;
        }
        else if (std::strcmp(argument, "--width") == 0 || std::strcmp(argument, "--height") == 0)
        {
            const bool is_width = std::strcmp(argument, "--width") == 0;
            if (!next_value(argc, argv, index, value, error))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed <= 0)
            {
                error = std::string(argument) + " must be a positive integer";
                return false;
            }
            if (is_width)
            {
                options.width = parsed;
                width_set = true;
            }
            else
            {
                options.height = parsed;
                height_set = true;
            }
        }
        else if (std::strcmp(argument, "--scale") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed <= 0)
            {
                error = "--scale must be a positive integer";
                return false;
            }
            options.scale = parsed;
            scale_set = true;
        }
        else if (std::strcmp(argument, "--start-frame") == 0 || std::strcmp(argument, "--frames") == 0)
        {
            if (!next_value(argc, argv, index, value, error, true))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed < 0 ||
                (std::strcmp(argument, "--frames") == 0 && parsed == 0))
            {
                error = std::string(argument) +
                        (std::strcmp(argument, "--start-frame") == 0 ? " must be a non-negative integer"
                                                                       : " must be a positive integer");
                return false;
            }
            if (std::strcmp(argument, "--start-frame") == 0)
                options.start_frame = parsed;
            else
                options.frame_count = parsed;
        }
        else if (std::strcmp(argument, "--vae-tile-size") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed < 32 || parsed % 16 != 0)
            {
                error = "--vae-tile-size must be a multiple of 16 and at least 32";
                return false;
            }
            options.vae_tile_size = parsed;
        }
        else if (std::strcmp(argument, "--gpu-id") == 0)
        {
            if (!next_value(argc, argv, index, value, error, true))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed < -1)
            {
                error = "--gpu-id must be an integer greater than or equal to -1";
                return false;
            }
            options.gpu_id = parsed;
        }
        else if (std::strcmp(argument, "--memory-budget-mib") == 0)
        {
            if (!next_value(argc, argv, index, value, error, true))
                return false;
            int parsed = 0;
            if (!parse_integer(value, parsed) || parsed < 0)
            {
                error = "--memory-budget-mib must be a non-negative integer";
                return false;
            }
            options.memory_budget_mib = static_cast<std::uint32_t>(parsed);
        }
        else
        {
            error = std::string("unknown option: ") + argument;
            return false;
        }
    }

    if (options.action == CliAction::Help || options.action == CliAction::Version ||
        options.action == CliAction::CheckModel)
        return true;
    if (!options.input_dir.empty() && !options.inputs.empty())
    {
        error = "--input-dir cannot be combined with --input";
        return false;
    }
    if (options.input_dir.empty() && options.inputs.empty())
    {
        error = "--input is required";
        return false;
    }
    if (!options.input_dir.empty())
    {
        if (options.output_dir.empty())
        {
            error = "--input-dir requires --output-dir";
            return false;
        }
        if (!options.outputs.empty())
        {
            error = "--input-dir cannot be combined with --output";
            return false;
        }
    }
    else if (!options.output_dir.empty())
    {
        error = "--output-dir requires --input-dir";
        return false;
    }
    if (options.input_dir.empty() && options.outputs.empty())
    {
        if (options.inputs.size() != 1)
        {
            error = "--input and --output must contain the same number of paths";
            return false;
        }
        options.outputs.push_back(options.output);
    }
    if (options.input_dir.empty() && options.inputs.size() != options.outputs.size())
    {
        error = "--input and --output must contain the same number of paths";
        return false;
    }
    if (width_set != height_set)
    {
        error = "--width and --height must be provided together";
        return false;
    }
    if (scale_set && (width_set || height_set))
    {
        error = "--scale cannot be combined with --width/--height";
        return false;
    }
    return true;
}

bool validate_model_directory(const std::filesystem::path& model_dir, std::string& error)
{
    error.clear();
    if (model_dir.empty())
    {
        error = "model directory is empty";
        return false;
    }

    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(model_dir, filesystem_error))
    {
        error = "model directory does not exist or is not a directory: " + model_dir.string();
        return false;
    }
    return true;
}

bool make_image_resolution_plan(const CliOptions& options, int input_width, int input_height,
                                ResolutionPlan& plan, std::string& error)
{
    if (input_width <= 0 || input_height <= 0)
    {
        error = "input image dimensions must be positive";
        return false;
    }

    if (options.scale > 0)
    {
        const long long scaled_height = static_cast<long long>(input_height) * options.scale;
        const long long scaled_width = static_cast<long long>(input_width) * options.scale;
        if (scaled_height > INT_MAX || scaled_width > INT_MAX)
        {
            error = "--scale produces dimensions that are too large";
            return false;
        }
        const int target_height = static_cast<int>(scaled_height);
        const int target_width = static_cast<int>(scaled_width);
        const int aligned_height = std::max(16, (target_height / 16) * 16);
        const int aligned_width = std::max(16, (target_width / 16) * 16);
        if (ResolutionPlan::from_explicit(aligned_height, aligned_width, plan, &error))
            return true;
    }
    else if (options.width == 0 && options.height == 0)
    {
        const long long input_area = static_cast<long long>(input_height) * input_width;
        const long long min_dimension = std::min(input_height, input_width);
        const long long max_dimension = std::max(input_height, input_width);
        const long long minimum_legal_area = (max_dimension * 16 * 16 + min_dimension - 1) / min_dimension;
        const long long bounded_area = std::max<long long>(minimum_legal_area,
                                                           std::min<long long>(kProductMaxArea, input_area));
        if (bounded_area > kProductMaxArea)
        {
            error = "input aspect ratio cannot fit the 256x256 low-resolution product limit";
            return false;
        }
        if (ResolutionPlan::from_input_area(input_height, input_width, plan, &error,
                                            static_cast<int>(bounded_area)))
            return true;
    }
    else if (ResolutionPlan::from_explicit(options.height, options.width, plan, &error))
    {
        const long long target_area = static_cast<long long>(plan.image_height) * plan.image_width;
        if (target_area <= kProductMaxArea)
            return true;

        error = "CLI target area must not exceed " + std::to_string(kProductMaxArea) +
                " (256x256 low-resolution product limit), got " + std::to_string(plan.image_height) + "x" +
                std::to_string(plan.image_width);
    }
    return false;
}

int select_video_frame_count(int source_frame_count, int start_frame, int requested_frame_count)
{
    if (source_frame_count <= 0)
        return requested_frame_count > 0 ? requested_frame_count : -1;
    if (start_frame >= source_frame_count)
        return 0;
    const int available_frames = source_frame_count - start_frame;
    return requested_frame_count > 0 ? std::min(requested_frame_count, available_frames) : available_frames;
}

} // namespace seedvr2
