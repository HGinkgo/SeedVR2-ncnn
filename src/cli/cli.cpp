#include "cli.h"

#include "resolution/resolution_plan.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace seedvr2
{
namespace
{

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
            options.input = value;
        }
        else if (std::strcmp(argument, "--output") == 0)
        {
            if (!next_value(argc, argv, index, value, error))
                return false;
            options.output = value;
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

    if (options.action != CliAction::Run)
        return true;
    if (options.input.empty())
    {
        error = "--input is required";
        return false;
    }
    if (width_set != height_set)
    {
        error = "--width and --height must be provided together";
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

    if (options.width == 0 && options.height == 0)
    {
        if (ResolutionPlan::from_input_area(input_height, input_width, plan, &error))
            return true;
    }
    else if (ResolutionPlan::from_explicit(options.height, options.width, plan, &error))
    {
        return true;
    }
    return false;
}

} // namespace seedvr2
