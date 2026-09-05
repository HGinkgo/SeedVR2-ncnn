#include "model/model_registry.h"

#include "resolution/resolution_plan.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace seedvr2
{
namespace
{

constexpr std::size_t kConditioningWidth = 5120;
constexpr std::size_t kPositiveTokens = 58;
constexpr std::size_t kNegativeTokens = 64;

bool is_regular_model_file(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

bool graph_exists(const std::filesystem::path& stem, std::string& error)
{
    const std::filesystem::path param = stem.string() + ".ncnn.param";
    const std::filesystem::path model = stem.string() + ".ncnn.bin";
    if (!is_regular_model_file(param))
    {
        error = "missing model graph artifact: " + param.string();
        return false;
    }
    if (!is_regular_model_file(model))
    {
        error = "missing model graph artifact: " + model.string();
        return false;
    }
    return true;
}

std::string variant_name(const ResolutionPlan& plan)
{
    return std::to_string(plan.image_height) + "x" + std::to_string(plan.image_width);
}

bool has_flat_graph_artifact(const std::filesystem::path& model_dir)
{
    const std::filesystem::path encode = model_dir / "vae_encode";
    return is_regular_model_file(encode.string() + ".ncnn.param") ||
           is_regular_model_file(encode.string() + ".ncnn.bin");
}

bool validate_manifest(const std::filesystem::path& model_dir, std::string& error)
{
    const std::filesystem::path manifest = model_dir / "manifest.sha256";
    std::ifstream input(manifest);
    if (!input)
    {
        error = "missing model package manifest: " + manifest.string();
        return false;
    }

    std::size_t records = 0;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const std::size_t separator = line.find("  ");
        if (separator != 64 || line.size() <= separator + 2)
        {
            error = "invalid model package manifest record at line " + std::to_string(records + 1);
            return false;
        }
        ++records;
    }
    if (records != 75)
    {
        error = "model package manifest must contain 75 records, got " + std::to_string(records);
        return false;
    }

    std::error_code traversal_error;
    for (std::filesystem::recursive_directory_iterator it(model_dir, traversal_error), end; it != end; it.increment(traversal_error))
    {
        if (traversal_error)
        {
            error = "failed to inspect model package: " + model_dir.string();
            return false;
        }
        const bool is_symlink = it->is_symlink(traversal_error);
        if (traversal_error)
        {
            error = "failed to inspect model package: " + model_dir.string();
            return false;
        }
        if (is_symlink)
        {
            error = "model package must not contain symbolic links: " + it->path().string();
            return false;
        }
    }
    return true;
}

} // namespace

bool ModelRegistry::open(const std::filesystem::path& model_dir, ModelRegistry& registry, std::string& error)
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

    registry.model_dir_ = model_dir;
    return true;
}

bool ModelRegistry::check_package(std::string& error) const
{
    error.clear();
    if (model_dir_.empty())
    {
        error = "model registry is not open";
        return false;
    }
    if (!validate_manifest(model_dir_, error))
        return false;

    ResolutionPlan plan;
    if (!ResolutionPlan::from_explicit(256, 256, plan, &error))
        return false;
    ModelGraphSet graphs;
    if (!resolve(plan, graphs, error))
        return false;
    return true;
}

bool ModelRegistry::resolve(const ResolutionPlan& plan, ModelGraphSet& graphs, std::string& error) const
{
    graphs = ModelGraphSet();
    error.clear();
    if (model_dir_.empty())
    {
        error = "model registry is not open";
        return false;
    }
    if (plan.image_height <= 0 || plan.image_width <= 0)
    {
        error = "model graph resolution requires positive target dimensions";
        return false;
    }

    std::filesystem::path graph_dir = model_dir_;
    if (!has_flat_graph_artifact(model_dir_))
    {
        graph_dir /= variant_name(plan);
        std::error_code filesystem_error;
        if (!std::filesystem::is_directory(graph_dir, filesystem_error))
        {
            error = "model variant is missing for target " + variant_name(plan) + ": " + graph_dir.string();
            return false;
        }
    }

    const std::filesystem::path vae_encode = graph_dir / "vae_encode";
    const std::filesystem::path vae_decode = graph_dir / "vae_decode";
    const std::filesystem::path dit_stack = graph_dir;
    if (!graph_exists(vae_encode, error) || !graph_exists(vae_decode, error))
        return false;

    const std::vector<std::filesystem::path> stack_graphs = {
        dit_stack / "dit_input",
        dit_stack / "dit_embedding",
        dit_stack / "dit_output",
    };
    for (const std::filesystem::path& graph : stack_graphs)
        if (!graph_exists(graph, error))
            return false;
    for (int index = 0; index < 32; index++)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "dit_block_%02d", index);
        if (!graph_exists(dit_stack / name, error))
            return false;
    }

    const std::filesystem::path conditioning = model_dir_ / "conditioning" / "pos_emb.f32";
    if (!is_regular_model_file(conditioning))
    {
        error = "missing positive conditioning artifact: " + conditioning.string();
        return false;
    }
    std::error_code size_error;
    const std::uintmax_t bytes = std::filesystem::file_size(conditioning, size_error);
    const std::uintmax_t bytes_per_token = kConditioningWidth * sizeof(float);
    if (size_error || (bytes != kPositiveTokens * bytes_per_token && bytes != kNegativeTokens * bytes_per_token))
    {
        std::ostringstream message;
        message << "positive conditioning artifact has unsupported size: " << conditioning.string();
        error = message.str();
        return false;
    }

    graphs.vae_encode_stem = vae_encode;
    graphs.dit_stack_dir = dit_stack;
    graphs.vae_decode_stem = vae_decode;
    graphs.conditioning_path = conditioning;
    graphs.text_tokens = static_cast<int>(bytes / bytes_per_token);
    return true;
}

} // namespace seedvr2
