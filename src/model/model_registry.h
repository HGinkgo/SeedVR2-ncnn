#pragma once

#include <filesystem>
#include <string>

namespace seedvr2
{

struct ResolutionPlan;

struct ModelGraphSet final
{
    std::filesystem::path vae_encode_stem;
    std::filesystem::path dit_stack_dir;
    std::filesystem::path vae_decode_stem;
    std::filesystem::path conditioning_path;
    int text_tokens = 0;
};

class ModelRegistry final
{
public:
    static bool open(const std::filesystem::path& model_dir, ModelRegistry& registry, std::string& error);

    bool resolve(const ResolutionPlan& plan, ModelGraphSet& graphs, std::string& error) const;

private:
    std::filesystem::path model_dir_;
};

} // namespace seedvr2
