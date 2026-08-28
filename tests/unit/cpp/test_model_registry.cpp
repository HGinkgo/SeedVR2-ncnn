#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "model/model_registry.h"
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

void touch(const std::filesystem::path& path, std::size_t size = 1)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    require(output.good(), "create fixture file");
    if (size > 0)
    {
        output.seekp(static_cast<std::streamoff>(size - 1));
        output.put('\0');
    }
}

void touch_graph(const std::filesystem::path& stem)
{
    touch(stem.string() + ".ncnn.param");
    touch(stem.string() + ".ncnn.bin");
}

void populate_variant(const std::filesystem::path& variant)
{
    touch_graph(variant / "vae_encode");
    touch_graph(variant / "vae_decode");
    touch_graph(variant / "dit_input");
    touch_graph(variant / "dit_embedding");
    touch_graph(variant / "dit_output");
    for (int index = 0; index < 32; index++)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "dit_block_%02d", index);
        touch_graph(variant / name);
    }
}

} // namespace

int main()
{
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "seedvr2-model-registry-test";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    std::filesystem::create_directories(root, cleanup_error);
    require(!cleanup_error, "create fixture root");

    std::string error;
    seedvr2::ModelRegistry registry;
    require(seedvr2::ModelRegistry::open(root, registry, error), error.c_str());
    seedvr2::ResolutionPlan plan;
    require(seedvr2::ResolutionPlan::from_explicit(128, 128, plan, &error), error.c_str());

    seedvr2::ModelGraphSet graphs;
    require(!registry.resolve(plan, graphs, error), "incomplete model variant rejected");
    require(error.find("128x128") != std::string::npos, "missing variant names target shape");

    populate_variant(root / "128x128");
    touch(root / "conditioning" / "pos_emb.f32", 58u * 5120u * sizeof(float));
    require(registry.resolve(plan, graphs, error), error.c_str());
    require(graphs.vae_encode_stem == root / "128x128" / "vae_encode", "encode stem");
    require(graphs.vae_decode_stem == root / "128x128" / "vae_decode", "decode stem");
    require(graphs.dit_stack_dir == root / "128x128", "DiT stack directory");
    require(graphs.conditioning_path == root / "conditioning" / "pos_emb.f32", "conditioning path");
    require(graphs.text_tokens == 58, "positive conditioning token count");

    seedvr2::ResolutionPlan unsupported;
    require(seedvr2::ResolutionPlan::from_explicit(256, 256, unsupported, &error), error.c_str());
    require(!registry.resolve(unsupported, graphs, error), "unsupported target shape rejected");
    require(error.find("256x256") != std::string::npos, "unsupported shape names target");

    std::filesystem::remove_all(root, cleanup_error);
    std::puts("seedvr2-model-registry: ok");
    return 0;
}
