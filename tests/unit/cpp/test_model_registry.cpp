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

void populate_package(const std::filesystem::path& root)
{
    touch_graph(root / "vae_encode");
    touch_graph(root / "vae_decode");
    touch_graph(root / "dit_input");
    touch_graph(root / "dit_embedding");
    touch_graph(root / "dit_output");
    for (int index = 0; index < 32; index++)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "dit_block_%02d", index);
        touch_graph(root / name);
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
    touch(root / "vae_encode.ncnn.param");
    require(!registry.resolve(plan, graphs, error), "incomplete flat model package rejected");
    require(error.find("vae_encode") != std::string::npos, "missing graph names flat package artifact");

    populate_package(root);
    touch(root / "conditioning" / "pos_emb.f32", 58u * 5120u * sizeof(float));
    require(registry.resolve(plan, graphs, error), error.c_str());
    require(graphs.vae_encode_stem == root / "vae_encode", "encode stem");
    require(graphs.vae_decode_stem == root / "vae_decode", "decode stem");
    require(graphs.dit_stack_dir == root, "DiT stack directory");
    require(graphs.conditioning_path == root / "conditioning" / "pos_emb.f32", "conditioning path");
    require(graphs.text_tokens == 58, "positive conditioning token count");

    seedvr2::ResolutionPlan second_shape;
    require(seedvr2::ResolutionPlan::from_explicit(256, 256, second_shape, &error), error.c_str());
    require(registry.resolve(second_shape, graphs, error), error.c_str());
    require(graphs.vae_encode_stem == root / "vae_encode" && graphs.dit_stack_dir == root,
            "flat package serves multiple target shapes");

    const std::filesystem::path legacy_root = root / "legacy";
    populate_package(legacy_root / "128x128");
    touch(legacy_root / "conditioning" / "pos_emb.f32", 58u * 5120u * sizeof(float));
    seedvr2::ModelRegistry legacy_registry;
    require(seedvr2::ModelRegistry::open(legacy_root, legacy_registry, error), error.c_str());
    require(legacy_registry.resolve(plan, graphs, error), error.c_str());
    require(graphs.vae_encode_stem == legacy_root / "128x128" / "vae_encode", "legacy encode stem");
    require(graphs.vae_decode_stem == legacy_root / "128x128" / "vae_decode", "legacy decode stem");
    require(graphs.dit_stack_dir == legacy_root / "128x128", "legacy DiT stack directory");

    std::filesystem::remove_all(root, cleanup_error);
    std::puts("seedvr2-model-registry: ok");
    return 0;
}
