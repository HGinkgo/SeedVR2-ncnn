// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

#include "cli/cli.h"
#include "model/model_registry.h"
#include "resolution/resolution_plan.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace
{
void require(bool ok, const std::string& message)
{
    if (!ok)
        throw std::runtime_error(message);
}

struct TemporaryPackage
{
    fs::path root = fs::temp_directory_path() /
        ("seedvr2-conditioning-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryPackage() { std::error_code error; fs::remove_all(root, error); }
};

void graph(const fs::path& stem)
{
    fs::create_directories(stem.parent_path());
    std::ofstream(stem.string() + ".ncnn.param") << "7767517\n0 0\n";
    std::ofstream(stem.string() + ".ncnn.bin");
}

void stack(const fs::path& root)
{
    for (const char* name : {"dit_input", "dit_embedding", "dit_output"})
        graph(root / name);
    for (int index = 0; index < 32; ++index)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "dit_block_%02d", index);
        graph(root / name);
    }
}

void conditioning(const fs::path& path, int tokens)
{
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(tokens) * 5120 * sizeof(float) - 1);
    output.put('\0');
}
} // namespace

int main()
{
    try
    {
        std::string error;
        seedvr2::CliOptions options;
        const char* valid[] = {"seedvr2-ncnn", "--cfg-scale", "1.5", "--help"};
        require(seedvr2::parse_cli(4, valid, options, error) && options.cfg_scale == 1.5f,
                "finite CFG scale is accepted: " + error);
        for (const char* invalid : {"nan", "inf", "-1", "0.5", "2x", "1e999"})
        {
            const char* args[] = {"seedvr2-ncnn", "--cfg-scale", invalid, "--help"};
            require(!seedvr2::parse_cli(4, args, options, error) && error.find("--cfg-scale") != std::string::npos,
                    "invalid CFG scale must be diagnosed");
        }
        for (const char* seed : {"0", "666", "4294967295"})
        {
            const char* args[] = {"seedvr2-ncnn", "--seed", seed, "--help"};
            require(seedvr2::parse_cli(4, args, options, error) && options.noise_seed.has_value(),
                    "uint32 noise seed must be accepted: " + error);
        }
        require(*options.noise_seed == 4294967295U, "seed must not be narrowed");
        for (const char* seed : {"", "-1", "+1", " 1", "1x", "1.0", "nan", "4294967296"})
        {
            const char* args[] = {"seedvr2-ncnn", "--seed", seed, "--help"};
            require(!seedvr2::parse_cli(4, args, options, error) && error.find("--seed") != std::string::npos,
                    "invalid noise seed must be diagnosed");
        }
        const char* legacy[] = {"seedvr2-ncnn", "--help"};
        require(seedvr2::parse_cli(2, legacy, options, error) && !options.noise_seed,
                "omitting seed retains the legacy noise fixture");
        require(!options.fp32, "default precision must use the validated mixed path");
        for (const char* precision : {"bf16", "fp32"})
        {
            const char* args[] = {"seedvr2-ncnn", "--precision", precision, "--input", "image.PNG"};
            require(seedvr2::parse_cli(5, args, options, error) && options.fp32 == (std::string(precision) == "fp32"),
                    "explicit image precision must be accepted: " + error);
        }
        const char* invalid_precision[] = {"seedvr2-ncnn", "--precision", "fp16", "--help"};
        require(!seedvr2::parse_cli(4, invalid_precision, options, error) && error.find("--precision") != std::string::npos,
                "unsupported precision must be diagnosed");
        const auto rejects_fp32 = [&](std::vector<const char*> extra) {
            std::vector<const char*> args = {"seedvr2-ncnn", "--precision", "fp32", "--input", "image.png"};
            args.insert(args.end(), extra.begin(), extra.end());
            require(!seedvr2::parse_cli(static_cast<int>(args.size()), args.data(), options, error) &&
                        error.find("--precision fp32") != std::string::npos,
                    "FP32 must reject unsupported runtime contracts before inference: " + error);
        };
        rejects_fp32({"--input", "second.png"});
        rejects_fp32({"--output", "video.avi"});
        rejects_fp32({"--vae-tile-size", "128"});
        rejects_fp32({"--steps", "5"});
        rejects_fp32({"--cfg-scale", "21"});
        const char* video[] = {"seedvr2-ncnn", "--precision", "fp32", "--input", "video.avi"};
        require(!seedvr2::parse_cli(5, video, options, error) && error.find("--precision fp32") != std::string::npos,
                "FP32 video must be rejected");

        TemporaryPackage package;
        stack(package.root);
        graph(package.root / "vae_encode");
        graph(package.root / "vae_decode");
        const fs::path positive = package.root / "conditioning/pos_emb.f32";
        const fs::path negative = package.root / "conditioning/neg_emb.f32";
        conditioning(positive, 58);
        seedvr2::ModelRegistry registry;
        seedvr2::ResolutionPlan plan;
        seedvr2::ModelGraphSet graphs;
        require(seedvr2::ResolutionPlan::from_explicit(256, 256, plan, &error), error);
        require(seedvr2::ModelRegistry::open(package.root, registry, error), error);
        require(registry.resolve(plan, graphs, error) && graphs.text_tokens == 58 && !graphs.has_negative_conditioning(),
                "positive-only package must retain its contract: " + error);
        conditioning(positive, 64);
        require(!registry.resolve(plan, graphs, error) && error.find("58x5120") != std::string::npos,
                "64-token positive embedding must be rejected");
        conditioning(positive, 58);
        conditioning(negative, 64);
        require(!registry.resolve(plan, graphs, error) && error.find("incomplete") != std::string::npos,
                "negative embedding without its graph stack must be rejected");
        stack(package.root / "cfg_64");
        require(registry.resolve(plan, graphs, error) && graphs.has_negative_conditioning() &&
                    graphs.text_tokens == 58 && graphs.negative_text_tokens == 64,
                "separate 58/64-token graph stacks must resolve: " + error);
        conditioning(negative, 58);
        require(!registry.resolve(plan, graphs, error) && error.find("64x5120") != std::string::npos,
                "negative embedding may not be padded or truncated");
        conditioning(negative, 64);
        fs::remove(package.root / "cfg_64/dit_block_31.ncnn.bin");
        require(!registry.resolve(plan, graphs, error) && error.find("dit_block_31") != std::string::npos,
                "incomplete negative stack must be rejected");
        graph(package.root / "cfg_64/dit_block_31");
        fs::create_directory(package.root / "256x256");
        std::vector<fs::path> artifacts;
        for (const auto& entry : fs::directory_iterator(package.root))
            if (entry.path().filename() != "conditioning" && entry.path().filename() != "256x256")
                artifacts.push_back(entry.path());
        for (const auto& artifact : artifacts)
            fs::rename(artifact, package.root / "256x256" / artifact.filename());
        require(registry.resolve(plan, graphs, error) && graphs.has_negative_conditioning(),
                "fixed-size CFG variant must resolve at its exported size: " + error);
        require(seedvr2::ResolutionPlan::from_explicit(128, 128, plan, &error), error);
        require(!registry.resolve(plan, graphs, error) && error.find("variant is missing") != std::string::npos,
                "fixed-size package must reject an unexported resolution");
        std::puts("conditioning-smoke: ok");
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "conditioning-smoke: %s\n", error.what());
        return 1;
    }
    return 0;
}
