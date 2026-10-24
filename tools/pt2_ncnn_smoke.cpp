// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

// Minimal ncnn runner for PT2 conversion artifacts.
//
// It accepts the legacy one-input form and a compact multi-input form for
// validating small exported subgraphs without becoming an application runtime.

#include "net.h"
#include "awa/awa_layers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace {

#if NCNN_VULKAN
struct VulkanInstanceGuard
{
    bool active = false;

    ~VulkanInstanceGuard()
    {
        if (active)
            ncnn::destroy_gpu_instance();
    }
};
#endif

void usage(const char* program)
{
    std::cerr << "usage: " << program
              << " <cpu|vulkan> <param> <bin> <input-blob> <input-f32>"
                 " <width> <height> <output-blob> <output-f32> [gpu-id]\n"
              << "   or: " << program
              << " <cpu|vulkan> <param> <bin> --multi <count>"
                 " <input-blob> <input-f32> <width> <height>..."
                 " <output-blob> <output-f32> [gpu-id]\n";
}

bool parse_int(const char* value, int& result)
{
    try
    {
        std::size_t consumed = 0;
        result = std::stoi(value, &consumed);
        return consumed == std::string(value).size();
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool read_f32(const std::string& path, float* data, std::size_t count)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    return stream.gcount() == static_cast<std::streamsize>(count * sizeof(float)) &&
           stream.peek() == std::char_traits<char>::eof();
}

bool write_f32(const std::string& path, const float* data, std::size_t count)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        return false;
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(stream);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 10)
    {
        usage(argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    struct InputSpec
    {
        std::string blob;
        std::string file;
        int width = 0;
        int height = 0;
    };
    std::vector<InputSpec> inputs;
    std::string output_blob;
    std::string output_file;
    int gpu_id = 0;
    int output_blob_index = 8;
    if (std::string(argv[4]) == "--multi")
    {
        int count = 0;
        if (argc < 8 || !parse_int(argv[5], count) || count <= 0 || count > 64)
        {
            usage(argv[0]);
            return 2;
        }
        const int expected_without_gpu = 8 + count * 4;
        if (argc != expected_without_gpu && argc != expected_without_gpu + 1)
        {
            usage(argv[0]);
            return 2;
        }
        for (int i = 0; i < count; i++)
        {
            const int offset = 6 + i * 4;
            InputSpec input;
            input.blob = argv[offset];
            input.file = argv[offset + 1];
            if (!parse_int(argv[offset + 2], input.width) || !parse_int(argv[offset + 3], input.height) ||
                input.width <= 0 || input.height <= 0)
            {
                usage(argv[0]);
                return 2;
            }
            inputs.push_back(input);
        }
        output_blob_index = 6 + count * 4;
        output_blob = argv[output_blob_index];
        output_file = argv[output_blob_index + 1];
        if (argc == expected_without_gpu + 1 && !parse_int(argv[output_blob_index + 2], gpu_id))
        {
            usage(argv[0]);
            return 2;
        }
    }
    else
    {
        if (argc > 11)
        {
            usage(argv[0]);
            return 2;
        }
        InputSpec input;
        input.blob = argv[4];
        input.file = argv[5];
        if (!parse_int(argv[6], input.width) || !parse_int(argv[7], input.height) ||
            (argc == 11 && !parse_int(argv[10], gpu_id)) || input.width <= 0 || input.height <= 0)
        {
            usage(argv[0]);
            return 2;
        }
        inputs.push_back(input);
        output_blob = argv[8];
        output_file = argv[9];
    }
    if (mode != "cpu" && mode != "vulkan")
    {
        usage(argv[0]);
        return 2;
    }
#if NCNN_VULKAN
    VulkanInstanceGuard vulkan_instance;
#endif
    ncnn::Net net;
    // ncnn enables fp16 options by default, including on capable x86 CPUs.
    // Use the same FP32 contract on both backends for this comparison.
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_int8_inference = false;
    net.opt.use_int8_packed = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_arithmetic = false;
    // Keep correctness probes on scalar FP32 Vulkan paths.  Packed and
    // uniform integer/half settings can otherwise be selected by ncnn even
    // for an FP32 graph and obscure graph-specific conversion errors.
    net.opt.use_packing_layout = true;
    net.opt.use_subgroup_ops = false;
    net.opt.use_shader_local_memory = false;
    net.opt.use_cooperative_matrix = false;
    net.opt.use_fp16_uniform = false;
    net.opt.use_int8_uniform = false;
    net.opt.use_int16_packed = false;
    net.opt.use_int16_storage = false;

    // The converted block graph stores -1 source dims and expects the runtime
    // spec to resolve them, matching the production DiT path contract.
    seedvr2::AwaRuntimeSpec runtime_spec;
    runtime_spec.source_t = 1;
    // Exported DiT graphs carry symbolic source dimensions.  The first input
    // of the input/block/output probes is a matrix whose height is the video
    // token count, so infer the square source grid for both the 8x8 and 16x16
    // fixed-shape validation artifacts.  Graphs without a video matrix (the
    // timestep embedding graph) retain the 8x8 default.
    runtime_spec.source_h = 8;
    runtime_spec.source_w = 8;
    runtime_spec.video_tokens = 64;
    if (!inputs.empty() && inputs.front().height > 0)
    {
        const int token_count = inputs.front().height;
        const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(token_count))));
        if (side > 0 && side * side == token_count)
        {
            runtime_spec.source_h = side;
            runtime_spec.source_w = side;
            runtime_spec.video_tokens = token_count;
        }
    }
    register_seedvr2_awa_layers(net, &runtime_spec);
    bool use_vulkan = mode == "vulkan";
#if NCNN_VULKAN
    if (use_vulkan)
    {
        if (ncnn::create_gpu_instance() != 0)
        {
            std::cerr << "requested Vulkan device is unavailable: " << gpu_id << "\n";
            return 1;
        }
        vulkan_instance.active = true;
        if (gpu_id < 0 || gpu_id >= ncnn::get_gpu_count())
        {
            std::cerr << "requested Vulkan device is unavailable: " << gpu_id << "\n";
            return 1;
        }
        net.set_vulkan_device(gpu_id);
        net.opt.use_vulkan_compute = true;
    }
#else
    if (use_vulkan)
    {
        std::cerr << "this smoke runner was built without NCNN_VULKAN\n";
        return 1;
    }
#endif

    if (net.load_param(argv[2]) != 0 || net.load_model(argv[3]) != 0)
    {
        std::cerr << "failed to load ncnn model\n";
        return 1;
    }

    ncnn::Extractor extractor = net.create_extractor();
    std::size_t input_count = 0;
    for (const InputSpec& spec : inputs)
    {
        const std::size_t count = static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height);
        std::vector<float> values(count);
        if (!read_f32(spec.file, values.data(), count))
        {
            std::cerr << "cannot read exactly " << count << " float32 values from " << spec.file << "\n";
            return 1;
        }
        ncnn::Mat input(spec.width, spec.height, sizeof(float));
        std::copy(values.begin(), values.end(), reinterpret_cast<float*>(input.data));
        if (extractor.input(spec.blob.c_str(), input) != 0)
        {
            std::cerr << "failed to set input blob " << spec.blob << "\n";
            return 1;
        }
        input_count += count;
    }
    ncnn::Mat output;
    if (extractor.extract(output_blob.c_str(), output) != 0)
    {
        std::cerr << "failed to extract output blob " << output_blob << "\n";
        return 1;
    }
    if (output.empty() || output.elempack != 1 || output.elemsize != sizeof(float))
    {
        std::cerr << "output must be a nonempty unpacked float32 tensor (dims=" << output.dims
                  << ", elempack=" << output.elempack
                  << ", elemsize=" << output.elemsize << ")\n";
        return 1;
    }
    const std::size_t output_count = output.total();
    if (!write_f32(output_file, reinterpret_cast<const float*>(output.data), output_count))
    {
        std::cerr << "cannot write output " << output_file << "\n";
        return 1;
    }
    std::cout << "mode=" << mode << " input_count=" << input_count << " output_count=" << output_count
              << " output_shape=" << output.w << "x" << output.h << "x" << output.c
              << " batch=" << output.n << " dims=" << output.dims
              << " elempack=" << output.elempack << " elemsize=" << output.elemsize << "\n";

    return 0;
}
