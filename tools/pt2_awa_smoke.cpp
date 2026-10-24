// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

// Validate a rewritten PT2 AWA graph with the repository's custom layers.

#include "awa/awa_layers.h"
#include "datareader.h"
#include "net.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#include "pipeline.h"
#include "layer/vulkan/shader/mmrope.comp.hex.h"
#endif

namespace
{

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
              << " <cpu|vulkan> <param> <bin> <video-f32> <text-f32>"
                 " <video-tokens> <text-tokens> <heads> <head-dim>"
                 " <video-output-f32> <text-output-f32> [gpu-id]\n"
              << "       " << program << " --check-rope-dispatch [gpu-id]\n"
              << "       " << program << " --check-awa-indices [gpu-id]\n"
              << "       " << program << " --check-attention-cancellation [gpu-id]\n"
              << "       " << program << " --check-attention-bf16 [gpu-id]\n";
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

#if NCNN_VULKAN
bool check_attention_bf16(ncnn::VulkanDevice* device, bool packed)
{
    constexpr int video_tokens = 4, text_tokens = 2, tokens = 6, heads = 2, width = 8;
    ncnn::Mat video(width, 3 * heads, video_tokens, sizeof(float));
    ncnn::Mat text(width, 3 * heads, text_tokens, sizeof(float));
    const auto round_bf16 = [](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        std::memcpy(&value, &bits, sizeof(bits));
        return value;
    };
    for (int token = 0; token < tokens; ++token)
    {
        float* data = token < video_tokens ? video.channel(token) : text.channel(token - video_tokens);
        for (int i = 0; i < 3 * heads * width; ++i)
            data[i] = ((token * 13 + i * 7) % 31 - 15) * 0.125f;
    }
    const auto at = [&](int token, int row, int feature) {
        const float* data = token < video_tokens ? video.channel(token) : text.channel(token - video_tokens);
        return data[row * width + feature];
    };
    std::vector<float> expected(tokens * heads * width);
    for (int query = 0; query < tokens; ++query)
        for (int head = 0; head < heads; ++head)
        {
            double scores[tokens], maximum = -1.e30;
            for (int key = 0; key < tokens; ++key)
            {
                double dot = 0.;
                for (int x = 0; x < width; ++x) dot += double(at(query, head, x)) * at(key, heads + head, x);
                scores[key] = dot / std::sqrt(double(width));
                maximum = std::max(maximum, scores[key]);
            }
            double denominator = 0.;
            for (int key = 0; key < tokens; ++key)
            {
                const double exponential = std::exp(scores[key] - maximum);
                // The mixed-precision path keeps the complete AWA envelope in
                // FP32. Only the Unpack output is rounded back to BF16.
                scores[key] = float(exponential);
                denominator += scores[key];
            }
            for (int x = 0; x < width; ++x)
            {
                double value = 0.;
                for (int key = 0; key < tokens; ++key) value += scores[key] * at(key, 2 * heads + head, x);
                expected[(query * heads + head) * width + x] = round_bf16(float(value / denominator));
            }
        }
    ncnn::Net net;
    net.set_vulkan_device(device);
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = true;
    net.opt.use_bf16_storage = !packed;
    register_seedvr2_awa_layers(net);
    const char* graph =
        "7767517\n5 7\nInput video 0 1 in0\nInput text 0 1 in1\n"
        "SeedVR2AWAPack pack 2 2 in0 in1 packed cu 0=1 1=2 2=2 3=4 4=3 5=3 6=2 7=0\n"
        "SeedVR2WindowAttention attention 1 1 packed attended 0=1 1=2 2=2 3=4 4=3 5=3 6=2 7=0\n"
        "SeedVR2AWAUnpack unpack 1 2 attended out0 out1 0=1 1=2 2=2 3=4 4=3 5=3 6=2 7=0\n";
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory reader(empty_model);
    if (net.load_param_mem(graph) != 0 || net.load_model(reader) != 0) return false;
    auto extractor = net.create_extractor();
    if (extractor.input("in0", video) != 0 || extractor.input("in1", text) != 0) return false;
    double maximum = 0.;
    for (int branch = 0; branch < 2; ++branch)
    {
        ncnn::Mat actual;
        if (extractor.extract(branch ? "out1" : "out0", actual) != 0 || actual.empty() ||
            actual.elempack != 1 || actual.elemsize != 4u) return false;
        for (int channel = 0; channel < actual.c; ++channel)
            for (int depth = 0; depth < actual.d; ++depth)
        {
            const float* values = actual.channel(channel).depth(depth);
            const int token = channel * actual.d + depth;
            const int offset = (token + (branch ? video_tokens : 0)) * heads * width;
            for (int i = 0; i < heads * width; ++i)
            {
                if (!std::isfinite(values[i])) return false;
                maximum = std::max(maximum, std::abs(double(values[i]) - expected[offset + i]));
            }
        }
    }
    std::cout << "attention FP32 envelope / BF16 output precision="
              << (packed ? "packed" : "native") << " max_abs=" << maximum << "\n";
    // The final cast is BF16, so one ULP of the rounded result is expected
    // when Vulkan exp/Kahan ordering differs from the host reference.
    return maximum <= 0.01;
}

// Equal scores make softmax exactly uniform mathematically. Large positive
// and negative V groups around small terms expose loss in sequential FP32
// accumulation. The independent double result needs no model/checkpoint.
bool check_attention_cancellation(ncnn::VulkanDevice* device)
{
    constexpr int video_tokens = 256, text_tokens = 58, heads = 2, width = 128;
    ncnn::Mat video(width, 3 * heads, video_tokens, sizeof(float));
    ncnn::Mat text(width, 3 * heads, text_tokens, sizeof(float));
    video.fill(0.f);
    text.fill(0.f);
    double expected_sum = 0.;
    for (int token = 0; token < video_tokens + text_tokens; ++token)
    {
        const float value = token < 100 ? 10000.f : token < 214 ? 1.f : -10000.f;
        expected_sum += value;
        float* values = token < video_tokens ? video.channel(token) : text.channel(token - video_tokens);
        std::fill(values + 2 * heads * width, values + 3 * heads * width, value);
    }
    const double expected = expected_sum / (video_tokens + text_tokens);
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = net.opt.use_bf16_storage = false;
    net.set_vulkan_device(device);
    register_seedvr2_awa_layers(net);
    const char* graph =
        "7767517\n5 7\nInput video 0 1 in0\nInput text 0 1 in1\n"
        "SeedVR2AWAPack pack 2 2 in0 in1 packed cu 0=1 1=16 2=16 3=4 4=3 5=3 6=58 7=0\n"
        "SeedVR2WindowAttention attention 1 1 packed attended 0=1 1=16 2=16 3=4 4=3 5=3 6=58 7=0\n"
        "SeedVR2AWAUnpack unpack 1 2 attended out0 out1 0=1 1=16 2=16 3=4 4=3 5=3 6=58 7=0\n";
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory reader(empty_model);
    if (net.load_param_mem(graph) != 0 || net.load_model(reader) != 0)
        return false;
    auto extractor = net.create_extractor();
    if (extractor.input("in0", video) != 0 || extractor.input("in1", text) != 0)
        return false;
    double maximum = 0.;
    for (const char* name : {"out0", "out1"})
    {
        ncnn::Mat actual;
        if (extractor.extract(name, actual) != 0 || actual.empty() ||
            actual.elempack != 1 || actual.elemsize != 4u)
            return false;
        for (int channel = 0; channel < actual.c; ++channel)
        {
            const float* values = actual.channel(channel);
            for (int index = 0; index < actual.w * actual.h * actual.d; ++index)
            {
                if (!std::isfinite(values[index]))
                    return false;
                maximum = std::max(maximum, std::abs(values[index] - expected));
            }
        }
    }
    std::cout << "attention-cancellation precision=fp32 tokens=314 max_abs=" << maximum
              << " limit=1e-5\n";
    return maximum <= 1.e-5;
}

// A three-token dispatch rounds up to four workgroups in z. The fourth
// channel is a sentinel, so a missing token bound becomes a deterministic
// failure without relying on an out-of-allocation write or model weights.
bool check_rope_dispatch(ncnn::VulkanDevice* device, bool bf16, bool packed = false, bool fp32_input = false)
{
    ncnn::VkBlobAllocator blobs(device);
    ncnn::VkStagingAllocator staging(device);
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = bf16;
    opt.use_bf16_storage = bf16 && !packed;
    opt.blob_vkallocator = opt.workspace_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(mmrope_comp_data, sizeof(mmrope_comp_data), opt, spirv) != 0)
        return false;
    ncnn::Pipeline pipeline(device);
    pipeline.set_optimal_local_size_xyz(4, 4, 4);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), {}) != 0)
        return false;

    constexpr int tokens = 3, allocated_tokens = 4, width = 8, heads = 1, rope = 6;
    ncnn::Mat input(width, 3 * heads, allocated_tokens, sizeof(float));
    ncnn::Mat initial(width, 3 * heads, allocated_tokens, sizeof(float));
    ncnn::Mat rotations(2 * rope, allocated_tokens, sizeof(float));
    input.fill(2.f);
    if (fp32_input)
        for (int token = 0; token < allocated_tokens; ++token)
            for (int i = 0; i < width * 3 * heads; ++i)
                static_cast<float*>(input.channel(token).data)[i] = 2.003f + i * 0.011f;
    initial.fill(99.f);
    for (int token = 0; token < allocated_tokens; ++token)
        for (int feature = 0; feature < 2 * rope; ++feature)
            rotations.row(token)[feature] = fp32_input ? (feature < rope ? 0.8f : 0.6f) :
                                                        (feature < rope ? 1.f : 0.f);
    ncnn::VkMat input_gpu, output_gpu, rotations_gpu;
    ncnn::VkCompute compute(device);
    ncnn::Option fp32 = opt;
    fp32.use_bf16_packed = fp32.use_bf16_storage = false;
    compute.record_upload(input, input_gpu, fp32_input ? fp32 : opt);
    compute.record_upload(initial, output_gpu, opt);
    compute.record_upload(rotations, rotations_gpu, fp32);
    // record_upload chooses pack4 for four channels even when the Net packing
    // option is off. Custom AWA kernels consume pack1, as enforced by Net.
    for (ncnn::VkMat* tensor : {&input_gpu, &output_gpu, &rotations_gpu})
    {
        ncnn::VkMat unpacked;
        device->convert_packing(*tensor, unpacked, 1, compute,
                                tensor == &rotations_gpu || (tensor == &input_gpu && fp32_input) ? fp32 : opt);
        *tensor = unpacked;
    }
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].i = static_cast<int>(input_gpu.cstep);
    constants[1].i = static_cast<int>(output_gpu.cstep);
    constants[2].i = heads;
    constants[3].i = width;
    constants[4].i = rope;
    constants[5].i = tokens;
    constants[6].i = input_gpu.elemsize == 4u;
    ncnn::VkMat dispatcher;
    dispatcher.w = width;
    dispatcher.h = 3 * heads;
    dispatcher.c = tokens;
    compute.record_pipeline(&pipeline, {input_gpu, output_gpu, rotations_gpu, input_gpu}, constants, dispatcher);
    ncnn::Mat result;
    compute.record_download(output_gpu, result, opt);
    if (compute.submit_and_wait() != 0 || result.empty() || result.elemsize != 4u)
    {
        std::cerr << "rope-dispatch download failed\n";
        return false;
    }
    for (int token = 0; token < allocated_tokens; ++token)
    {
        const float* values = result.channel(token);
        for (int i = 0; i < width * 3 * heads; ++i)
        {
            float expected = token < tokens ? 2.f : 99.f;
            if (fp32_input && token < tokens)
            {
                const float* source = input.channel(token);
                const int feature = i % width;
                expected = source[i];
                if (i / width < 2 * heads && feature < rope)
                {
                    const float paired = source[i + (feature % 2 == 0 ? 1 : -1)];
                    expected = feature % 2 == 0 ? expected * 0.8f - paired * 0.6f :
                                                 expected * 0.8f + paired * 0.6f;
                }
                if (bf16)
                {
                    uint32_t bits;
                    std::memcpy(&bits, &expected, sizeof(bits));
                    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
                    std::memcpy(&expected, &bits, sizeof(bits));
                }
            }
            if (values[i] != expected)
            {
                std::cerr << "rope-dispatch mismatch precision=" << (bf16 ? "bf16" : "fp32")
                          << " token=" << token << " feature=" << i << " actual=" << values[i] << "\n";
                return false;
            }
        }
    }
    std::cout << "rope-dispatch precision=" << (bf16 ? (packed ? "bf16-packed" : "bf16-native") : "fp32")
              << " fp32_input=" << fp32_input
              << " logical_tokens=3 sentinel_tokens=1: pass\n";
    return true;
}

bool check_awa_indices(ncnn::VulkanDevice* device, bool bf16, bool shifted, bool packed = false)
{
    // Token-specific, exactly representable BF16 values expose corrupt index
    // uploads. Several clipped windows also exercise repeated text indices.
    ncnn::Mat video(8, 3, 64, sizeof(float));
    ncnn::Mat text(8, 3, 5, sizeof(float));
    for (int token = 0; token < 64; ++token)
        for (int i = 0; i < 24; ++i)
            static_cast<float*>(video.channel(token).data)[i] = (token * 7 + i) % 127 - 63.f;
    for (int token = 0; token < 5; ++token)
        for (int i = 0; i < 24; ++i)
            static_cast<float*>(text.channel(token).data)[i] = token * 13 + i - 63.f;
    std::ostringstream graph;
    graph << "7767517\n4 6\nInput video 0 1 in0\nInput text 0 1 in1\n"
          << "SeedVR2AWAPack pack 2 2 in0 in1 packed cu 0=1 1=8 2=8 3=1 4=3 5=3 6=5 7="
          << shifted << "\nSeedVR2AWAUnpack unpack 1 2 packed out0 out1 0=1 1=8 2=8 3=1 4=3 5=3 6=5 7="
          << shifted << "\n";
    ncnn::Net reference, candidate;
    for (ncnn::Net* net : {&reference, &candidate})
    {
        net->opt.use_packing_layout = false;
        net->opt.use_fp16_packed = net->opt.use_fp16_storage = net->opt.use_fp16_arithmetic = false;
        net->opt.use_bf16_packed = net == &candidate && bf16;
        net->opt.use_bf16_storage = net == &candidate && bf16 && !packed;
        if (net == &candidate)
        {
            net->opt.use_vulkan_compute = true;
            net->set_vulkan_device(device);
        }
        register_seedvr2_awa_layers(*net);
        const unsigned char* empty_model = nullptr;
        ncnn::DataReaderFromMemory reader(empty_model);
        if (net->load_param_mem(graph.str().c_str()) != 0 || net->load_model(reader) != 0)
            return false;
    }
    auto cpu = reference.create_extractor();
    auto gpu = candidate.create_extractor();
    for (ncnn::Extractor* extractor : {&cpu, &gpu})
        if (extractor->input("in0", video) != 0 || extractor->input("in1", text) != 0)
            return false;
    double max_error = 0.;
    for (const char* output : {"packed", "out0", "out1"})
    {
        ncnn::Mat expected, actual;
        if (cpu.extract(output, expected) != 0 || gpu.extract(output, actual) != 0 ||
            expected.empty() || actual.empty() || actual.elemsize != 4u ||
            expected.w != actual.w || expected.h != actual.h || expected.d != actual.d ||
            expected.c != actual.c || expected.n != actual.n)
            return false;
        for (int batch = 0; batch < actual.n; ++batch)
            for (int channel = 0; channel < actual.c; ++channel)
            {
                const float* a = static_cast<const float*>(actual.data) + batch * actual.nstep + channel * actual.cstep;
                const float* b = static_cast<const float*>(expected.data) + batch * expected.nstep + channel * expected.cstep;
                for (int i = 0; i < actual.w * actual.h * actual.d; ++i)
                {
                    if (!std::isfinite(a[i]))
                        return false;
                    max_error = std::max(max_error, std::abs(static_cast<double>(a[i]) - b[i]));
                }
            }
    }
    std::cout << "awa-indices precision=" << (bf16 ? (packed ? "bf16-packed" : "bf16-native") : "fp32")
              << " shifted=" << shifted << " max_abs=" << max_error << "\n";
    return max_error == 0.;
}
#endif

bool read_f32(const std::string& path, float* data, std::size_t count)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    return stream.gcount() == static_cast<std::streamsize>(count * sizeof(float)) &&
           stream.peek() == std::char_traits<char>::eof();
}

bool write_f32(const std::string& path, const ncnn::Mat& output)
{
    if (output.empty() || output.elempack != 1 || output.elemsize != sizeof(float))
        return false;
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        return false;
    const float* base = static_cast<const float*>(output.data);
    const size_t row_size = static_cast<size_t>(output.w);
    const size_t rows_per_channel = static_cast<size_t>(output.h);
    const size_t depths_per_channel = output.dims >= 4 ? static_cast<size_t>(output.d) : 1u;
    const size_t channels = output.dims >= 3 ? static_cast<size_t>(output.c) : 1u;
    const size_t channel_stride = output.dims >= 3 ? output.cstep : output.w * output.h;
    const size_t depth_stride = row_size * rows_per_channel;
    const size_t batch_count =
#if NCNN_BATCH
        static_cast<size_t>(output.n);
#else
        1u;
#endif
    const size_t batch_stride =
#if NCNN_BATCH
        output.nstep;
#else
        output.total();
#endif
    for (size_t batch = 0; batch < batch_count; batch++)
    {
        const float* batch_data = base + batch * batch_stride;
        for (size_t channel = 0; channel < channels; channel++)
        {
            const float* channel_data = batch_data + channel * channel_stride;
            for (size_t depth = 0; depth < depths_per_channel; depth++)
            {
                const float* row_data = channel_data + depth * depth_stride;
                for (size_t row = 0; row < rows_per_channel; row++)
                {
                    stream.write(reinterpret_cast<const char*>(row_data + row * row_size),
                                 static_cast<std::streamsize>(row_size * sizeof(float)));
                    if (!stream)
                        return false;
                }
            }
        }
    }
    return true;
}

size_t logical_count(const ncnn::Mat& output)
{
    size_t count = static_cast<size_t>(output.w);
    if (output.dims >= 2)
        count *= static_cast<size_t>(output.h);
    if (output.dims >= 3)
        count *= static_cast<size_t>(output.c);
    if (output.dims >= 4)
        count *= static_cast<size_t>(output.d);
#if NCNN_BATCH
    count *= static_cast<size_t>(output.n);
#endif
    return count;
}

} // namespace

int main(int argc, char** argv)
{
#if NCNN_VULKAN
    if (argc >= 2 && (std::string(argv[1]) == "--check-rope-dispatch" ||
                      std::string(argv[1]) == "--check-awa-indices" ||
                      std::string(argv[1]) == "--check-attention-cancellation" ||
                      std::string(argv[1]) == "--check-attention-bf16"))
    {
        int gpu_id = 0;
        if (argc > 3 || (argc == 3 && !parse_int(argv[2], gpu_id)))
            return 2;
        VulkanInstanceGuard instance;
        if (ncnn::create_gpu_instance() != 0)
            return 1;
        instance.active = true;
        if (gpu_id < 0 || gpu_id >= ncnn::get_gpu_count())
            return 1;
        ncnn::VulkanDevice* device = ncnn::get_gpu_device(gpu_id);
        if (std::string(argv[1]) == "--check-attention-cancellation")
            return check_attention_cancellation(device) ? 0 : 1;
        if (std::string(argv[1]) == "--check-attention-bf16")
            return check_attention_bf16(device, false) && check_attention_bf16(device, true) ? 0 : 1;
        if (std::string(argv[1]) == "--check-awa-indices")
            return check_awa_indices(device, false, false) && check_awa_indices(device, false, true) &&
                   check_awa_indices(device, true, false) && check_awa_indices(device, true, true) &&
                   check_awa_indices(device, true, false, true) && check_awa_indices(device, true, true, true) ? 0 : 1;
        return check_rope_dispatch(device, false) && check_rope_dispatch(device, true) &&
               check_rope_dispatch(device, true, true) && check_rope_dispatch(device, true, false, true) &&
               check_rope_dispatch(device, true, true, true) ? 0 : 1;
    }
#endif
    if (argc < 12 || argc > 13)
    {
        usage(argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    int video_tokens = 0;
    int text_tokens = 0;
    int heads = 0;
    int head_dim = 0;
    int gpu_id = 0;
    if (!parse_int(argv[6], video_tokens) || !parse_int(argv[7], text_tokens) ||
        !parse_int(argv[8], heads) || !parse_int(argv[9], head_dim) ||
        (argc == 13 && !parse_int(argv[12], gpu_id)) ||
        (mode != "cpu" && mode != "vulkan") || video_tokens <= 0 || text_tokens <= 0 ||
        heads <= 0 || head_dim <= 0)
    {
        usage(argv[0]);
        return 2;
    }

    const std::size_t video_count = static_cast<std::size_t>(video_tokens) * 3 * heads * head_dim;
    const std::size_t text_count = static_cast<std::size_t>(text_tokens) * 3 * heads * head_dim;
    std::vector<float> video_values(video_count);
    std::vector<float> text_values(text_count);
    if (!read_f32(argv[4], video_values.data(), video_count) ||
        !read_f32(argv[5], text_values.data(), text_count))
    {
        std::cerr << "cannot read input float32 files with the declared sizes\n";
        return 1;
    }

    // [tokens, qkv, heads, head_dim] has the same contiguous order as this
    // ncnn [w=head_dim, h=3*heads, c=tokens] representation.
    ncnn::Mat video(head_dim, 3 * heads, video_tokens, sizeof(float));
    ncnn::Mat text(head_dim, 3 * heads, text_tokens, sizeof(float));
    const std::size_t video_row = static_cast<std::size_t>(3 * heads * head_dim);
    const std::size_t text_row = static_cast<std::size_t>(3 * heads * head_dim);
    for (int token = 0; token < video_tokens; token++)
        std::copy(video_values.begin() + static_cast<std::size_t>(token) * video_row,
                  video_values.begin() + static_cast<std::size_t>(token + 1) * video_row,
                  static_cast<float*>(video.channel(token).data));
    for (int token = 0; token < text_tokens; token++)
        std::copy(text_values.begin() + static_cast<std::size_t>(token) * text_row,
                  text_values.begin() + static_cast<std::size_t>(token + 1) * text_row,
                  static_cast<float*>(text.channel(token).data));

#if NCNN_VULKAN
    VulkanInstanceGuard vulkan_instance;
#endif
    ncnn::Net net;
    net.opt.use_packing_layout = false;
    net.opt.use_bf16_packed = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    const bool use_vulkan = mode == "vulkan";
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
        // Keep the small reference comparison in fp32.  This isolates graph
        // and custom-layer correctness from optional bf16/fp16 quantization.
        net.opt.num_threads = 8;
    }
#else
    if (use_vulkan)
    {
        std::cerr << "this smoke runner was built without NCNN_VULKAN\n";
        return 1;
    }
#endif

    register_seedvr2_awa_layers(net);
    if (net.load_param(argv[2]) != 0 || net.load_model(argv[3]) != 0)
    {
        std::cerr << "failed to load AWA ncnn model\n";
        return 1;
    }

    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("in0", video) != 0 || extractor.input("in1", text) != 0)
    {
        std::cerr << "failed to set AWA inputs\n";
        return 1;
    }
    ncnn::Mat video_output;
    ncnn::Mat text_output;
    if (extractor.extract("out0", video_output) != 0 || extractor.extract("out1", text_output) != 0)
    {
        std::cerr << "failed to extract AWA outputs\n";
        return 1;
    }
    if (!write_f32(argv[10], video_output) || !write_f32(argv[11], text_output))
    {
        std::cerr << "cannot write AWA outputs\n";
        return 1;
    }
    size_t video_batches = 1;
    size_t text_batches = 1;
#if NCNN_BATCH
    video_batches = static_cast<size_t>(video_output.n);
    text_batches = static_cast<size_t>(text_output.n);
#endif
    std::cout << "mode=" << mode << " video_count=" << logical_count(video_output)
              << " text_count=" << logical_count(text_output) << " video_dims="
              << video_output.dims << " video_shape=" << video_output.w << "x" << video_output.h
              << "x" << video_output.d << "x" << video_output.c << "x" << video_batches
              << " video_cstep=" << video_output.cstep
              << " text_dims=" << text_output.dims << " text_shape=" << text_output.w << "x"
              << text_output.h << "x" << text_output.d << "x" << text_output.c << "x"
              << text_batches << " text_cstep=" << text_output.cstep << "\n";
    return 0;
}
