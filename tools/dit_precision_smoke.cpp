// SPDX-License-Identifier: BSD-3-Clause
#include "dit/dit_precision.h"
#include "command.h"
#include "datareader.h"
#include "net.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

float bf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

ncnn::Mat extract(ncnn::Extractor& extractor, ncnn::Net& net, const char* name, int bits)
{
    ncnn::VkMat gpu;
    ncnn::VkCompute compute(net.vulkan_device());
    require(extractor.extract(name, gpu, compute) == 0, "extract precision boundary");
    require(gpu.elembits() == bits, "unexpected boundary scalar type");
    ncnn::Mat result;
    compute.record_download(gpu, result, net.opt);
    require(compute.submit_and_wait() == 0 && !result.empty(), "download boundary");
    return result;
}

void run(ncnn::VulkanDevice* device, bool packed)
{
    constexpr int width = 128, rows = 3;
    ncnn::VkBlobAllocator blobs(device);
    ncnn::VkStagingAllocator staging(device);
    ncnn::Net net;
    net.set_vulkan_device(device);
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = true;
    net.opt.use_bf16_storage = !packed;
    net.opt.blob_vkallocator = net.opt.workspace_vkallocator = &blobs;
    net.opt.staging_vkallocator = &staging;
    seedvr2::register_dit_block_precision_layers(net);

    // Two successive scale/shift/gate groups exercise the six FP32 parameter
    // sums and the separate BF16 in-place rounding boundaries.
    const char* names[] = {"attn_scale", "attn_shift", "attn_gate", "mlp_scale", "mlp_shift", "mlp_gate"};
    std::ostringstream body;
    body << "Input hidden 0 1 in0\nInput embedding 0 1 in1\n"
         << "RMSNorm norm 1 1 in0 normalized 0=128 1=0.00001 2=0\n"
         << "RMSNorm affine 1 1 in0 affine_out 0=128 1=0.00001 2=1\n";
    std::vector<float> weights;
    for (int x = 0; x < width; ++x) weights.push_back(1.003f + x * 0.00073f);
    for (int group = 0; group < 6; ++group)
    {
        body << "MemoryData constant_p_ada_vid_" << names[group] << " 0 1 p" << group << " 0=128\n"
             << "Reshape view" << group << " 1 1 p" << group << " v" << group << " 0=128 1=1\n"
             << "BinaryOp sum" << group << " 2 1 in1 v" << group << " s" << group << " 0=0\n"
             << "BinaryOp apply" << group << " 2 1 " << (group ? "h" + std::to_string(group - 1) : "normalized")
             << " s" << group << " h" << group << " 0=" << (group % 3 == 1 ? 0 : 2) << "\n";
        for (int x = 0; x < width; ++x)
            weights.push_back((group % 3 == 1 ? -0.103f : 0.997f) + (x % 13) * 0.00113f);
    }
    const std::string graph = "7767517\n28 28\n" + body.str();
    require(net.load_param_mem(graph.c_str()) == 0 && seedvr2::configure_dit_block_precision_boundaries(net),
            "load mixed precision block");
    const unsigned char* cursor = reinterpret_cast<const unsigned char*>(weights.data());
    ncnn::DataReaderFromMemory reader(cursor);
    require(net.load_model(reader) == 0, "load FP32 gamma and Ada parameters");
    ncnn::Mat input(width, rows, sizeof(float)), embedding(width, rows, sizeof(float));
    for (int i = 0; i < width * rows; ++i)
    {
        input[i] = bf16(((i * 17) % 127 - 63) * 0.043f);
        embedding[i] = bf16(((i * 7) % 31 - 15) * 0.0123f);
    }
    auto extractor = net.create_extractor();
    extractor.set_light_mode(false);
    require(extractor.input("in0", input) == 0 && extractor.input("in1", embedding) == 0, "input block");
    ncnn::Mat norm = extract(extractor, net, "normalized", 16);
    ncnn::Mat affine = extract(extractor, net, "affine_out", 32);
    ncnn::Mat final = extract(extractor, net, "h5", 16);
    for (int row = 0; row < rows; ++row)
    {
        double sum = 0;
        for (int x = 0; x < width; ++x) sum += double(input[row * width + x]) * input[row * width + x];
        const float coefficient = 1.f / std::sqrt(float(sum / width) + 0.00001f);
        for (int x = 0; x < width; ++x)
        {
            const int i = row * width + x;
            const float normalized = input[i] * coefficient;
            require(norm[i] == bf16(normalized), "RMS square/reduction must stay FP32");
            require(std::fabs(affine[i] - normalized * weights[x]) < 2.e-6f,
                    "FP32 gamma promotes affine RMS output");
            float expected = bf16(normalized);
            for (int group = 0; group < 6; ++group)
            {
                const float factor = embedding[i] + weights[(group + 1) * width + x];
                expected = bf16(group % 3 == 1 ? expected + factor : expected * factor);
            }
            require(final[i] == expected, "Ada parameters stay FP32 until each in-place cast");
        }
    }
    std::printf("DiT FP32 RMS statistics/gamma/Ada and BF16 boundaries (%s): PASS\n", packed ? "packed" : "native");
}

void run_output(ncnn::VulkanDevice* device, bool packed)
{
    constexpr int width = 128, rows = 3;
    ncnn::VkBlobAllocator blobs(device);
    ncnn::VkStagingAllocator staging(device);
    ncnn::Net net;
    net.set_vulkan_device(device);
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_packed = true;
    net.opt.use_bf16_storage = !packed;
    net.opt.blob_vkallocator = net.opt.workspace_vkallocator = &blobs;
    net.opt.staging_vkallocator = &staging;
    seedvr2::register_dit_output_precision_layers(net);
    const char* graph =
        "7767517\n16 16\nInput hidden 0 1 in0\nInput embedding 0 1 in1\n"
        "UnaryOp square 1 1 in0 square 0=4\n"
        "Reduction mean 1 1 square mean 0=3 1=0 -23303=1,1 4=1 5=1\n"
        "BinaryOp epsilon 1 1 mean epsilon 0=0 1=1 2=0.00001\n"
        "UnaryOp rsqrt 1 1 epsilon coefficient 0=6\n"
        "BinaryOp normalize 2 1 in0 coefficient norm 0=2\n"
        "MemoryData constant_p_norm 0 1 gamma 0=128 1=1\n"
        "BinaryOp affine 2 1 norm gamma affine 0=2\n"
        "MemoryData constant_p_scale 0 1 scale 0=128 1=1\n"
        "BinaryOp sum_scale 2 1 in1 scale factor 0=0\n"
        "BinaryOp multiply 2 1 affine factor multiplied 0=2\n"
        "MemoryData constant_p_shift 0 1 shift 0=128 1=1\n"
        "BinaryOp sum_shift 2 1 in1 shift offset 0=0\n"
        "BinaryOp add 2 1 multiplied offset cast 0=0\n"
        "Gemm identity 1 1 cast out0 2=0 3=1 4=0 5=1 6=0 7=3 8=128 9=128 10=-1\n";
    require(net.load_param_mem(graph) == 0 && seedvr2::configure_dit_output_precision_layers(net),
            "load output precision graph");
    std::vector<float> weights(3 * width + 1 + width * width, 0.f);
    for (int x = 0; x < width; ++x)
    {
        weights[x] = 0.413f + (x % 11) * 0.0037f;
        weights[width + x] = 0.997f + (x % 13) * 0.00113f;
        weights[2 * width + x] = -0.103f + (x % 7) * 0.0053f;
        weights[3 * width + 1 + x * width + x] = 1.f;
    }
    const unsigned char* cursor = reinterpret_cast<const unsigned char*>(weights.data());
    ncnn::DataReaderFromMemory reader(cursor);
    require(net.load_model(reader) == 0, "load output weights");
    ncnn::Mat input(width, rows, sizeof(float)), embedding(width, rows, sizeof(float));
    for (int i = 0; i < width * rows; ++i)
    {
        input[i] = bf16(((i * 17) % 127 - 63) * 0.043f);
        embedding[i] = bf16(((i * 7) % 31 - 15) * 0.0123f);
    }
    auto extractor = net.create_extractor();
    extractor.set_light_mode(false);
    require(extractor.input("in0", input) == 0 && extractor.input("in1", embedding) == 0, "input output head");
    ncnn::Mat square = extract(extractor, net, "square", 32);
    ncnn::Mat affine = extract(extractor, net, "affine", 32);
    ncnn::Mat cast = extract(extractor, net, "cast", 16);
    ncnn::Mat projected = extract(extractor, net, "out0", 16);
    for (int row = 0; row < rows; ++row)
    {
        double sum = 0;
        for (int x = 0; x < width; ++x) sum += double(input[row * width + x]) * input[row * width + x];
        const float coefficient = 1.f / std::sqrt(float(sum / width) + 0.00001f);
        for (int x = 0; x < width; ++x)
        {
            const int i = row * width + x;
            const float normalized = (input[i] * coefficient) * weights[x];
            const float scaled = normalized * (embedding[i] + weights[width + x]);
            const float expected = bf16(scaled + (embedding[i] + weights[2 * width + x]));
            require(square[i] == input[i] * input[i], "output square must stay FP32");
            require(std::fabs(affine[i] - normalized) < 2.e-6f, "output norm and gamma stay FP32");
            require(cast[i] == expected && projected[i] == expected, "output casts only at projection");
        }
    }
    std::printf("DiT output FP32 norm/Ada and BF16 projection (%s): PASS\n", packed ? "packed" : "native");
}
}

int main()
{
    ncnn::create_gpu_instance();
    int status = 0;
    try
    {
        require(ncnn::get_gpu_count() > 0, "Vulkan GPU required");
        run(ncnn::get_gpu_device(0), false);
        run(ncnn::get_gpu_device(0), true);
        run_output(ncnn::get_gpu_device(0), false);
        run_output(ncnn::get_gpu_device(0), true);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "DiT precision: %s\n", error.what());
        status = 1;
    }
    ncnn::destroy_gpu_instance();
    return status;
}
