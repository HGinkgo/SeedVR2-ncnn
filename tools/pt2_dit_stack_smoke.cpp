// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

// Execute the exported 32-block PT2 stack through the production
// DitStackSession Vulkan path and compare its final matrix with the portable
// eager golden record emitted by tools/export_dit_stack.py.

#include "dit/dit_stack.h"
#include "dit/dit_stack_internal.h"
#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "net.h"
#include "resolution/resolution_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace
{

struct TensorRecord final
{
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

template <typename T>
bool read_scalar(std::ifstream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(stream);
}

bool read_golden(const std::string& path, std::map<std::string, TensorRecord>& records)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        std::cerr << "cannot open golden file: " << path << "\n";
        return false;
    }

    char magic[8] = {};
    std::uint32_t version = 0;
    std::uint32_t record_count = 0;
    if (!stream.read(magic, sizeof(magic)) || !read_scalar(stream, version) || !read_scalar(stream, record_count) ||
        std::memcmp(magic, "SVR2F32\0", sizeof(magic)) != 0 || version != 1)
    {
        std::cerr << "invalid portable golden header: " << path << "\n";
        return false;
    }

    for (std::uint32_t index = 0; index < record_count; ++index)
    {
        std::uint16_t name_length = 0;
        std::uint8_t rank = 0;
        std::uint8_t reserved = 0;
        std::uint64_t element_count = 0;
        std::uint64_t dimensions[4] = {};
        if (!read_scalar(stream, name_length) || !read_scalar(stream, rank) || !read_scalar(stream, reserved) ||
            !read_scalar(stream, element_count) || !stream.read(reinterpret_cast<char*>(dimensions),
                                                                  sizeof(dimensions)) ||
            rank > 4)
        {
            std::cerr << "invalid portable golden record header at index " << index << "\n";
            return false;
        }

        std::string name(name_length, '\0');
        if (!stream.read(name.data(), static_cast<std::streamsize>(name.size())) || name.empty() ||
            records.find(name) != records.end())
        {
            std::cerr << "invalid or duplicate portable golden record name\n";
            return false;
        }

        TensorRecord record;
        record.shape.assign(dimensions, dimensions + rank);
        record.values.resize(static_cast<std::size_t>(element_count));
        if (element_count > 0 &&
            !stream.read(reinterpret_cast<char*>(record.values.data()),
                         static_cast<std::streamsize>(element_count * sizeof(float))))
        {
            std::cerr << "truncated portable golden record: " << name << "\n";
            return false;
        }
        records.emplace(std::move(name), std::move(record));
    }
    return true;
}

bool require_record(const std::map<std::string, TensorRecord>& records, const char* name, TensorRecord& result)
{
    const auto iterator = records.find(name);
    if (iterator == records.end())
    {
        std::cerr << "golden record is missing: " << name << "\n";
        return false;
    }
    result = iterator->second;
    return true;
}

bool shape_is_matrix(const TensorRecord& record, std::uint64_t rows, std::uint64_t columns)
{
    return record.shape.size() == 2 && record.shape[0] == rows && record.shape[1] == columns &&
           record.values.size() == static_cast<std::size_t>(rows * columns);
}

bool upload(const ncnn::Mat& input, ncnn::VulkanDevice* vkdev, const ncnn::Option& option,
            ncnn::VkMat& output)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_upload(input, output, option);
    return compute.submit_and_wait() == 0 && !output.empty();
}

bool download(const ncnn::VkMat& input, ncnn::VulkanDevice* vkdev, const ncnn::Option& option,
              ncnn::Mat& output)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(input, output, option);
    return compute.submit_and_wait() == 0 && !output.empty();
}

bool parse_float(const char* text, float& value)
{
    try
    {
        std::size_t consumed = 0;
        value = std::stof(text, &consumed);
        return consumed == std::string(text).size() && std::isfinite(value) && value >= 0.f;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_int(const char* text, int& value)
{
    try
    {
        std::size_t consumed = 0;
        value = std::stoi(text, &consumed);
        return consumed == std::string(text).size() && value >= 0;
    }
    catch (...)
    {
        return false;
    }
}

bool check_fp32_patch_contract(const seedvr2::ResolutionPlan& plan, ncnn::VulkanDevice* vkdev,
                               ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator,
                               const ncnn::Option& option)
{
    ncnn::Mat latent_cpu(plan.latent_width, plan.latent_height, 1, 16);
    if (latent_cpu.empty())
        return false;
    for (std::size_t index = 0; index < latent_cpu.total(); ++index)
        latent_cpu[index] = 0.01f * static_cast<float>((index * 29u) % 97u) - 0.48f;

    ncnn::VkMat latent_gpu;
    if (!upload(latent_cpu, vkdev, option, latent_gpu))
        return false;

    ncnn::VkMat input_patches_gpu;
    if (!seedvr2::make_dit_input_patches_gpu(latent_gpu, latent_gpu, plan, vkdev, blob_allocator,
                                              staging_allocator, input_patches_gpu, nullptr, nullptr, true) ||
        input_patches_gpu.empty() || input_patches_gpu.dims != 2 || input_patches_gpu.w != 132 ||
        input_patches_gpu.h != plan.video_tokens || input_patches_gpu.elemsize != sizeof(float))
    {
        std::cerr << "PT2 FP32 input patch contract failed\n";
        return false;
    }

    ncnn::VkMat output_patches_gpu;
    if (!seedvr2::patch_latent_for_dit_output_gpu(latent_gpu, plan, vkdev, blob_allocator, staging_allocator,
                                                  output_patches_gpu, nullptr, nullptr, true) ||
        output_patches_gpu.empty() || output_patches_gpu.dims != 2 || output_patches_gpu.w != 64 ||
        output_patches_gpu.h != plan.video_tokens || output_patches_gpu.elemsize != sizeof(float))
    {
        std::cerr << "PT2 FP32 output patch contract failed\n";
        return false;
    }

    ncnn::VkMat roundtrip_gpu;
    if (!seedvr2::unpatch_dit_output_gpu(output_patches_gpu, plan, vkdev, blob_allocator, staging_allocator,
                                         roundtrip_gpu, nullptr, nullptr, true))
    {
        std::cerr << "PT2 FP32 unpatch contract failed\n";
        return false;
    }
    ncnn::Mat roundtrip_cpu;
    if (!download(roundtrip_gpu, vkdev, option, roundtrip_cpu) || roundtrip_cpu.total() != latent_cpu.total() ||
        roundtrip_cpu.elemsize != sizeof(float))
    {
        std::cerr << "PT2 FP32 latent roundtrip download failed\n";
        return false;
    }

    float maximum_error = 0.f;
    for (std::size_t index = 0; index < latent_cpu.total(); ++index)
        maximum_error = std::max(maximum_error, std::fabs(latent_cpu[index] - roundtrip_cpu[index]));
    std::cout << "pt2_patch_contract_vulkan latent_roundtrip_max_abs=" << maximum_error
              << " input_patch_shape=" << input_patches_gpu.w << "x" << input_patches_gpu.h
              << " output_patch_shape=" << output_patches_gpu.w << "x" << output_patches_gpu.h << "\n";
    return maximum_error <= 1.0e-6f;
}

void usage(const char* program)
{
    std::cerr << "usage: " << program
              << " <stack-dir> <dit_stack_golden.f32> [gpu-id] [--max-abs value]"
                 " [--check-patch-roundtrip] [--bf16] [--repeat count] [--trace-stages] [--compare-stages]\n";
}

std::uint64_t fingerprint(const float* values, std::size_t count)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t index = 0; index < count; ++index)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, values + index, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8)
        {
            hash ^= (bits >> shift) & 0xffu;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        usage(argv[0]);
        return 2;
    }

    const std::string stack_dir = argv[1];
    const std::string golden_path = argv[2];
    int gpu_id = 0;
    float max_abs_limit = -1.f;
    bool check_patch_roundtrip = false;
    bool bf16 = false;
    bool trace_stages = false;
    bool compare_stages = false;
    int repeat_count = 1;
    int argument_index = 3;
    while (argument_index < argc)
    {
        const std::string argument = argv[argument_index];
        if (argument == "--compare-stages")
        {
            trace_stages = compare_stages = true;
            ++argument_index;
            continue;
        }
        if (argument == "--trace-stages")
        {
            trace_stages = true;
            ++argument_index;
            continue;
        }
        if (argument == "--bf16")
        {
            bf16 = true;
            ++argument_index;
            continue;
        }
        if (argument == "--repeat")
        {
            if (argument_index + 1 >= argc || !parse_int(argv[argument_index + 1], repeat_count) || repeat_count < 1)
            {
                usage(argv[0]);
                return 2;
            }
            argument_index += 2;
            continue;
        }
        if (argument == "--max-abs")
        {
            if (argument_index + 1 >= argc || !parse_float(argv[argument_index + 1], max_abs_limit))
            {
                usage(argv[0]);
                return 2;
            }
            argument_index += 2;
            continue;
        }
        if (argument == "--check-patch-roundtrip")
        {
            check_patch_roundtrip = true;
            ++argument_index;
            continue;
        }
        if (argument_index == 3 && parse_int(argv[argument_index], gpu_id))
        {
            ++argument_index;
            continue;
        }
        usage(argv[0]);
        return 2;
    }
    if (bf16 && check_patch_roundtrip)
    {
        std::cerr << "--check-patch-roundtrip requires FP32 storage\n";
        return 2;
    }

    std::map<std::string, TensorRecord> records;
    if (!read_golden(golden_path, records))
        return 1;
    TensorRecord input_video;
    TensorRecord input_text;
    TensorRecord expected_output;
    TensorRecord timestep_record;
    if (!require_record(records, "input_video_patches", input_video) ||
        !require_record(records, "input_text", input_text) || !require_record(records, "output_video", expected_output) ||
        !require_record(records, "timestep", timestep_record))
        return 1;
    if (timestep_record.values.size() != 1 || !std::isfinite(timestep_record.values[0]))
    {
        std::cerr << "golden timestep must contain one finite float\n";
        return 1;
    }
    const float timestep = timestep_record.values[0];

    if (input_video.shape.size() != 2 || input_video.shape[1] != 132 || input_video.shape[0] == 0 ||
        !shape_is_matrix(input_text, input_text.shape.empty() ? 0 : input_text.shape[0], 5120) ||
        expected_output.shape.size() != 2 || expected_output.shape[0] != input_video.shape[0] ||
        expected_output.shape[1] != 64 || expected_output.values.size() != input_video.shape[0] * 64)
    {
        std::cerr << "golden stack records do not match the fixed matrix contract\n";
        return 1;
    }
    const int video_tokens = static_cast<int>(input_video.shape[0]);
    const int source_side = static_cast<int>(std::sqrt(static_cast<double>(video_tokens)));
    if (source_side <= 0 || source_side * source_side != video_tokens)
    {
        std::cerr << "golden video token count must be a square fixed source grid, got " << video_tokens << "\n";
        return 1;
    }

    seedvr2::ResolutionPlan plan;
    std::string plan_error;
    if (!seedvr2::ResolutionPlan::from_explicit(source_side * 16, source_side * 16, plan, &plan_error))
    {
        std::cerr << "cannot create fixed stack resolution plan: " << plan_error << "\n";
        return 1;
    }

    if (ncnn::create_gpu_instance() != 0)
    {
        std::cerr << "failed to initialize Vulkan\n";
        return 1;
    }
    struct InstanceGuard final
    {
        ~InstanceGuard() { ncnn::destroy_gpu_instance(); }
    } instance_guard;

    if (gpu_id < 0 || gpu_id >= ncnn::get_gpu_count())
    {
        std::cerr << "requested Vulkan device is unavailable: " << gpu_id << "\n";
        return 1;
    }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_id);
    if (!vkdev)
    {
        std::cerr << "failed to acquire Vulkan device: " << gpu_id << "\n";
        return 1;
    }
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator)
    {
        std::cerr << "failed to acquire Vulkan allocators\n";
        if (blob_allocator)
            vkdev->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator)
            vkdev->reclaim_staging_allocator(staging_allocator);
        return 1;
    }

    ncnn::Net transfer;
    if (bf16)
        seedvr2::configure_dit_vulkan_net(transfer, vkdev, blob_allocator, staging_allocator, nullptr);
    else
        seedvr2::configure_dit_vulkan_net_fp32(transfer, vkdev, blob_allocator, staging_allocator, nullptr);
    ncnn::Mat video_cpu(132, video_tokens, sizeof(float));
    ncnn::Mat text_cpu(static_cast<int>(input_text.shape[1]), static_cast<int>(input_text.shape[0]), sizeof(float));
    std::copy(input_video.values.begin(), input_video.values.end(), reinterpret_cast<float*>(video_cpu.data));
    std::copy(input_text.values.begin(), input_text.values.end(), reinterpret_cast<float*>(text_cpu.data));

    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    bool ok = upload(video_cpu, vkdev, transfer.opt, video_gpu) && upload(text_cpu, vkdev, transfer.opt, text_gpu);
    if (ok && check_patch_roundtrip)
        ok = check_fp32_patch_contract(plan, vkdev, blob_allocator, staging_allocator, transfer.opt);
    seedvr2::DitStackSession session;
    if (ok)
    {
        ok = seedvr2::DitStackSession::open(stack_dir, plan, vkdev, blob_allocator, staging_allocator, session,
                                             nullptr, nullptr, !bf16);
    }

    ncnn::VkMat output_gpu;
    const char* trace_directory = std::getenv("SEEDVR2_TRACE_DIRECTORY");
    seedvr2::DitStackObserver observer = [&](const char* name, const ncnn::VkMat& gpu) {
        ncnn::Mat cpu;
        if (!download(gpu, vkdev, transfer.opt, cpu) || cpu.elempack != 1 || cpu.elemsize != sizeof(float))
            return false;
        const float* values = reinterpret_cast<const float*>(cpu.data);
        for (std::size_t index = 0; index < cpu.total(); ++index)
            if (!std::isfinite(values[index]))
            {
                std::cerr << "non-finite serial stage " << name << " at " << index << "\n";
                return false;
            }
        std::cout << "serial_stage name=" << name << " fnv1a64=" << std::hex
                  << fingerprint(values, cpu.total()) << std::dec << std::endl;
        if (trace_directory && *trace_directory)
        {
            const std::string path = std::string(trace_directory) + "/" + name + ".raw";
            std::ofstream trace(path, std::ios::binary | std::ios::trunc);
            trace.write(reinterpret_cast<const char*>(values),
                        static_cast<std::streamsize>(cpu.total() * sizeof(float)));
            if (!trace)
            {
                std::cerr << "cannot write serial stage trace " << path << "\n";
                return false;
            }
        }
        if (compare_stages)
        {
            std::string key = name;
            if (key == "input_video" || key == "input_text")
                key += "_hidden";
            else if (key == "output")
                key = "output_video";
            else if (key.compare(0, 6, "block_") == 0 && key.size() > 7 && key[7] == '_')
                key.insert(6, "0");
            const auto found = records.find(key);
            if (found == records.end() || found->second.values.size() != cpu.total())
            {
                std::cerr << "missing or mismatched serial stage reference: " << key << "\n";
                return false;
            }
            double squared = 0.;
            float maximum = 0.f;
            float peak = 0.f;
            for (std::size_t index = 0; index < cpu.total(); ++index)
            {
                const float expected = found->second.values[index];
                if (!std::isfinite(expected))
                    return false;
                const float difference = std::fabs(values[index] - expected);
                squared += static_cast<double>(difference) * difference;
                maximum = std::max(maximum, difference);
                peak = std::max(peak, std::fabs(expected));
            }
            std::cout << std::setprecision(10) << "serial_stage_error {\"stage\":\"" << key
                      << "\",\"max_abs\":" << maximum << ",\"rmse\":" << std::sqrt(squared / cpu.total())
                      << ",\"reference_peak\":" << peak << "}" << std::endl;
        }
        return true;
    };
    const auto* stage_observer = trace_stages ? &observer : nullptr;
    if (ok)
        ok = session.run(video_gpu, text_cpu, timestep, plan, output_gpu, stage_observer);

    ncnn::Mat output_cpu;
    if (ok)
        ok = download(output_gpu, vkdev, transfer.opt, output_cpu);
    if (!ok)
    {
        std::cerr << "serial DitStackSession Vulkan execution failed\n";
        session.clear();
        transfer.clear();
        vkdev->reclaim_blob_allocator(blob_allocator);
        vkdev->reclaim_staging_allocator(staging_allocator);
        return 1;
    }

    const std::size_t output_count = output_cpu.total();
    if (output_count != expected_output.values.size() || output_cpu.elempack != 1 || output_cpu.elemsize != sizeof(float))
    {
        std::cerr << "unexpected serial output shape/count: dims=" << output_cpu.dims << " w=" << output_cpu.w
                  << " h=" << output_cpu.h << " d=" << output_cpu.d << " c=" << output_cpu.c
                  << " count=" << output_count << " elemsize=" << output_cpu.elemsize
                  << " elempack=" << output_cpu.elempack << "\n";
        session.clear();
        transfer.clear();
        vkdev->reclaim_blob_allocator(blob_allocator);
        vkdev->reclaim_staging_allocator(staging_allocator);
        return 1;
    }

    const float* actual = reinterpret_cast<const float*>(output_cpu.data);
    double squared_error = 0.0;
    float maximum_error = 0.f;
    for (std::size_t index = 0; index < output_count; ++index)
    {
        const float difference = std::isfinite(actual[index]) && std::isfinite(expected_output.values[index])
            ? std::fabs(actual[index] - expected_output.values[index])
            : std::numeric_limits<float>::infinity();
        maximum_error = std::max(maximum_error, difference);
        squared_error += static_cast<double>(difference) * difference;
    }
    const double rmse = std::sqrt(squared_error / static_cast<double>(output_count));
    std::cout << "serial_stack_vulkan gpu=" << gpu_id << " output_shape=" << output_cpu.w << "x" << output_cpu.h
              << " max_abs=" << maximum_error << " rmse=" << rmse
              << " precision=" << (bf16 ? "bf16" : "fp32") << " fnv1a64=" << std::hex
              << fingerprint(actual, output_count) << std::dec << std::endl;

    bool repeat_ok = true;
    const std::vector<float> initial(actual, actual + output_count);
    for (int iteration = 1; iteration < repeat_count; ++iteration)
    {
        output_gpu.release();
        ncnn::Mat repeated;
        if (!session.run(video_gpu, text_cpu, timestep, plan, output_gpu, stage_observer) ||
            !download(output_gpu, vkdev, transfer.opt, repeated) || repeated.total() != output_count ||
            repeated.elempack != 1 || repeated.elemsize != sizeof(float))
        {
            std::cerr << "serial repeat execution/shape failed at iteration " << iteration << "\n";
            repeat_ok = false;
            break;
        }
        const float* values = reinterpret_cast<const float*>(repeated.data);
        float repeat_max = 0.f;
        double repeat_squared = 0.;
        for (std::size_t index = 0; index < output_count; ++index)
        {
            const float difference = std::isfinite(values[index]) && std::isfinite(initial[index])
                ? std::fabs(values[index] - initial[index]) : std::numeric_limits<float>::infinity();
            repeat_max = std::max(repeat_max, difference);
            repeat_squared += static_cast<double>(difference) * difference;
        }
        std::cout << "serial_stack_repeat iteration=" << iteration << " max_abs=" << repeat_max
                  << " rmse=" << std::sqrt(repeat_squared / output_count) << " fnv1a64=" << std::hex
                  << fingerprint(values, output_count) << std::dec << std::endl;
        repeat_ok = repeat_max == 0.f && fingerprint(values, output_count) ==
                    fingerprint(initial.data(), output_count) && repeat_ok;
    }

    output_gpu.release();
    video_gpu.release();
    text_gpu.release();
    session.clear();
    transfer.clear();
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return !repeat_ok || !std::isfinite(maximum_error) || !std::isfinite(rmse) ||
                   (max_abs_limit >= 0.f && maximum_error > max_abs_limit)
               ? 1 : 0;
}
