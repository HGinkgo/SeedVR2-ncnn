#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "command.h"
#include "conditioning/conditioning.h"
#include "dit/dit_stack.h"
#include "gpu.h"

namespace
{

constexpr char kGoldenMagic[8] = {'S', 'V', 'R', '2', 'F', '3', '2', '\0'};
constexpr std::uint32_t kGoldenVersion = 1;
constexpr float kAbsoluteTolerance = 6.e-3f;
constexpr float kRelativeTolerance = 6.e-3f;

struct GoldenRecord
{
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

bool read_bytes(std::istream& input, void* destination, std::size_t size)
{
    return size == 0 || static_cast<bool>(input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size)));
}

std::uint16_t read_u16(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t read_u32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t read_u64(const std::uint8_t* bytes)
{
    std::uint64_t value = 0;
    for (int index = 0; index < 8; index++)
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    return value;
}

bool load_golden(const std::string& path, std::unordered_map<std::string, GoldenRecord>& records)
{
    std::ifstream input(path, std::ios::binary);
    std::uint8_t header[16];
    if (!input || !read_bytes(input, header, sizeof(header)) ||
        std::memcmp(header, kGoldenMagic, sizeof(kGoldenMagic)) != 0 || read_u32(header + 8) != kGoldenVersion)
        return false;

    const std::uint32_t record_count = read_u32(header + 12);
    if (record_count == 0 || record_count > 1024)
        return false;
    for (std::uint32_t record_index = 0; record_index < record_count; record_index++)
    {
        std::uint8_t record_header[44];
        if (!read_bytes(input, record_header, sizeof(record_header)))
            return false;
        const std::uint16_t name_length = read_u16(record_header);
        const std::uint8_t rank = record_header[2];
        const std::uint64_t count = read_u64(record_header + 4);
        if (name_length == 0 || rank > 4 || count == 0 || count > (1ull << 31))
            return false;

        std::string name(name_length, '\0');
        if (!read_bytes(input, name.data(), name.size()) || records.count(name) != 0)
            return false;
        std::uint64_t expected_count = 1;
        GoldenRecord record;
        record.shape.resize(rank);
        for (std::uint8_t dimension = 0; dimension < rank; dimension++)
        {
            record.shape[dimension] = read_u64(record_header + 12 + dimension * 8);
            if (record.shape[dimension] == 0 ||
                expected_count > std::numeric_limits<std::uint64_t>::max() / record.shape[dimension])
                return false;
            expected_count *= record.shape[dimension];
        }
        if (expected_count != count || count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        std::vector<std::uint8_t> raw(static_cast<std::size_t>(count) * sizeof(float));
        if (!read_bytes(input, raw.data(), raw.size()))
            return false;
        record.values.resize(static_cast<std::size_t>(count));
        for (std::size_t value_index = 0; value_index < record.values.size(); value_index++)
        {
            const std::uint32_t bits = read_u32(raw.data() + value_index * sizeof(float));
            std::memcpy(&record.values[value_index], &bits, sizeof(float));
        }
        records.emplace(std::move(name), std::move(record));
    }
    return true;
}

bool make_matrix(const GoldenRecord& record, int width, int height, ncnn::Mat& matrix)
{
    if (record.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(height), static_cast<std::uint64_t>(width)} ||
        record.values.size() != static_cast<std::size_t>(width) * height)
        return false;
    matrix.create(width, height);
    if (matrix.empty())
        return false;
    std::copy(record.values.begin(), record.values.end(), static_cast<float*>(matrix.data));
    return true;
}

bool configure(ncnn::Option& opt, ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    return true;
}

bool compare_output(const ncnn::Mat& actual, const GoldenRecord& expected, float& maximum)
{
    if (actual.empty() || actual.dims != 2 || actual.w != 64 || actual.h != 3600 || actual.elemsize != 4u ||
        expected.shape != std::vector<std::uint64_t>{3600, 64} || actual.total() != expected.values.size())
    {
        std::fprintf(stderr,
                     "dynamic stack output shape mismatch: actual=(dims=%d,w=%d,h=%d,elemsize=%zu,total=%zu) "
                     "expected=(3600,64,%zu)\n",
                     actual.dims, actual.w, actual.h, actual.elemsize, actual.total(), expected.values.size());
        return false;
    }
    const float* values = static_cast<const float*>(actual.data);
    maximum = 0.f;
    std::size_t maximum_index = 0;
    float maximum_actual = 0.f;
    float maximum_expected = 0.f;
    float maximum_ratio = 0.f;
    std::size_t maximum_ratio_index = 0;
    float maximum_ratio_actual = 0.f;
    float maximum_ratio_expected = 0.f;
    float maximum_ratio_allowed = 0.f;
    bool matched = true;
    std::size_t first_mismatch_index = 0;
    float first_mismatch_actual = 0.f;
    float first_mismatch_expected = 0.f;
    float first_mismatch_delta = 0.f;
    float first_mismatch_allowed = 0.f;
    for (std::size_t index = 0; index < expected.values.size(); index++)
    {
        if (!std::isfinite(values[index]))
        {
            std::fprintf(stderr, "dynamic stack output is non-finite at %zu: actual=%g\n", index, values[index]);
            return false;
        }
        const float delta = std::fabs(values[index] - expected.values[index]);
        const float allowed = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected.values[index]);
        if (delta > maximum)
        {
            maximum = delta;
            maximum_index = index;
            maximum_actual = values[index];
            maximum_expected = expected.values[index];
        }
        const float ratio = delta / allowed;
        if (ratio > maximum_ratio)
        {
            maximum_ratio = ratio;
            maximum_ratio_index = index;
            maximum_ratio_actual = values[index];
            maximum_ratio_expected = expected.values[index];
            maximum_ratio_allowed = allowed;
        }
        if (delta > allowed && matched)
        {
            matched = false;
            first_mismatch_index = index;
            first_mismatch_actual = values[index];
            first_mismatch_expected = expected.values[index];
            first_mismatch_delta = delta;
            first_mismatch_allowed = allowed;
        }
    }
    std::fprintf(stderr,
                 "dynamic stack golden values=%zu max_abs_error=%g at %zu actual=%g expected=%g; "
                 "max_ratio=%g at %zu actual=%g expected=%g allowed=%g\n",
                 expected.values.size(), maximum, maximum_index, maximum_actual, maximum_expected, maximum_ratio,
                 maximum_ratio_index, maximum_ratio_actual, maximum_ratio_expected, maximum_ratio_allowed);
    if (!matched)
    {
        std::fprintf(stderr,
                     "dynamic stack first mismatch at %zu: actual=%g expected=%g error=%g allowed=%g\n",
                     first_mismatch_index, first_mismatch_actual, first_mismatch_expected, first_mismatch_delta,
                     first_mismatch_allowed);
    }
    return matched;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: test_dynamic_dit_stack_vulkan <1x45x80-export-dir>\n");
        return 2;
    }

    seedvr2::ResolutionPlan plan;
    if (!seedvr2::ResolutionPlan::from_explicit(720, 1280, plan) || plan.source_height != 45 ||
        plan.source_width != 80 || plan.video_tokens != 3600)
        return 1;

    const std::string stack_dir = argv[1];
    std::unordered_map<std::string, GoldenRecord> records;
    if (!load_golden(stack_dir + "/dit_stack_golden.f32", records))
    {
        std::fprintf(stderr, "failed to load dynamic stack golden\n");
        return 1;
    }
    const auto patches = records.find("input_video_patches");
    const auto text = records.find("input_text");
    const auto timestep = records.find("timestep");
    const auto expected = records.find("output_video");
    if (patches == records.end() || text == records.end() || timestep == records.end() || expected == records.end() ||
        timestep->second.values.size() != 1)
    {
        std::fprintf(stderr, "dynamic stack golden is incomplete\n");
        return 1;
    }

    ncnn::Mat patch_matrix;
    ncnn::Mat text_matrix;
    if (!make_matrix(patches->second, 132, plan.video_tokens, patch_matrix) ||
        !make_matrix(text->second, 5120, 58, text_matrix))
    {
        std::fprintf(stderr, "dynamic stack golden shapes are incompatible\n");
        return 1;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device available\n");
        return 1;
    }
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    ncnn::Option opt;
    configure(opt, blob_allocator, staging_allocator);

    int result = 1;
    ncnn::VkMat patches_gpu;
    ncnn::VkMat output_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(patch_matrix, patches_gpu, opt);
        if (upload.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "dynamic stack patch upload failed\n");
        }
        else if (!seedvr2::run_dit_stack_gpu(patches_gpu, text_matrix, timestep->second.values[0], stack_dir, plan,
                                             vkdev, blob_allocator, staging_allocator, output_gpu))
        {
            std::fprintf(stderr, "dynamic stack execution failed\n");
        }
        else
        {
            ncnn::Mat output;
            ncnn::VkCompute download(vkdev);
            download.record_download(output_gpu, output, opt);
            if (download.submit_and_wait() != 0)
            {
                std::fprintf(stderr, "dynamic stack output download failed\n");
            }
            else
            {
                float error = 0.f;
                if (compare_output(output, expected->second, error))
                {
                    std::printf("seedvr2-dynamic-dit-stack-vulkan: ok max_abs_error=%g\n", error);
                    result = 0;
                }
                else
                {
                    std::fprintf(stderr, "dynamic stack output mismatches golden\n");
                }
            }
        }
    }
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return result;
}
