#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "command.h"
#include "gpu.h"
#include "net.h"
#include "vae/temporal_pad.h"
#include "vulkan/transient_staging_allocator.h"

namespace
{

constexpr char kGoldenMagic[8] = {'S', 'V', 'R', '2', 'F', '3', '2', '\0'};
constexpr std::uint32_t kGoldenVersion = 1;
constexpr float kAbsoluteTolerance = 2.e-2f;
constexpr float kRelativeTolerance = 2.e-2f;

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

bool load_golden(const char* path, std::unordered_map<std::string, GoldenRecord>& records)
{
    std::ifstream input(path, std::ios::binary);
    std::uint8_t header[16];
    if (!input || !read_bytes(input, header, sizeof(header)) ||
        std::memcmp(header, kGoldenMagic, sizeof(kGoldenMagic)) != 0 || read_u32(header + 8) != kGoldenVersion)
        return false;

    const std::uint32_t record_count = read_u32(header + 12);
    if (record_count != 3)
        return false;
    for (std::uint32_t record_index = 0; record_index < record_count; record_index++)
    {
        std::uint8_t record_header[44];
        if (!read_bytes(input, record_header, sizeof(record_header)))
            return false;
        const std::uint16_t name_length = read_u16(record_header);
        const std::uint8_t rank = record_header[2];
        const std::uint64_t count = read_u64(record_header + 4);
        if (name_length == 0 || rank != 4 || count == 0 || count > (1ull << 31))
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
    return records.count("input") == 1 && records.count("latent") == 1 && records.count("reconstruction") == 1;
}

bool make_cthw_mat(const GoldenRecord& record, ncnn::Mat& matrix)
{
    if (record.shape.size() != 4 || record.shape[0] == 0 || record.shape[1] != 1 || record.shape[2] == 0 ||
        record.shape[3] == 0 || record.shape[0] > static_cast<std::uint64_t>(INT_MAX) ||
        record.shape[2] > static_cast<std::uint64_t>(INT_MAX) || record.shape[3] > static_cast<std::uint64_t>(INT_MAX) ||
        record.values.size() != static_cast<std::size_t>(record.shape[0] * record.shape[1] * record.shape[2] * record.shape[3]))
        return false;

    matrix.create(static_cast<int>(record.shape[3]), static_cast<int>(record.shape[2]), 1,
                  static_cast<int>(record.shape[0]));
    if (matrix.empty())
        return false;
    std::copy(record.values.begin(), record.values.end(), static_cast<float*>(matrix.data));
    return true;
}

bool compare_cthw(const char* name, const ncnn::Mat& actual, const GoldenRecord& expected, float absolute_tolerance,
                  float relative_tolerance)
{
    if (expected.shape.size() != 4 || expected.shape[1] != 1 || actual.empty() || actual.dims != 3 ||
        actual.w != static_cast<int>(expected.shape[3]) || actual.h != static_cast<int>(expected.shape[2]) ||
        actual.c != static_cast<int>(expected.shape[0]) || actual.elemsize != 4u ||
        actual.total() != expected.values.size())
    {
        std::fprintf(stderr, "%s golden shape mismatch\n", name);
        return false;
    }

    const float* values = static_cast<const float*>(actual.data);
    float maximum = 0.f;
    float maximum_ratio = 0.f;
    std::size_t maximum_index = 0;
    bool matched = true;
    for (std::size_t index = 0; index < expected.values.size(); index++)
    {
        if (!std::isfinite(values[index]))
        {
            std::fprintf(stderr, "%s output is non-finite at %zu\n", name, index);
            return false;
        }
        const float delta = std::fabs(values[index] - expected.values[index]);
        const float allowed = absolute_tolerance + relative_tolerance * std::fabs(expected.values[index]);
        if (delta > maximum)
        {
            maximum = delta;
            maximum_index = index;
        }
        maximum_ratio = std::max(maximum_ratio, delta / allowed);
        if (delta > allowed)
            matched = false;
    }
    std::fprintf(stderr, "%s golden values=%zu max_abs_error=%g at %zu max_ratio=%g\n", name,
                 expected.values.size(), maximum, maximum_index, maximum_ratio);
    return matched;
}

ncnn::Option make_vulkan_option(ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    return opt;
}

bool load_graph(ncnn::Net& net, const char* param_path, const char* model_path,
                ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
                ncnn::VkAllocator* staging_allocator, bool low_memory_cpu = false)
{
    net.opt = make_vulkan_option(blob_allocator, staging_allocator);
    net.opt.use_local_pool_allocator = !low_memory_cpu;
    net.set_vulkan_device(vkdev);
    register_seedvr2_vae_layers(net);
    return net.load_param(param_path) == 0 && net.load_model(model_path) == 0;
}

bool clone_to_allocator(const ncnn::VkMat& source, ncnn::VkMat& destination, ncnn::VulkanDevice* vkdev,
                        ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_clone(source, destination, make_vulkan_option(blob_allocator, staging_allocator));
    return !destination.empty() && compute.submit_and_wait() == 0;
}

bool parse_dimension(const char* text, int& value)
{
    char* end = 0;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;

    value = static_cast<int>(parsed);
    return true;
}

bool finite(const ncnn::Mat& values)
{
    const float* data = values;
    for (size_t index = 0; index < values.total(); index++)
    {
        if (!std::isfinite(data[index]))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 7 && argc != 8)
    {
        std::fprintf(stderr,
                     "usage: test_vae_end_to_end_vulkan <encode.param> <encode.bin> <decode.param> <decode.bin> "
                     "[height width [golden.f32]]\n");
        return 2;
    }

    int sample_height = 128;
    int sample_width = 128;
    if (argc >= 7 &&
        (!parse_dimension(argv[5], sample_height) || !parse_dimension(argv[6], sample_width)))
    {
        std::fprintf(stderr, "height and width must be positive integers\n");
        return 2;
    }
    std::unordered_map<std::string, GoldenRecord> golden_records;
    const bool verify_golden = argc == 8;
    if (verify_golden && !load_golden(argv[7], golden_records))
    {
        std::fprintf(stderr, "failed to load VAE golden: %s\n", argv[7]);
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;
    std::fprintf(stderr, "vulkan-heap-budget-mib=%u\n", vkdev->get_heap_budget());
    ncnn::VkAllocator* encode_blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* encode_staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net encode;
    std::fprintf(stderr, "stage=load-encode\n");
    if (!load_graph(encode, argv[1], argv[2], vkdev, encode_blob_allocator, encode_staging_allocator))
        return 1;

    ncnn::Mat sample(sample_width, sample_height, 1, 3);
    if (verify_golden && !make_cthw_mat(golden_records.at("input"), sample))
    {
        std::fprintf(stderr, "VAE golden input has an incompatible CTHW shape\n");
        return 2;
    }
    if (!verify_golden)
        sample.fill(0.f);
    if (sample.dims != 4 || sample.w != sample_width || sample.h != sample_height || sample.d != 1 || sample.c != 3 ||
        !finite(sample))
    {
        std::fprintf(stderr, "VAE sample shape does not match the requested resolution\n");
        return 2;
    }
    ncnn::VkMat sample_gpu;
    {
        std::fprintf(stderr, "stage=upload-sample\n");
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(sample, sample_gpu, encode.opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat latent_gpu;
    {
        ncnn::Extractor encode_extractor = encode.create_extractor();
        encode_extractor.set_light_mode(false);
        if (encode_extractor.input("in0", sample_gpu) != 0)
            return 1;
        std::fprintf(stderr, "stage=encode-extract\n");
        ncnn::VkCompute compute(vkdev);
        if (encode_extractor.extract("out0", latent_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Mat latent;
    if (verify_golden)
    {
        std::fprintf(stderr, "stage=download-latent\n");
        ncnn::VkCompute download(vkdev);
        download.record_download(latent_gpu, latent, encode.opt);
        if (download.submit_and_wait() != 0 ||
            !compare_cthw("latent", latent, golden_records.at("latent"), kAbsoluteTolerance, kRelativeTolerance))
            return 1;
    }

    // Decode is the peak-memory phase at large resolutions. Release the
    // completed encoder's graph, input, and extractor before allocating it.
    sample_gpu.release();
    encode.clear();

    ncnn::VkAllocator* decode_blob_allocator = vkdev->acquire_blob_allocator();
    seedvr2::TransientVkStagingAllocator decode_staging_allocator(vkdev);
    ncnn::VkMat decode_latent_gpu;
    std::fprintf(stderr, "stage=handoff-latent\n");
    if (!clone_to_allocator(latent_gpu, decode_latent_gpu, vkdev, decode_blob_allocator, &decode_staging_allocator))
        return 1;

    latent_gpu.release();
    encode_blob_allocator->clear();
    encode_staging_allocator->clear();
    vkdev->reclaim_blob_allocator(encode_blob_allocator);
    vkdev->reclaim_staging_allocator(encode_staging_allocator);

    ncnn::Net decode;
    std::fprintf(stderr, "stage=load-decode\n");
    if (!load_graph(decode, argv[3], argv[4], vkdev, decode_blob_allocator, &decode_staging_allocator, true))
        return 1;

    ncnn::Mat reconstruction;
    {
        ncnn::Extractor decode_extractor = decode.create_extractor();
        decode_extractor.set_light_mode(false);
        if (decode_extractor.input("in0", decode_latent_gpu) != 0)
            return 1;
        std::fprintf(stderr, "stage=decode-extract\n");
        // The graph ends in the custom causal Conv3D layer. Host extraction
        // keeps the final image boundary explicit while convolution runs on
        // the Vulkan path.
        if (decode_extractor.extract("out0", reconstruction) != 0)
        {
            return 1;
        }
    }

    if (reconstruction.empty() || reconstruction.dims != 3 || reconstruction.w != sample_width ||
        reconstruction.h != sample_height || reconstruction.c != 3 ||
        reconstruction.total() != static_cast<size_t>(3) * sample_width * sample_height ||
        !finite(reconstruction))
    {
        std::fprintf(stderr, "unexpected Vulkan reconstruction shape\n");
        return 1;
    }
    if (verify_golden &&
        !compare_cthw("reconstruction", reconstruction, golden_records.at("reconstruction"), kAbsoluteTolerance,
                      kRelativeTolerance))
        return 1;

    decode_latent_gpu.release();
    decode.clear();
    decode_blob_allocator->clear();
    decode_staging_allocator.clear();
    vkdev->reclaim_blob_allocator(decode_blob_allocator);

    std::puts("seedvr2-vae-end-to-end-vulkan: ok");
    return 0;
}
