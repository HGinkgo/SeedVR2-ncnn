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
#include "conditioning/conditioning.h"
#include "dit/dit_stack.h"
#include "gpu.h"
#include "net.h"
#include "sampler/vulkan_sampler.h"
#include "vae/temporal_pad.h"
#include "vulkan/transient_staging_allocator.h"

namespace
{

constexpr int kLatentChannels = 16;
constexpr char kGoldenMagic[8] = {'S', 'V', 'R', '2', 'F', '3', '2', '\0'};
constexpr std::uint32_t kGoldenVersion = 1;
constexpr float kVaeAbsoluteTolerance = 2.e-2f;
constexpr float kVaeRelativeTolerance = 2.e-2f;
constexpr float kDitAbsoluteTolerance = 8.e-3f;
constexpr float kDitRelativeTolerance = 8.e-3f;
// Upstream VAE output is compared before entering the DiT stack. Its small
// Vulkan-vs-PyTorch error can be amplified by 32 DiT blocks, so the full-chain
// boundary uses the VAE gate while the isolated DiT test keeps its tighter gate.
constexpr float kFullChainAbsoluteTolerance = 2.e-2f;
constexpr float kFullChainRelativeTolerance = 2.e-2f;

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
        if (name_length == 0 || rank == 0 || rank > 4 || count == 0 || count > (1ull << 31))
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

bool compare_flat(const char* name, const ncnn::Mat& actual, const GoldenRecord& expected,
                  const std::vector<std::uint64_t>& shape, float absolute_tolerance, float relative_tolerance)
{
    if (expected.shape != shape || actual.empty() || actual.elemsize != 4u ||
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
        if (!std::isfinite(values[index]) || !std::isfinite(expected.values[index]))
        {
            std::fprintf(stderr, "%s contains non-finite value at %zu\n", name, index);
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

bool compare_matrix(const char* name, const ncnn::Mat& actual, const GoldenRecord& expected, int height, int width,
                    float absolute_tolerance = kDitAbsoluteTolerance,
                    float relative_tolerance = kDitRelativeTolerance)
{
    return actual.dims == 2 && actual.w == width && actual.h == height &&
           compare_flat(name, actual, expected,
                        {static_cast<std::uint64_t>(height), static_cast<std::uint64_t>(width)},
                        absolute_tolerance, relative_tolerance);
}

bool compare_cthw(const char* name, const ncnn::Mat& actual, const GoldenRecord& expected,
                  int channels, int height, int width, float absolute_tolerance, float relative_tolerance)
{
    return actual.dims == 3 && actual.w == width && actual.h == height && actual.c == channels &&
           compare_flat(name, actual, expected,
                        {static_cast<std::uint64_t>(channels), 1u, static_cast<std::uint64_t>(height),
                         static_cast<std::uint64_t>(width)},
                        absolute_tolerance, relative_tolerance);
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

bool configure(ncnn::Net& net, ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
               ncnn::VkAllocator* staging_allocator)
{
    net.opt = make_vulkan_option(blob_allocator, staging_allocator);
    net.set_vulkan_device(vkdev);
    return true;
}

bool load_vae(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
              ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator,
              bool low_memory_cpu = false)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    net.opt.use_local_pool_allocator = !low_memory_cpu;
    register_seedvr2_vae_layers(net);
    return net.load_param((stem + ".ncnn.param").c_str()) == 0 &&
           net.load_model((stem + ".ncnn.bin").c_str()) == 0;
}

bool clone_to_allocator(const ncnn::VkMat& source, ncnn::VkMat& destination, ncnn::VulkanDevice* vkdev,
                        ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_clone(source, destination, make_vulkan_option(blob_allocator, staging_allocator));
    return !destination.empty() && compute.submit_and_wait() == 0;
}

bool finite(const ncnn::Mat& value)
{
    if (value.empty())
        return false;
    const float* data = static_cast<const float*>(value.data);
    for (size_t index = 0; index < value.total(); index++)
        if (!std::isfinite(data[index]))
            return false;
    return true;
}

bool parse_dimension(const char* text, int& value)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

void print_stats(const char* label, const ncnn::Mat& value)
{
    size_t nonfinite = 0;
    float maximum = 0.f;
    const float* data = static_cast<const float*>(value.data);
    for (size_t index = 0; index < value.total(); index++)
    {
        if (!std::isfinite(data[index]))
        {
            nonfinite++;
            continue;
        }
        maximum = std::max(maximum, std::fabs(data[index]));
    }
    std::fprintf(stderr, "%s total=%zu nonfinite=%zu max_abs=%g\n", label, value.total(), nonfinite, maximum);
}

bool download(const ncnn::VkMat& source, ncnn::Mat& destination, ncnn::VulkanDevice* vkdev,
              const ncnn::Option& opt)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(source, destination, opt);
    return compute.submit_and_wait() == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6 && argc != 8 && argc != 9)
    {
        std::fprintf(stderr,
                     "usage: test_vae_dit_stack_vulkan <encode-stem> <stack-dir> <decode-stem> "
                     "<condition-f32> [text-tokens [height width [golden.f32]]]\n");
        return 2;
    }

    const std::string encode_stem = argv[1];
    const std::string stack_dir = argv[2];
    const std::string decode_stem = argv[3];
    const std::string condition_path = argv[4];
    const int text_tokens = argc >= 6 ? std::atoi(argv[5]) : 58;
    if (text_tokens != 58 && text_tokens != 64)
    {
        std::fprintf(stderr, "text-tokens must be 58 or 64\n");
        return 2;
    }

    int image_height = 128;
    int image_width = 128;
    const int dimension_offset = argc >= 8 ? 6 : 0;
    if (dimension_offset != 0 &&
        (!parse_dimension(argv[dimension_offset], image_height) ||
         !parse_dimension(argv[dimension_offset + 1], image_width)))
    {
        std::fprintf(stderr, "height and width must be positive integers\n");
        return 2;
    }
    seedvr2::ResolutionPlan resolution_plan;
    if (!seedvr2::ResolutionPlan::from_explicit(image_height, image_width, resolution_plan))
    {
        std::fprintf(stderr, "height and width must be positive multiples of 16\n");
        return 2;
    }

    std::unordered_map<std::string, GoldenRecord> golden_records;
    const bool verify_golden = argc == 9;
    if (verify_golden)
    {
        if (!load_golden(argv[8], golden_records))
        {
            std::fprintf(stderr, "failed to load full-chain golden: %s\n", argv[8]);
            return 2;
        }
        static const char* required_records[] = {"input", "vae_latent", "noise", "dit_input_patches", "text",
                                                 "dit_prediction_patches", "noise_patches", "endpoint_patches",
                                                 "output_latent", "reconstruction"};
        for (const char* name : required_records)
        {
            if (golden_records.count(name) == 0)
            {
                std::fprintf(stderr, "full-chain golden is missing record: %s\n", name);
                return 2;
            }
        }
    }

    std::fprintf(stderr, "stage=initialize-vulkan\n");
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device available\n");
        return 1;
    }
    std::fprintf(stderr, "vulkan-heap-budget-mib=%u\n", vkdev->get_heap_budget());
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net encode;
    std::fprintf(stderr, "stage=load-encode\n");
    if (!load_vae(encode, encode_stem, vkdev, blob_allocator, staging_allocator))
    {
        std::fprintf(stderr, "failed to load VAE or packing graph\n");
        return 1;
    }

    ncnn::Mat sample(image_width, image_height, 1, 3);
    sample.fill(0.f);
    ncnn::VkMat sample_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(sample, sample_gpu, encode.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat latent_gpu;
    std::fprintf(stderr, "stage=vae-encode\n");
    {
        ncnn::Extractor extractor = encode.create_extractor();
        extractor.set_light_mode(false);
        ncnn::VkCompute compute(vkdev);
        const int input_status = extractor.input("in0", sample_gpu);
        const int extract_status = input_status == 0 ? extractor.extract("out0", latent_gpu, compute) : -1;
        const int submit_status = extract_status == 0 ? compute.submit_and_wait() : -1;
        if (input_status != 0 || extract_status != 0 || submit_status != 0)
        {
            std::fprintf(stderr, "stage=vae-encode failed input=%d extract=%d submit=%d\n", input_status,
                         extract_status, submit_status);
            return 1;
        }
    }
    if (verify_golden)
    {
        ncnn::Mat latent;
        if (!download(latent_gpu, latent, vkdev, encode.opt) ||
            !compare_cthw("vae_latent", latent, golden_records.at("vae_latent"), kLatentChannels,
                          resolution_plan.latent_height, resolution_plan.latent_width, kVaeAbsoluteTolerance,
                          kVaeRelativeTolerance))
            return 1;
    }
    sample_gpu = ncnn::VkMat();
    const ncnn::Option encode_opt = encode.opt;
    encode.clear();

    ncnn::Mat text;
    std::fprintf(stderr, "stage=conditioning-load\n");
    if (!seedvr2::load_conditioning_f32(condition_path.c_str(), text_tokens, text))
    {
        std::fprintf(stderr, "stage=conditioning-load failed path=%s\n", condition_path.c_str());
        return 1;
    }

    ncnn::Mat noise(resolution_plan.latent_width, resolution_plan.latent_height, 1, kLatentChannels);
    for (size_t index = 0; index < noise.total(); index++)
        noise[index] = 0.01f * static_cast<float>((index * 17) % 101) - 0.5f;
    ncnn::VkMat noise_gpu;
    std::fprintf(stderr, "stage=noise-upload\n");
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(noise, noise_gpu, encode_opt);
        if (compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "stage=noise-upload failed\n");
            return 1;
        }
    }

    // Keep the DiT graph weights scoped to the inference section so the
    // decoder never competes with the 32-block stack for Vulkan memory.
    ncnn::VkAllocator* decode_blob_allocator = vkdev->acquire_blob_allocator();
    seedvr2::TransientVkStagingAllocator decode_staging_allocator(vkdev);
    ncnn::VkMat decode_latent_gpu;
    ncnn::VkMat output_latent_gpu;
    {
    ncnn::VkMat input_patches_gpu;
    std::fprintf(stderr, "stage=dit-input-patchify\n");
    if (!seedvr2::make_dit_input_patches_gpu(noise_gpu, latent_gpu, resolution_plan, vkdev, blob_allocator,
                                             staging_allocator, input_patches_gpu))
    {
        std::fprintf(stderr, "stage=dit-input-patchify failed\n");
        return 1;
    }
    if (verify_golden)
    {
        ncnn::Mat input_patches;
        if (!download(input_patches_gpu, input_patches, vkdev, encode_opt) ||
            !compare_matrix("dit_input_patches", input_patches, golden_records.at("dit_input_patches"),
                            resolution_plan.video_tokens, 132, kVaeAbsoluteTolerance, kVaeRelativeTolerance))
            return 1;
    }
    ncnn::VkMat prediction_gpu;
    std::fprintf(stderr, "stage=positive-dit-stack\n");
    if (!seedvr2::run_dit_stack_gpu(input_patches_gpu, text, 1000.f, stack_dir, resolution_plan, vkdev,
                                    blob_allocator, staging_allocator, prediction_gpu))
    {
        std::fprintf(stderr, "stage=positive-dit-stack failed\n");
        return 1;
    }
    if (verify_golden)
    {
        ncnn::Mat prediction;
        if (!download(prediction_gpu, prediction, vkdev, encode_opt) ||
            !compare_matrix("dit_prediction_patches", prediction, golden_records.at("dit_prediction_patches"),
                            resolution_plan.video_tokens, 64, kFullChainAbsoluteTolerance,
                            kFullChainRelativeTolerance))
            return 1;
    }

    ncnn::VkMat noise_patches_gpu;
    std::fprintf(stderr, "stage=noise-patchify\n");
    if (!seedvr2::patch_latent_for_dit_output_gpu(noise_gpu, resolution_plan, vkdev, blob_allocator,
                                                   staging_allocator, noise_patches_gpu))
    {
        std::fprintf(stderr, "stage=noise-patchify failed\n");
        return 1;
    }
    if (verify_golden)
    {
        ncnn::Mat noise_patches;
        if (!download(noise_patches_gpu, noise_patches, vkdev, encode_opt) ||
            !compare_matrix("noise_patches", noise_patches, golden_records.at("noise_patches"),
                            resolution_plan.video_tokens, 64))
            return 1;
    }

    ncnn::VkMat endpoint_patches_gpu;
    std::fprintf(stderr, "stage=v-lerp-endpoint\n");
    if (!seedvr2::apply_cfg_v_lerp_endpoint_vulkan(prediction_gpu, noise_patches_gpu, vkdev, blob_allocator,
                                                    staging_allocator, endpoint_patches_gpu))
    {
        std::fprintf(stderr, "stage=v-lerp-endpoint failed\n");
        return 1;
    }
    if (verify_golden)
    {
        ncnn::Mat endpoint_patches;
        if (!download(endpoint_patches_gpu, endpoint_patches, vkdev, encode_opt) ||
            !compare_matrix("endpoint_patches", endpoint_patches, golden_records.at("endpoint_patches"),
                            resolution_plan.video_tokens, 64, kFullChainAbsoluteTolerance,
                            kFullChainRelativeTolerance))
            return 1;
    }

    std::fprintf(stderr, "stage=latent-unpatch\n");
    if (!seedvr2::unpatch_dit_output_gpu(endpoint_patches_gpu, resolution_plan, vkdev, blob_allocator,
                                          staging_allocator, output_latent_gpu))
    {
        std::fprintf(stderr, "stage=latent-unpatch failed\n");
        return 1;
    }
    if (verify_golden)
    {
        ncnn::Mat output_latent;
        if (!download(output_latent_gpu, output_latent, vkdev, encode_opt) ||
            !compare_cthw("output_latent", output_latent, golden_records.at("output_latent"), kLatentChannels,
                          resolution_plan.latent_height, resolution_plan.latent_width, kFullChainAbsoluteTolerance,
                          kFullChainRelativeTolerance))
            return 1;
    }

    std::fprintf(stderr, "stage=handoff-latent\n");
    if (!clone_to_allocator(output_latent_gpu, decode_latent_gpu, vkdev, decode_blob_allocator,
                            &decode_staging_allocator))
    {
        std::fprintf(stderr, "stage=handoff-latent failed\n");
        return 1;
    }

    latent_gpu.release();
    noise_gpu.release();
    input_patches_gpu.release();
    prediction_gpu.release();
    noise_patches_gpu.release();
    endpoint_patches_gpu.release();
    output_latent_gpu.release();
    blob_allocator->clear();
    staging_allocator->clear();
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    }

    ncnn::Net decode;
    std::fprintf(stderr, "stage=load-decode\n");
    if (!load_vae(decode, decode_stem, vkdev, decode_blob_allocator, &decode_staging_allocator, true))
    {
        std::fprintf(stderr, "stage=load-decode failed\n");
        return 1;
    }

    ncnn::Mat reconstruction;
    std::fprintf(stderr, "stage=vae-decode\n");
    {
        ncnn::Extractor extractor = decode.create_extractor();
        extractor.set_light_mode(false);
        if (extractor.input("in0", decode_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
        {
            std::fprintf(stderr, "stage=vae-decode failed\n");
            return 1;
        }
    }
    if (!finite(reconstruction) || reconstruction.dims != 3 || reconstruction.w != image_width ||
        reconstruction.h != image_height || reconstruction.c != 3)
    {
        ncnn::Mat debug_latent;
        if (download(decode_latent_gpu, debug_latent, vkdev, decode.opt))
            print_stats("output-latent", debug_latent);
        print_stats("reconstruction", reconstruction);
        std::fprintf(stderr, "stage=decode-download-or-shape failed dims=%d w=%d h=%d c=%d\n", reconstruction.dims,
                     reconstruction.w, reconstruction.h, reconstruction.c);
        return 1;
    }
    if (verify_golden &&
        !compare_cthw("reconstruction", reconstruction, golden_records.at("reconstruction"), 3, image_height,
                      image_width, kFullChainAbsoluteTolerance, kFullChainRelativeTolerance))
        return 1;

    decode_latent_gpu.release();
    decode.clear();
    decode_blob_allocator->clear();
    decode_staging_allocator.clear();
    vkdev->reclaim_blob_allocator(decode_blob_allocator);
    std::puts("seedvr2-vae-dit-stack-vulkan: ok");
    return 0;
}
