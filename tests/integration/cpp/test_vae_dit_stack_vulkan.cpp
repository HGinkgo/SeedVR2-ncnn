#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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
    if (argc != 5 && argc != 6 && argc != 8)
    {
        std::fprintf(stderr,
                     "usage: test_vae_dit_stack_vulkan <encode-stem> <stack-dir> <decode-stem> "
                     "<condition-f32> [text-tokens [height width]]\n");
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
    const int dimension_offset = argc == 8 ? 6 : 0;
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
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", sample_gpu) != 0 || extractor.extract("out0", latent_gpu, compute) != 0 ||
            compute.submit_and_wait() != 0)
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

    ncnn::VkMat input_patches_gpu;
    std::fprintf(stderr, "stage=dit-input-patchify\n");
    if (!seedvr2::make_dit_input_patches_gpu(noise_gpu, latent_gpu, resolution_plan, vkdev, blob_allocator,
                                             staging_allocator, input_patches_gpu))
    {
        std::fprintf(stderr, "stage=dit-input-patchify failed\n");
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

    ncnn::VkMat noise_patches_gpu;
    std::fprintf(stderr, "stage=noise-patchify\n");
    if (!seedvr2::patch_latent_for_dit_output_gpu(noise_gpu, resolution_plan, vkdev, blob_allocator,
                                                   staging_allocator, noise_patches_gpu))
    {
        std::fprintf(stderr, "stage=noise-patchify failed\n");
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

    ncnn::VkMat output_latent_gpu;
    std::fprintf(stderr, "stage=latent-unpatch\n");
    if (!seedvr2::unpatch_dit_output_gpu(endpoint_patches_gpu, resolution_plan, vkdev, blob_allocator,
                                          staging_allocator, output_latent_gpu))
    {
        std::fprintf(stderr, "stage=latent-unpatch failed\n");
        return 1;
    }

    ncnn::VkAllocator* decode_blob_allocator = vkdev->acquire_blob_allocator();
    seedvr2::TransientVkStagingAllocator decode_staging_allocator(vkdev);
    ncnn::VkMat decode_latent_gpu;
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

    decode_latent_gpu.release();
    decode.clear();
    decode_blob_allocator->clear();
    decode_staging_allocator.clear();
    vkdev->reclaim_blob_allocator(decode_blob_allocator);
    std::puts("seedvr2-vae-dit-stack-vulkan: ok");
    return 0;
}
