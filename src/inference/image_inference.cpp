#include "inference/image_inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

#include "platform.h"

#if NCNN_VULKAN
#include "allocator.h"
#include "command.h"
#include "conditioning/conditioning.h"
#include "dit/dit_stack.h"
#include "gpu.h"
#include "net.h"
#include "sampler/vulkan_sampler.h"
#include "vae/temporal_pad.h"
#include "vulkan/transient_staging_allocator.h"
#endif

namespace seedvr2
{
namespace
{

constexpr int kLatentChannels = 16;

bool valid_rgb_image(const RgbImage& image)
{
    if (image.width <= 0 || image.height <= 0)
        return false;
    const std::size_t expected = static_cast<std::size_t>(image.width) *
                                 static_cast<std::size_t>(image.height) * 3u;
    return image.pixels.size() == expected;
}

#if NCNN_VULKAN

class GpuInstance final
{
public:
    bool open(std::string& error)
    {
        if (ncnn::create_gpu_instance() != 0)
        {
            error = "failed to initialize the Vulkan runtime";
            return false;
        }
        active_ = true;
        return true;
    }

    ~GpuInstance()
    {
        if (active_)
            ncnn::destroy_gpu_instance();
    }

private:
    bool active_ = false;
};

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

bool load_vae(ncnn::Net& net,
              const std::filesystem::path& stem,
              ncnn::VulkanDevice* vkdev,
              ncnn::VkAllocator* blob_allocator,
              ncnn::VkAllocator* staging_allocator,
              bool low_memory_cpu)
{
    net.opt = make_vulkan_option(blob_allocator, staging_allocator);
    net.opt.use_local_pool_allocator = !low_memory_cpu;
    net.set_vulkan_device(vkdev);
    register_seedvr2_vae_layers(net);
    const std::string stem_string = stem.string();
    return net.load_param((stem_string + ".ncnn.param").c_str()) == 0 &&
           net.load_model((stem_string + ".ncnn.bin").c_str()) == 0;
}

bool clone_to_allocator(const ncnn::VkMat& source,
                        ncnn::VkMat& destination,
                        ncnn::VulkanDevice* vkdev,
                        ncnn::VkAllocator* blob_allocator,
                        ncnn::VkAllocator* staging_allocator)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_clone(source, destination, make_vulkan_option(blob_allocator, staging_allocator));
    return !destination.empty() && compute.submit_and_wait() == 0;
}

bool prepare_input(const RgbImage& input, const ResolutionPlan& plan, ncnn::Mat& prepared)
{
    if (!valid_rgb_image(input) || plan.resized_width <= 0 || plan.resized_height <= 0 || plan.crop_left < 0 ||
        plan.crop_top < 0 || plan.crop_left + plan.image_width > plan.resized_width ||
        plan.crop_top + plan.image_height > plan.resized_height)
        return false;

    prepared.create(plan.image_width, plan.image_height, 1, 3);
    if (prepared.empty())
        return false;

    // Match the official pipeline: area resize, centered crop, then [0, 1] to [-1, 1].
    for (int target_y = 0; target_y < plan.image_height; target_y++)
    {
        const int resized_y = target_y + plan.crop_top;
        const float source_y0 = static_cast<float>(resized_y) * input.height / plan.resized_height;
        const float source_y1 = static_cast<float>(resized_y + 1) * input.height / plan.resized_height;
        const int source_y_begin = std::max(0, static_cast<int>(std::floor(source_y0)));
        const int source_y_end = std::min(input.height, static_cast<int>(std::ceil(source_y1)));
        for (int target_x = 0; target_x < plan.image_width; target_x++)
        {
            const int resized_x = target_x + plan.crop_left;
            const float source_x0 = static_cast<float>(resized_x) * input.width / plan.resized_width;
            const float source_x1 = static_cast<float>(resized_x + 1) * input.width / plan.resized_width;
            const int source_x_begin = std::max(0, static_cast<int>(std::floor(source_x0)));
            const int source_x_end = std::min(input.width, static_cast<int>(std::ceil(source_x1)));
            const float area = (source_x1 - source_x0) * (source_y1 - source_y0);
            if (area <= 0.f)
                return false;

            float channels[3] = {0.f, 0.f, 0.f};
            for (int source_y = source_y_begin; source_y < source_y_end; source_y++)
            {
                const float overlap_y = std::max(0.f, std::min(source_y1, static_cast<float>(source_y + 1)) -
                                                          std::max(source_y0, static_cast<float>(source_y)));
                if (overlap_y == 0.f)
                    continue;
                for (int source_x = source_x_begin; source_x < source_x_end; source_x++)
                {
                    const float overlap_x = std::max(0.f, std::min(source_x1, static_cast<float>(source_x + 1)) -
                                                              std::max(source_x0, static_cast<float>(source_x)));
                    if (overlap_x == 0.f)
                        continue;
                    const float weight = overlap_x * overlap_y / area;
                    const std::size_t offset = (static_cast<std::size_t>(source_y) * input.width + source_x) * 3u;
                    for (int channel = 0; channel < 3; channel++)
                        channels[channel] += static_cast<float>(input.pixels[offset + channel]) * weight / 255.f;
                }
            }

            for (int channel = 0; channel < 3; channel++)
                prepared.channel(channel).row(target_y)[target_x] = channels[channel] * 2.f - 1.f;
        }
    }
    return true;
}

bool reconstruction_to_rgb(const ncnn::Mat& reconstruction, const ResolutionPlan& plan, RgbImage& output)
{
    if (reconstruction.empty() || !reconstruction.data || (reconstruction.dims != 3 && reconstruction.dims != 4) ||
        reconstruction.w != plan.image_width || reconstruction.h != plan.image_height || reconstruction.c != 3 ||
        reconstruction.d != 1 || reconstruction.elemsize != 4u)
        return false;

    output.width = plan.image_width;
    output.height = plan.image_height;
    output.pixels.resize(static_cast<std::size_t>(output.width) * output.height * 3u);
    for (int channel = 0; channel < 3; channel++)
    {
        const ncnn::Mat plane = reconstruction.channel(channel);
        for (int y = 0; y < output.height; y++)
        {
            const float* source = plane.row(y);
            for (int x = 0; x < output.width; x++)
            {
                const float value = source[x];
                if (!std::isfinite(value))
                    return false;
                const float normalized = std::max(0.f, std::min(1.f, value * 0.5f + 0.5f));
                output.pixels[(static_cast<std::size_t>(y) * output.width + x) * 3u + channel] =
                    static_cast<unsigned char>(std::lround(normalized * 255.f));
            }
        }
    }
    return true;
}

bool run_vulkan_image_inference(const ModelGraphSet& graphs,
                                const RgbImage& input,
                                const ResolutionPlan& plan,
                                ncnn::VulkanDevice* vkdev,
                                RgbImage& output,
                                std::string& error)
{
    ncnn::Mat sample;
    if (!prepare_input(input, plan, sample))
    {
        error = "failed to prepare the input image";
        return false;
    }

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator)
    {
        error = "failed to acquire Vulkan allocators";
        if (blob_allocator)
            vkdev->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator)
            vkdev->reclaim_staging_allocator(staging_allocator);
        return false;
    }

    ncnn::Net encode;
    std::fprintf(stderr, "stage=load-encode\n");
    if (!load_vae(encode, graphs.vae_encode_stem, vkdev, blob_allocator, staging_allocator, false))
    {
        error = "stage=load-encode failed";
        vkdev->reclaim_blob_allocator(blob_allocator);
        vkdev->reclaim_staging_allocator(staging_allocator);
        return false;
    }

    ncnn::VkMat sample_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(sample, sample_gpu, encode.opt);
        if (compute.submit_and_wait() != 0)
        {
            error = "stage=input-upload failed";
            return false;
        }
    }

    ncnn::VkMat latent_gpu;
    std::fprintf(stderr, "stage=vae-encode\n");
    {
        ncnn::Extractor extractor = encode.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", sample_gpu) != 0 || extractor.extract("out0", latent_gpu, compute) != 0 ||
            compute.submit_and_wait() != 0)
        {
            error = "stage=vae-encode failed";
            return false;
        }
    }
    sample_gpu.release();
    const ncnn::Option encode_opt = encode.opt;
    encode.clear();

    ncnn::Mat text;
    std::fprintf(stderr, "stage=conditioning-load\n");
    if (!load_conditioning_f32(graphs.conditioning_path.string().c_str(), graphs.text_tokens, text))
    {
        error = "stage=conditioning-load failed";
        return false;
    }

    ncnn::Mat noise(plan.latent_width, plan.latent_height, 1, kLatentChannels);
    if (noise.empty())
    {
        error = "stage=noise-create failed";
        return false;
    }
    for (std::size_t index = 0; index < noise.total(); index++)
        noise[index] = 0.01f * static_cast<float>((index * 17u) % 101u) - 0.5f;

    ncnn::VkMat noise_gpu;
    std::fprintf(stderr, "stage=noise-upload\n");
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(noise, noise_gpu, encode_opt);
        if (compute.submit_and_wait() != 0)
        {
            error = "stage=noise-upload failed";
            return false;
        }
    }

    ncnn::VkMat input_patches_gpu;
    std::fprintf(stderr, "stage=dit-input-patchify\n");
    if (!make_dit_input_patches_gpu(noise_gpu, latent_gpu, plan, vkdev, blob_allocator, staging_allocator,
                                    input_patches_gpu))
    {
        error = "stage=dit-input-patchify failed";
        return false;
    }

    ncnn::VkMat prediction_gpu;
    std::fprintf(stderr, "stage=dit-stack\n");
    if (!run_dit_stack_gpu(input_patches_gpu, text, 1000.f, graphs.dit_stack_dir.string(), plan, vkdev,
                           blob_allocator, staging_allocator, prediction_gpu))
    {
        error = "stage=dit-stack failed";
        return false;
    }

    ncnn::VkMat noise_patches_gpu;
    std::fprintf(stderr, "stage=noise-patchify\n");
    if (!patch_latent_for_dit_output_gpu(noise_gpu, plan, vkdev, blob_allocator, staging_allocator,
                                         noise_patches_gpu))
    {
        error = "stage=noise-patchify failed";
        return false;
    }

    ncnn::VkMat endpoint_patches_gpu;
    std::fprintf(stderr, "stage=v-lerp-endpoint\n");
    if (!apply_cfg_v_lerp_endpoint_vulkan(prediction_gpu, noise_patches_gpu, vkdev, blob_allocator,
                                          staging_allocator, endpoint_patches_gpu))
    {
        error = "stage=v-lerp-endpoint failed";
        return false;
    }

    ncnn::VkMat output_latent_gpu;
    std::fprintf(stderr, "stage=latent-unpatch\n");
    if (!unpatch_dit_output_gpu(endpoint_patches_gpu, plan, vkdev, blob_allocator, staging_allocator,
                                output_latent_gpu))
    {
        error = "stage=latent-unpatch failed";
        return false;
    }

    ncnn::VkAllocator* decode_blob_allocator = vkdev->acquire_blob_allocator();
    TransientVkStagingAllocator decode_staging_allocator(vkdev);
    ncnn::VkMat decode_latent_gpu;
    std::fprintf(stderr, "stage=handoff-latent\n");
    if (!decode_blob_allocator ||
        !clone_to_allocator(output_latent_gpu, decode_latent_gpu, vkdev, decode_blob_allocator,
                            &decode_staging_allocator))
    {
        error = "stage=handoff-latent failed";
        if (decode_blob_allocator)
            vkdev->reclaim_blob_allocator(decode_blob_allocator);
        return false;
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
    if (!load_vae(decode, graphs.vae_decode_stem, vkdev, decode_blob_allocator, &decode_staging_allocator, true))
    {
        error = "stage=load-decode failed";
        return false;
    }

    ncnn::Mat reconstruction;
    std::fprintf(stderr, "stage=vae-decode\n");
    {
        ncnn::Extractor extractor = decode.create_extractor();
        if (extractor.input("in0", decode_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
        {
            error = "stage=vae-decode failed";
            return false;
        }
    }
    if (!reconstruction_to_rgb(reconstruction, plan, output))
    {
        error = "stage=output-postprocess failed";
        return false;
    }

    decode_latent_gpu.release();
    decode.clear();
    decode_blob_allocator->clear();
    decode_staging_allocator.clear();
    vkdev->reclaim_blob_allocator(decode_blob_allocator);
    return true;
}

#endif

} // namespace

bool run_image_inference(const ModelGraphSet& graphs,
                         const RgbImage& input,
                         const ResolutionPlan& plan,
                         int gpu_id,
                         RgbImage& output,
                         std::string& error)
{
    output = RgbImage();
    error.clear();
#if NCNN_VULKAN
    if (!valid_rgb_image(input) || plan.image_width <= 0 || plan.image_height <= 0 || plan.latent_width <= 0 ||
        plan.latent_height <= 0)
    {
        error = "input image or resolution plan is invalid";
        return false;
    }
    if (gpu_id < -1)
    {
        error = "Vulkan GPU id must be greater than or equal to -1";
        return false;
    }

    GpuInstance instance;
    std::fprintf(stderr, "stage=initialize-vulkan\n");
    if (!instance.open(error))
        return false;
    const int selected_gpu = gpu_id >= 0 ? gpu_id : ncnn::get_default_gpu_index();
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(selected_gpu);
    if (!vkdev)
    {
        error = "requested Vulkan GPU is unavailable";
        return false;
    }
    return run_vulkan_image_inference(graphs, input, plan, vkdev, output, error);
#else
    (void)graphs;
    (void)input;
    (void)plan;
    (void)gpu_id;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

} // namespace seedvr2
