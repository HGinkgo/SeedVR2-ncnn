#include "inference/image_inference.h"
#include "inference/memory_diagnostics.h"
#include "inference/latent_spool.h"
#include "inference/rgb_spool.h"
#include "postprocess/color_fix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

    void close()
    {
        if (active_)
        {
            ncnn::destroy_gpu_instance();
            active_ = false;
        }
    }

    ~GpuInstance()
    {
        close();
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

std::uint64_t query_max_allocation_mib(const ncnn::VulkanDevice* vkdev)
{
    if (!vkdev->info.support_VK_KHR_maintenance3() || !ncnn::vkGetPhysicalDeviceProperties2KHR)
        return 0;

    VkPhysicalDeviceMaintenance3Properties maintenance3{};
    maintenance3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &maintenance3;
    ncnn::vkGetPhysicalDeviceProperties2KHR(vkdev->info.physicalDevice(), &properties);
    return static_cast<std::uint64_t>(maintenance3.maxMemoryAllocationSize / (1024u * 1024u));
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

struct VulkanInferenceContext final
{
    GpuInstance instance;
    ncnn::VulkanDevice* vkdev = nullptr;
    VulkanMemoryDiagnostics diagnostics;
    ncnn::VkAllocator* encode_blob_allocator = nullptr;
    ncnn::VkAllocator* encode_staging_allocator = nullptr;
    ncnn::VkAllocator* dit_blob_allocator = nullptr;
    ncnn::VkAllocator* dit_staging_allocator = nullptr;
    ncnn::VkAllocator* decode_blob_allocator = nullptr;
    std::unique_ptr<TransientVkStagingAllocator> decode_staging_allocator;
    ncnn::Mat text;
    ResolutionPlan plan;
    ModelGraphSet graphs;

    void clear()
    {
        decode_staging_allocator.reset();
        if (vkdev)
        {
            if (decode_blob_allocator)
                vkdev->reclaim_blob_allocator(decode_blob_allocator);
            if (dit_blob_allocator)
                vkdev->reclaim_blob_allocator(dit_blob_allocator);
            if (encode_blob_allocator)
                vkdev->reclaim_blob_allocator(encode_blob_allocator);
            if (dit_staging_allocator)
                vkdev->reclaim_staging_allocator(dit_staging_allocator);
            if (encode_staging_allocator)
                vkdev->reclaim_staging_allocator(encode_staging_allocator);
        }
        decode_blob_allocator = nullptr;
        dit_blob_allocator = nullptr;
        encode_blob_allocator = nullptr;
        dit_staging_allocator = nullptr;
        encode_staging_allocator = nullptr;
        vkdev = nullptr;
        instance.close();
    }

    ~VulkanInferenceContext() { clear(); }
};

void clear_frame_staging_allocators(VulkanInferenceContext& context)
{
    if (context.encode_blob_allocator)
        context.encode_blob_allocator->clear();
    if (context.dit_blob_allocator)
        context.dit_blob_allocator->clear();
    if (context.decode_blob_allocator)
        context.decode_blob_allocator->clear();
    if (context.encode_staging_allocator)
        context.encode_staging_allocator->clear();
    if (context.dit_staging_allocator)
        context.dit_staging_allocator->clear();
    if (context.decode_staging_allocator)
        context.decode_staging_allocator->clear();
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

bool initialize_vulkan_context(const ModelGraphSet& graphs,
                               const ResolutionPlan& plan,
                               int gpu_id,
                               std::uint32_t memory_budget_mib,
                               VulkanInferenceContext& context,
                               std::string& error,
                               const PerformanceProfile& profile)
{
    if (gpu_id < -1)
    {
        error = "Vulkan GPU id must be greater than or equal to -1";
        return false;
    }

    const ProfileScope initialize_scope(profile, "initialize-vulkan");
    std::fprintf(stderr, "stage=initialize-vulkan\n");
    if (!context.instance.open(error))
        return false;
    const int selected_gpu = gpu_id >= 0 ? gpu_id : ncnn::get_default_gpu_index();
    context.vkdev = ncnn::get_gpu_device(selected_gpu);
    if (!context.vkdev)
    {
        error = "requested Vulkan GPU is unavailable";
        return false;
    }

    context.diagnostics.gpu_id = selected_gpu;
    context.diagnostics.device_name = context.vkdev->info.device_name();
    context.diagnostics.heap_budget_mib = context.vkdev->get_heap_budget();
    context.diagnostics.max_allocation_mib = query_max_allocation_mib(context.vkdev);
    context.diagnostics.target_width = plan.image_width;
    context.diagnostics.target_height = plan.image_height;
    std::fprintf(stderr, "vulkan-gpu=%d name=%s heap-budget-mib=%u max-allocation-mib=%llu target=%dx%d\n",
                 context.diagnostics.gpu_id, context.diagnostics.device_name.c_str(),
                 context.diagnostics.heap_budget_mib,
                 static_cast<unsigned long long>(context.diagnostics.max_allocation_mib),
                 context.diagnostics.target_width, context.diagnostics.target_height);
    if (memory_budget_mib > 0 && context.diagnostics.heap_budget_mib < memory_budget_mib)
    {
        error = format_vulkan_memory_preflight_error(context.diagnostics, memory_budget_mib);
        return false;
    }

    const auto stage_error = [&](const char* stage, const char* detail) {
        error = format_vulkan_stage_error(stage, context.diagnostics, detail);
    };
    context.encode_blob_allocator = context.vkdev->acquire_blob_allocator();
    context.encode_staging_allocator = context.vkdev->acquire_staging_allocator();
    context.dit_blob_allocator = context.vkdev->acquire_blob_allocator();
    context.dit_staging_allocator = context.vkdev->acquire_staging_allocator();
    context.decode_blob_allocator = context.vkdev->acquire_blob_allocator();
    context.decode_staging_allocator = std::make_unique<TransientVkStagingAllocator>(context.vkdev);
    if (!context.encode_blob_allocator || !context.encode_staging_allocator || !context.dit_blob_allocator ||
        !context.dit_staging_allocator || !context.decode_blob_allocator || !context.decode_staging_allocator)
    {
        stage_error("allocator-acquire", "failed to acquire Vulkan allocators");
        return false;
    }

    std::fprintf(stderr, "stage=conditioning-load\n");
    if (!load_conditioning_f32(graphs.conditioning_path.string().c_str(), graphs.text_tokens, context.text))
    {
        error = "stage=conditioning-load failed";
        return false;
    }
    context.plan = plan;
    context.graphs = graphs;
    return true;
}

bool encode_batch_vulkan(const std::vector<RgbImage>& inputs,
                         const ResolutionPlan& plan,
                         VulkanInferenceContext& context,
                         std::vector<ncnn::Mat>& condition_latents,
                         std::string& error,
                         const PerformanceProfile& profile,
                         std::size_t frame_offset)
{
    condition_latents.clear();
    const auto stage_error = [&](std::size_t frame_index, const char* stage, const char* detail) {
        error = "frame=" + std::to_string(frame_index) + " " +
                format_vulkan_stage_error(stage, context.diagnostics, detail);
    };
    {
        ncnn::Net encode;
        {
            const ProfileScope load_scope(profile, "load-encode");
            std::fprintf(stderr, "stage=load-encode\n");
            if (!load_vae(encode, context.graphs.vae_encode_stem, context.vkdev, context.encode_blob_allocator,
                          context.encode_staging_allocator, false))
            {
                stage_error(0, "load-encode", "ncnn graph load returned failure");
                return false;
            }
        }
        const ncnn::Option encode_opt = encode.opt;

        for (std::size_t frame_index = 0; frame_index < inputs.size(); frame_index++)
        {
            const ProfileScope frame_scope(profile, "vae-encode", frame_offset + frame_index);
            ncnn::Mat sample;
            if (!prepare_input(inputs[frame_index], plan, sample))
            {
                error = "frame=" + std::to_string(frame_index) + " failed to prepare the input image";
                return false;
            }

            ncnn::VkMat sample_gpu;
            {
                ncnn::VkCompute compute(context.vkdev);
                compute.record_upload(sample, sample_gpu, encode_opt);
                if (compute.submit_and_wait() != 0)
                {
                    stage_error(frame_index, "input-upload", "Vulkan command submission returned failure");
                    return false;
                }
            }

            ncnn::VkMat latent_gpu;
            std::fprintf(stderr, "stage=vae-encode\n");
            {
                ncnn::Extractor extractor = encode.create_extractor();
                extractor.set_light_mode(false);
                ncnn::VkCompute compute(context.vkdev);
                if (extractor.input("in0", sample_gpu) != 0 || extractor.extract("out0", latent_gpu, compute) != 0 ||
                    compute.submit_and_wait() != 0)
                {
                    stage_error(frame_index, "vae-encode", "ncnn extraction or Vulkan command submission returned failure");
                    return false;
                }
            }
            sample_gpu.release();

            ncnn::Mat latent;
            {
                ncnn::VkCompute compute(context.vkdev);
                compute.record_download(latent_gpu, latent, encode_opt);
                if (compute.submit_and_wait() != 0 || latent.empty())
                {
                    stage_error(frame_index, "vae-encode", "latent download returned failure");
                    return false;
                }
            }
            condition_latents.push_back(std::move(latent));
        }
    }

    context.encode_blob_allocator->clear();
    context.encode_staging_allocator->clear();
    return true;
}

bool denoise_batch_vulkan(const std::vector<ncnn::Mat>& condition_latents,
                          const ResolutionPlan& plan,
                          VulkanInferenceContext& context,
                          std::vector<ncnn::Mat>& output_latents,
                          std::string& error,
                          const PerformanceProfile& profile,
                          std::size_t frame_offset)
{
    output_latents.clear();
    const auto stage_error = [&](std::size_t frame_index, const char* stage, const char* detail) {
        error = "frame=" + std::to_string(frame_index) + " " +
                format_vulkan_stage_error(stage, context.diagnostics, detail);
    };
    const ncnn::Option dit_opt = make_vulkan_option(context.dit_blob_allocator, context.dit_staging_allocator);

    ncnn::Mat noise(plan.latent_width, plan.latent_height, 1, kLatentChannels);
    if (noise.empty())
    {
        error = "stage=noise-create failed";
        return false;
    }
    for (std::size_t index = 0; index < noise.total(); index++)
        noise[index] = 0.01f * static_cast<float>((index * 17u) % 101u) - 0.5f;

    {
        DitStackSession dit;
        {
            const ProfileScope load_scope(profile, "load-dit-stack");
            std::fprintf(stderr, "stage=load-dit-stack\n");
            if (!DitStackSession::open(context.graphs.dit_stack_dir.string(), plan, context.vkdev,
                                       context.dit_blob_allocator, context.dit_staging_allocator, dit, &profile))
            {
                stage_error(0, "load-dit-stack", "ncnn graph load returned failure");
                return false;
            }
        }

        for (std::size_t frame_index = 0; frame_index < condition_latents.size(); frame_index++)
        {
            const ProfileScope frame_scope(profile, "dit-stack", frame_offset + frame_index);
            ncnn::VkMat condition_gpu;
            ncnn::VkMat noise_gpu;
            {
                ncnn::VkCompute compute(context.vkdev);
                compute.record_upload(condition_latents[frame_index], condition_gpu, dit_opt);
                compute.record_upload(noise, noise_gpu, dit_opt);
                std::fprintf(stderr, "stage=noise-upload\n");
                if (compute.submit_and_wait() != 0)
                {
                    stage_error(frame_index, "noise-upload", "Vulkan command submission returned failure");
                    return false;
                }
            }

            ncnn::VkMat input_patches_gpu;
            std::fprintf(stderr, "stage=dit-input-patchify\n");
            if (!make_dit_input_patches_gpu(noise_gpu, condition_gpu, plan, context.vkdev,
                                            context.dit_blob_allocator, context.dit_staging_allocator,
                                            input_patches_gpu))
            {
                stage_error(frame_index, "dit-input-patchify", "GPU patch assembly returned failure");
                return false;
            }

            ncnn::VkMat prediction_gpu;
            std::fprintf(stderr, "stage=dit-stack\n");
            if (!dit.run(input_patches_gpu, context.text, 1000.f, plan, prediction_gpu))
            {
                stage_error(frame_index, "dit-stack", "GPU DiT execution returned failure");
                return false;
            }

            ncnn::VkMat noise_patches_gpu;
            std::fprintf(stderr, "stage=noise-patchify\n");
            if (!patch_latent_for_dit_output_gpu(noise_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                                 context.dit_staging_allocator, noise_patches_gpu))
            {
                stage_error(frame_index, "noise-patchify", "GPU patch assembly returned failure");
                return false;
            }

            ncnn::VkMat endpoint_patches_gpu;
            std::fprintf(stderr, "stage=v-lerp-endpoint\n");
            if (!apply_cfg_v_lerp_endpoint_vulkan(prediction_gpu, noise_patches_gpu, context.vkdev,
                                                  context.dit_blob_allocator, context.dit_staging_allocator,
                                                  endpoint_patches_gpu))
            {
                stage_error(frame_index, "v-lerp-endpoint", "GPU sampler endpoint returned failure");
                return false;
            }

            ncnn::VkMat output_latent_gpu;
            std::fprintf(stderr, "stage=latent-unpatch\n");
            if (!unpatch_dit_output_gpu(endpoint_patches_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                        context.dit_staging_allocator, output_latent_gpu))
            {
                stage_error(frame_index, "latent-unpatch", "GPU patch removal returned failure");
                return false;
            }

            ncnn::Mat output_latent;
            std::fprintf(stderr, "stage=handoff-latent\n");
            {
                ncnn::VkCompute compute(context.vkdev);
                compute.record_download(output_latent_gpu, output_latent, dit_opt);
                if (compute.submit_and_wait() != 0 || output_latent.empty())
                {
                    stage_error(frame_index, "handoff-latent", "latent download returned failure");
                    return false;
                }
            }
            output_latents.push_back(std::move(output_latent));
        }
    }

    context.dit_blob_allocator->clear();
    context.dit_staging_allocator->clear();
    return true;
}

bool decode_batch_vulkan(const std::vector<ncnn::Mat>& output_latents,
                         const ResolutionPlan& plan,
                         VulkanInferenceContext& context,
                         std::vector<RgbImage>& outputs,
                         std::string& error,
                         const PerformanceProfile& profile,
                         std::size_t frame_offset)
{
    outputs.clear();
    const auto stage_error = [&](std::size_t frame_index, const char* stage, const char* detail) {
        error = "frame=" + std::to_string(frame_index) + " " +
                format_vulkan_stage_error(stage, context.diagnostics, detail);
    };
    {
        ncnn::Net decode;
        {
            const ProfileScope load_scope(profile, "load-decode");
            std::fprintf(stderr, "stage=load-decode\n");
            if (!load_vae(decode, context.graphs.vae_decode_stem, context.vkdev, context.decode_blob_allocator,
                          context.decode_staging_allocator.get(), true))
            {
                stage_error(0, "load-decode", "ncnn graph load returned failure");
                return false;
            }
        }

        for (std::size_t frame_index = 0; frame_index < output_latents.size(); frame_index++)
        {
            const ProfileScope frame_scope(profile, "vae-decode", frame_offset + frame_index);
            ncnn::VkMat decode_latent_gpu;
            {
                ncnn::VkCompute compute(context.vkdev);
                compute.record_upload(output_latents[frame_index], decode_latent_gpu, decode.opt);
                if (compute.submit_and_wait() != 0)
                {
                    stage_error(frame_index, "handoff-latent", "latent upload returned failure");
                    return false;
                }
            }

            ncnn::Mat reconstruction;
            std::fprintf(stderr, "stage=vae-decode\n");
            ncnn::Extractor extractor = decode.create_extractor();
            extractor.set_light_mode(false);
            if (extractor.input("in0", decode_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
            {
                stage_error(frame_index, "vae-decode", "ncnn extraction returned failure");
                return false;
            }
            if (!reconstruction_to_rgb(reconstruction, plan, outputs.emplace_back()))
            {
                outputs.pop_back();
                error = "frame=" + std::to_string(frame_index) + " stage=output-postprocess failed";
                return false;
            }
        }
    }

    context.decode_blob_allocator->clear();
    context.decode_staging_allocator->clear();
    return true;
}

bool ncnn_mat_to_latent_frame(const ncnn::Mat& latent, LatentFrame& frame)
{
    if (latent.empty() || !latent.data || (latent.dims != 3 && latent.dims != 4) || latent.d != 1 ||
        latent.w <= 0 || latent.h <= 0 || latent.c <= 0 || latent.elempack != 1 || latent.elemsize != 4u)
        return false;
    frame.width = latent.w;
    frame.height = latent.h;
    frame.channels = latent.c;
    const std::size_t count = latent.total();
    frame.values.resize(count);
    std::memcpy(frame.values.data(), latent.data, count * sizeof(float));
    return true;
}

bool latent_frame_to_ncnn_mat(const LatentFrame& frame, ncnn::Mat& latent)
{
    if (frame.width <= 0 || frame.height <= 0 || frame.channels <= 0 ||
        frame.values.size() != static_cast<std::size_t>(frame.width) * frame.height * frame.channels)
        return false;
    latent.create(frame.width, frame.height, 1, frame.channels);
    if (latent.empty())
        return false;
    std::memcpy(latent.data, frame.values.data(), frame.values.size() * sizeof(float));
    return true;
}

bool encode_video_vulkan(const ImageInferenceSession::VideoFrameReader& reader,
                         const ResolutionPlan& plan,
                         VulkanInferenceContext& context,
                         LatentSpool& condition_spool,
                         RgbFrameSpool& reference_spool,
                         std::size_t& frame_count,
                         std::string& error,
                         const PerformanceProfile& profile,
                         std::size_t frame_offset)
{
    frame_count = 0;
    if (!LatentSpool::create(condition_spool, error))
        return false;

    ncnn::Net encode;
    {
        const ProfileScope load_scope(profile, "load-encode");
        std::fprintf(stderr, "stage=load-encode\n");
        if (!load_vae(encode, context.graphs.vae_encode_stem, context.vkdev, context.encode_blob_allocator,
                      context.encode_staging_allocator, false))
        {
            error = format_vulkan_stage_error("load-encode", context.diagnostics,
                                              "ncnn graph load returned failure");
            return false;
        }
    }
    const ncnn::Option encode_opt = encode.opt;
    double read_ms = 0.0;
    for (;;)
    {
        RgbImage input;
        const auto read_start = PerformanceProfile::Clock::now();
        std::string read_error;
        const bool has_frame = reader(input, read_error);
        read_ms += profile.elapsed_ms(read_start);
        if (!has_frame)
        {
            if (!read_error.empty())
            {
                profile.report("video-read", read_ms);
                error = "stage=video-decode failed: " + read_error;
                return false;
            }
            break;
        }
        const std::size_t absolute_index = frame_offset + frame_count;
        std::fprintf(stderr, "stage=video-frame index=%zu\n", absolute_index);
        const ProfileScope frame_scope(profile, "vae-encode", absolute_index);
        ncnn::Mat sample;
        RgbImage reference;
        if (!prepare_color_reference(input, plan, reference, error) || !reference_spool.append(reference, error))
        {
            if (error.empty())
                error = "frame=" + std::to_string(absolute_index) + " failed to spool color reference";
            profile.report("video-read", read_ms);
            return false;
        }
        if (!prepare_input(input, plan, sample))
        {
            error = "frame=" + std::to_string(absolute_index) + " failed to prepare the input image";
            profile.report("video-read", read_ms);
            return false;
        }

        ncnn::VkMat sample_gpu;
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_upload(sample, sample_gpu, encode_opt);
            if (compute.submit_and_wait() != 0)
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("input-upload", context.diagnostics,
                                                  "Vulkan command submission returned failure");
                profile.report("video-read", read_ms);
                return false;
            }
        }

        ncnn::VkMat latent_gpu;
        std::fprintf(stderr, "stage=vae-encode\n");
        {
            ncnn::Extractor extractor = encode.create_extractor();
            extractor.set_light_mode(false);
            ncnn::VkCompute compute(context.vkdev);
            if (extractor.input("in0", sample_gpu) != 0 || extractor.extract("out0", latent_gpu, compute) != 0 ||
                compute.submit_and_wait() != 0)
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("vae-encode", context.diagnostics,
                                                  "ncnn extraction or Vulkan command submission returned failure");
                profile.report("video-read", read_ms);
                return false;
            }
        }
        sample_gpu.release();

        ncnn::Mat latent;
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_download(latent_gpu, latent, encode_opt);
            if (compute.submit_and_wait() != 0 || latent.empty())
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("vae-encode", context.diagnostics,
                                                  "latent download returned failure");
                profile.report("video-read", read_ms);
                return false;
            }
        }
        LatentFrame stored;
        if (!ncnn_mat_to_latent_frame(latent, stored) || !condition_spool.append(stored, error))
        {
            if (error.empty())
                error = "frame=" + std::to_string(absolute_index) + " stage=vae-encode latent spool append failed";
            profile.report("video-read", read_ms);
            return false;
        }
        frame_count++;
    }
    profile.report("video-read", read_ms);
    if (frame_count == 0)
    {
        error = "stage=video-decode failed: video contains no decodable frames";
        return false;
    }
    if (!condition_spool.rewind(error))
        return false;
    context.encode_blob_allocator->clear();
    context.encode_staging_allocator->clear();
    return true;
}

bool denoise_video_vulkan(LatentSpool& condition_spool,
                          const ResolutionPlan& plan,
                          VulkanInferenceContext& context,
                          LatentSpool& output_spool,
                          std::size_t& frame_count,
                          std::string& error,
                          const PerformanceProfile& profile,
                          std::size_t frame_offset)
{
    frame_count = 0;
    if (!LatentSpool::create(output_spool, error))
        return false;

    const ncnn::Option dit_opt = make_vulkan_option(context.dit_blob_allocator, context.dit_staging_allocator);
    ncnn::Mat noise(plan.latent_width, plan.latent_height, 1, kLatentChannels);
    if (noise.empty())
    {
        error = "stage=noise-create failed";
        return false;
    }
    for (std::size_t index = 0; index < noise.total(); index++)
        noise[index] = 0.01f * static_cast<float>((index * 17u) % 101u) - 0.5f;

    DitStackSession dit;
    {
        const ProfileScope load_scope(profile, "load-dit-stack");
        std::fprintf(stderr, "stage=load-dit-stack\n");
        if (!DitStackSession::open(context.graphs.dit_stack_dir.string(), plan, context.vkdev,
                                   context.dit_blob_allocator, context.dit_staging_allocator, dit, &profile))
        {
            error = format_vulkan_stage_error("load-dit-stack", context.diagnostics,
                                              "ncnn graph load returned failure");
            return false;
        }
    }

    for (;;)
    {
        LatentFrame stored;
        if (!condition_spool.read_next(stored, error))
        {
            if (!error.empty())
                return false;
            break;
        }
        const std::size_t absolute_index = frame_offset + frame_count;
        const ProfileScope frame_scope(profile, "dit-stack", absolute_index);
        ncnn::Mat condition_latent;
        if (!latent_frame_to_ncnn_mat(stored, condition_latent))
        {
            error = "frame=" + std::to_string(absolute_index) + " stage=handoff-latent latent spool record is invalid";
            return false;
        }
        ncnn::VkMat condition_gpu;
        ncnn::VkMat noise_gpu;
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_upload(condition_latent, condition_gpu, dit_opt);
            compute.record_upload(noise, noise_gpu, dit_opt);
            std::fprintf(stderr, "stage=noise-upload\n");
            if (compute.submit_and_wait() != 0)
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("noise-upload", context.diagnostics,
                                                  "Vulkan command submission returned failure");
                return false;
            }
        }

        ncnn::VkMat input_patches_gpu;
        std::fprintf(stderr, "stage=dit-input-patchify\n");
        if (!make_dit_input_patches_gpu(noise_gpu, condition_gpu, plan, context.vkdev,
                                        context.dit_blob_allocator, context.dit_staging_allocator, input_patches_gpu))
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("dit-input-patchify", context.diagnostics,
                                              "GPU patch assembly returned failure");
            return false;
        }

        ncnn::VkMat prediction_gpu;
        std::fprintf(stderr, "stage=dit-stack\n");
        if (!dit.run(input_patches_gpu, context.text, 1000.f, plan, prediction_gpu))
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("dit-stack", context.diagnostics,
                                              "GPU DiT execution returned failure");
            return false;
        }

        ncnn::VkMat noise_patches_gpu;
        std::fprintf(stderr, "stage=noise-patchify\n");
        if (!patch_latent_for_dit_output_gpu(noise_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                             context.dit_staging_allocator, noise_patches_gpu))
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("noise-patchify", context.diagnostics,
                                              "GPU patch assembly returned failure");
            return false;
        }

        ncnn::VkMat endpoint_patches_gpu;
        std::fprintf(stderr, "stage=v-lerp-endpoint\n");
        if (!apply_cfg_v_lerp_endpoint_vulkan(prediction_gpu, noise_patches_gpu, context.vkdev,
                                               context.dit_blob_allocator, context.dit_staging_allocator,
                                               endpoint_patches_gpu))
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("v-lerp-endpoint", context.diagnostics,
                                              "GPU sampler endpoint returned failure");
            return false;
        }

        ncnn::VkMat output_latent_gpu;
        std::fprintf(stderr, "stage=latent-unpatch\n");
        if (!unpatch_dit_output_gpu(endpoint_patches_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                    context.dit_staging_allocator, output_latent_gpu))
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("latent-unpatch", context.diagnostics,
                                              "GPU patch removal returned failure");
            return false;
        }

        ncnn::Mat output_latent;
        std::fprintf(stderr, "stage=handoff-latent\n");
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_download(output_latent_gpu, output_latent, dit_opt);
            if (compute.submit_and_wait() != 0 || output_latent.empty())
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("handoff-latent", context.diagnostics,
                                                  "latent download returned failure");
                return false;
            }
        }
        LatentFrame output_stored;
        if (!ncnn_mat_to_latent_frame(output_latent, output_stored) || !output_spool.append(output_stored, error))
        {
            if (error.empty())
                error = "frame=" + std::to_string(absolute_index) + " stage=handoff-latent latent spool append failed";
            return false;
        }
        frame_count++;
    }

    if (!output_spool.rewind(error))
        return false;
    context.dit_blob_allocator->clear();
    context.dit_staging_allocator->clear();
    return true;
}

bool decode_video_vulkan(LatentSpool& output_spool,
                         const ResolutionPlan& plan,
                         VulkanInferenceContext& context,
                         const ImageInferenceSession::VideoFrameWriter& writer,
                         RgbFrameSpool& reference_spool,
                         std::size_t expected_frames,
                         std::size_t& frame_count,
                         std::string& error,
                         const PerformanceProfile& profile,
                         std::size_t frame_offset)
{
    frame_count = 0;
    ncnn::Net decode;
    {
        const ProfileScope load_scope(profile, "load-decode");
        std::fprintf(stderr, "stage=load-decode\n");
        if (!load_vae(decode, context.graphs.vae_decode_stem, context.vkdev, context.decode_blob_allocator,
                      context.decode_staging_allocator.get(), true))
        {
            error = format_vulkan_stage_error("load-decode", context.diagnostics,
                                              "ncnn graph load returned failure");
            return false;
        }
    }

    double write_ms = 0.0;
    for (;;)
    {
        LatentFrame stored;
        if (!output_spool.read_next(stored, error))
        {
            if (!error.empty())
                return false;
            break;
        }
        const std::size_t absolute_index = frame_offset + frame_count;
        const ProfileScope frame_scope(profile, "vae-decode", absolute_index);
        ncnn::Mat output_latent;
        if (!latent_frame_to_ncnn_mat(stored, output_latent))
        {
            error = "frame=" + std::to_string(absolute_index) + " stage=handoff-latent latent spool record is invalid";
            return false;
        }
        ncnn::VkMat decode_latent_gpu;
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_upload(output_latent, decode_latent_gpu, decode.opt);
            if (compute.submit_and_wait() != 0)
            {
                error = "frame=" + std::to_string(absolute_index) + " " +
                        format_vulkan_stage_error("handoff-latent", context.diagnostics,
                                                  "latent upload returned failure");
                return false;
            }
        }

        ncnn::Mat reconstruction;
        std::fprintf(stderr, "stage=vae-decode\n");
        ncnn::Extractor extractor = decode.create_extractor();
        extractor.set_light_mode(false);
        if (extractor.input("in0", decode_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
        {
            error = "frame=" + std::to_string(absolute_index) + " " +
                    format_vulkan_stage_error("vae-decode", context.diagnostics,
                                              "ncnn extraction returned failure");
            return false;
        }
        RgbImage output;
        if (!reconstruction_to_rgb(reconstruction, plan, output))
        {
            error = "frame=" + std::to_string(absolute_index) + " stage=output-postprocess failed";
            return false;
        }
        RgbImage reference;
        if (!reference_spool.read_next(reference, error) || !apply_wavelet_color_fix(output, reference, output, error))
        {
            if (error.empty())
                error = "frame=" + std::to_string(absolute_index) + " stage=color-reconstruction failed";
            return false;
        }
        const auto write_start = PerformanceProfile::Clock::now();
        std::string write_error;
        const bool write_ok = writer(output, write_error);
        write_ms += profile.elapsed_ms(write_start);
        if (!write_ok)
        {
            profile.report("video-write", write_ms);
            error = "stage=video-encode frame=" + std::to_string(absolute_index) + ": " + write_error;
            return false;
        }
        frame_count++;
    }
    profile.report("video-write", write_ms);
    if (frame_count != expected_frames)
    {
        error = "stage=video-inference: staged frame count changed during processing";
        return false;
    }
    context.decode_blob_allocator->clear();
    context.decode_staging_allocator->clear();
    return true;
}

bool run_vulkan_image_video(const ImageInferenceSession::VideoFrameReader& reader,
                            const ImageInferenceSession::VideoFrameWriter& writer,
                            const ResolutionPlan& plan,
                            VulkanInferenceContext& context,
                            std::size_t& frame_count,
                            std::string& error,
                            const PerformanceProfile& profile,
                            std::size_t frame_offset)
{
    struct FrameCleanup final
    {
        VulkanInferenceContext& context;
        ~FrameCleanup() { clear_frame_staging_allocators(context); }
    } cleanup{context};

    frame_count = 0;
    const auto batch_start = PerformanceProfile::Clock::now();
    LatentSpool condition_spool;
    LatentSpool output_spool;
    RgbFrameSpool reference_spool;
    if (!RgbFrameSpool::create(reference_spool, error))
        return false;
    std::size_t encoded_frames = 0;
    if (!encode_video_vulkan(reader, plan, context, condition_spool, reference_spool, encoded_frames, error, profile,
                             frame_offset))
    {
        profile.report_batch("video-batch", encoded_frames, profile.elapsed_ms(batch_start));
        return false;
    }
    std::size_t denoised_frames = 0;
    if (!denoise_video_vulkan(condition_spool, plan, context, output_spool, denoised_frames, error, profile,
                              frame_offset))
    {
        frame_count = denoised_frames;
        profile.report_batch("video-batch", encoded_frames, profile.elapsed_ms(batch_start));
        return false;
    }
    if (denoised_frames != encoded_frames)
    {
        error = "stage=video-inference: staged frame count changed after DiT";
        profile.report_batch("video-batch", encoded_frames, profile.elapsed_ms(batch_start));
        return false;
    }
    if (!reference_spool.rewind(error))
    {
        profile.report_batch("video-batch", encoded_frames, profile.elapsed_ms(batch_start));
        return false;
    }
    if (!decode_video_vulkan(output_spool, plan, context, writer, reference_spool, denoised_frames, frame_count,
                             error, profile, frame_offset))
    {
        profile.report_batch("video-batch", encoded_frames, profile.elapsed_ms(batch_start));
        return false;
    }
    profile.report_batch("video-batch", frame_count, profile.elapsed_ms(batch_start));
    return true;
}

bool run_vulkan_image_batch(const std::vector<RgbImage>& inputs,
                            const ResolutionPlan& plan,
                            VulkanInferenceContext& context,
                            std::vector<RgbImage>& outputs,
                            std::string& error,
                            const PerformanceProfile& profile,
                            std::size_t frame_offset)
{
    struct FrameCleanup final
    {
        VulkanInferenceContext& context;
        ~FrameCleanup() { clear_frame_staging_allocators(context); }
    } cleanup{context};

    std::vector<ncnn::Mat> condition_latents;
    std::vector<ncnn::Mat> output_latents;
    if (!encode_batch_vulkan(inputs, plan, context, condition_latents, error, profile, frame_offset) ||
        !denoise_batch_vulkan(condition_latents, plan, context, output_latents, error, profile, frame_offset) ||
        !decode_batch_vulkan(output_latents, plan, context, outputs, error, profile, frame_offset))
    {
        outputs.clear();
        return false;
    }
    for (std::size_t index = 0; index < outputs.size(); index++)
    {
        RgbImage reference;
        if (!prepare_color_reference(inputs[index], plan, reference, error) ||
            !apply_wavelet_color_fix(outputs[index], reference, outputs[index], error))
        {
            outputs.clear();
            return false;
        }
    }
    return true;
}

#endif

} // namespace

struct ImageInferenceSession::Impl
{
#if NCNN_VULKAN
    VulkanInferenceContext context;
#endif
    // Borrowed from the caller; null when profiling is disabled.
    const PerformanceProfile* profile = nullptr;
};

ImageInferenceSession::ImageInferenceSession() = default;

ImageInferenceSession::~ImageInferenceSession() = default;

ImageInferenceSession::ImageInferenceSession(ImageInferenceSession&&) noexcept = default;

ImageInferenceSession& ImageInferenceSession::operator=(ImageInferenceSession&&) noexcept = default;

bool ImageInferenceSession::open(const ModelGraphSet& graphs,
                                 const ResolutionPlan& plan,
                                 int gpu_id,
                                 ImageInferenceSession& session,
                                 std::string& error,
                                 std::uint32_t memory_budget_mib,
                                 const PerformanceProfile* profile)
{
    error.clear();
#if NCNN_VULKAN
    if (plan.image_width <= 0 || plan.image_height <= 0 || plan.latent_width <= 0 || plan.latent_height <= 0)
    {
        error = "input image or resolution plan is invalid";
        return false;
    }
    static const PerformanceProfile kDisabledProfile;
    std::unique_ptr<Impl> candidate(new Impl);
    candidate->profile = profile ? profile : &kDisabledProfile;
    if (!initialize_vulkan_context(graphs, plan, gpu_id, memory_budget_mib, candidate->context, error,
                                  *candidate->profile))
        return false;
    session.impl_ = std::move(candidate);
    return true;
#else
    (void)graphs;
    (void)plan;
    (void)gpu_id;
    (void)memory_budget_mib;
    (void)profile;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

bool ImageInferenceSession::run_frame(const RgbImage& input, RgbImage& output, std::string& error) const
{
    output = RgbImage();
    error.clear();
    std::vector<RgbImage> inputs;
    inputs.push_back(input);
    std::vector<RgbImage> outputs;
    if (!run_batch(inputs, outputs, error))
        return false;
    output = std::move(outputs.front());
    return true;
}

bool ImageInferenceSession::run_batch(const std::vector<RgbImage>& inputs,
                                      std::vector<RgbImage>& outputs,
                                      std::string& error,
                                      std::size_t frame_offset) const
{
    outputs.clear();
    error.clear();
    if (inputs.empty() || inputs.size() > kMaxBatchFrames)
    {
        error = "inference batch must contain 1 or 2 frames";
        return false;
    }

#if NCNN_VULKAN
    if (!impl_)
    {
        error = "inference session is not open";
        return false;
    }
    for (const RgbImage& input : inputs)
    {
        if (!valid_rgb_image(input))
        {
            error = "input image or resolution plan is invalid";
            return false;
        }
    }
    return run_vulkan_image_batch(inputs, impl_->context.plan, impl_->context, outputs, error,
                                  *impl_->profile, frame_offset);
#else
    (void)inputs;
    (void)frame_offset;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

bool ImageInferenceSession::run_video(const VideoFrameReader& reader,
                                      const VideoFrameWriter& writer,
                                      std::size_t& frame_count,
                                      std::string& error,
                                      std::size_t frame_offset) const
{
    frame_count = 0;
    error.clear();
    if (!reader || !writer)
    {
        error = "video reader and writer callbacks are required";
        return false;
    }
#if NCNN_VULKAN
    if (!impl_)
    {
        error = "inference session is not open";
        return false;
    }
    return run_vulkan_image_video(reader, writer, impl_->context.plan, impl_->context, frame_count, error,
                                  *impl_->profile, frame_offset);
#else
    (void)frame_offset;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

bool run_image_inference(const ModelGraphSet& graphs,
                         const RgbImage& input,
                         const ResolutionPlan& plan,
                         int gpu_id,
                         RgbImage& output,
                         std::string& error,
                         std::uint32_t memory_budget_mib,
                         const PerformanceProfile* profile)
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
    ImageInferenceSession session;
    if (!ImageInferenceSession::open(graphs, plan, gpu_id, session, error, memory_budget_mib, profile))
        return false;
    return session.run_frame(input, output, error);
#else
    (void)graphs;
    (void)input;
    (void)plan;
    (void)gpu_id;
    (void)memory_budget_mib;
    (void)profile;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

} // namespace seedvr2
