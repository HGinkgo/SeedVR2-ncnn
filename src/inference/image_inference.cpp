#include "inference/image_inference.h"
#include "inference/memory_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

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
                               std::string& error)
{
    if (gpu_id < -1)
    {
        error = "Vulkan GPU id must be greater than or equal to -1";
        return false;
    }

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

bool run_vulkan_image_frame(const RgbImage& input,
                            const ResolutionPlan& plan,
                            VulkanInferenceContext& context,
                            RgbImage& output,
                            std::string& error)
{
    const auto stage_error = [&](const char* stage, const char* detail) {
        error = format_vulkan_stage_error(stage, context.diagnostics, detail);
    };
    struct FrameCleanup final
    {
        VulkanInferenceContext& context;
        ~FrameCleanup() { clear_frame_staging_allocators(context); }
    } cleanup{context};

    ncnn::Mat sample;
    if (!prepare_input(input, plan, sample))
    {
        error = "failed to prepare the input image";
        return false;
    }

    const ncnn::Option dit_opt = make_vulkan_option(context.dit_blob_allocator, context.dit_staging_allocator);
    ncnn::VkMat decode_latent_gpu;
    {
        ncnn::Net encode;
        std::fprintf(stderr, "stage=load-encode\n");
        if (!load_vae(encode, context.graphs.vae_encode_stem, context.vkdev, context.encode_blob_allocator,
                      context.encode_staging_allocator, false))
        {
            stage_error("load-encode", "ncnn graph load returned failure");
            return false;
        }
        const ncnn::Option encode_opt = encode.opt;

        ncnn::VkMat sample_gpu;
        {
            ncnn::VkCompute compute(context.vkdev);
            compute.record_upload(sample, sample_gpu, encode_opt);
            if (compute.submit_and_wait() != 0)
            {
                stage_error("input-upload", "Vulkan command submission returned failure");
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
                stage_error("vae-encode", "ncnn extraction or Vulkan command submission returned failure");
                return false;
            }
        }
        sample_gpu.release();

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
            ncnn::VkCompute compute(context.vkdev);
            compute.record_upload(noise, noise_gpu, dit_opt);
            if (compute.submit_and_wait() != 0)
            {
                stage_error("noise-upload", "Vulkan command submission returned failure");
                return false;
            }
        }

        ncnn::VkMat input_patches_gpu;
        std::fprintf(stderr, "stage=dit-input-patchify\n");
        if (!make_dit_input_patches_gpu(noise_gpu, latent_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                        context.dit_staging_allocator, input_patches_gpu))
        {
            stage_error("dit-input-patchify", "GPU patch assembly returned failure");
            return false;
        }

        DitStackSession dit;
        std::fprintf(stderr, "stage=load-dit-stack\n");
        if (!DitStackSession::open(context.graphs.dit_stack_dir.string(), plan, context.vkdev,
                                   context.dit_blob_allocator, context.dit_staging_allocator, dit))
        {
            stage_error("load-dit-stack", "ncnn graph load returned failure");
            return false;
        }

        ncnn::VkMat prediction_gpu;
        std::fprintf(stderr, "stage=dit-stack\n");
        if (!dit.run(input_patches_gpu, context.text, 1000.f, plan, prediction_gpu))
        {
            stage_error("dit-stack", "GPU DiT execution returned failure");
            return false;
        }

        ncnn::VkMat noise_patches_gpu;
        std::fprintf(stderr, "stage=noise-patchify\n");
        if (!patch_latent_for_dit_output_gpu(noise_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                             context.dit_staging_allocator, noise_patches_gpu))
        {
            stage_error("noise-patchify", "GPU patch assembly returned failure");
            return false;
        }

        ncnn::VkMat endpoint_patches_gpu;
        std::fprintf(stderr, "stage=v-lerp-endpoint\n");
        if (!apply_cfg_v_lerp_endpoint_vulkan(prediction_gpu, noise_patches_gpu, context.vkdev,
                                              context.dit_blob_allocator, context.dit_staging_allocator,
                                              endpoint_patches_gpu))
        {
            stage_error("v-lerp-endpoint", "GPU sampler endpoint returned failure");
            return false;
        }

        ncnn::VkMat output_latent_gpu;
        std::fprintf(stderr, "stage=latent-unpatch\n");
        if (!unpatch_dit_output_gpu(endpoint_patches_gpu, plan, context.vkdev, context.dit_blob_allocator,
                                    context.dit_staging_allocator, output_latent_gpu))
        {
            stage_error("latent-unpatch", "GPU patch removal returned failure");
            return false;
        }

        std::fprintf(stderr, "stage=handoff-latent\n");
        if (!clone_to_allocator(output_latent_gpu, decode_latent_gpu, context.vkdev, context.decode_blob_allocator,
                                context.decode_staging_allocator.get()))
        {
            stage_error("handoff-latent", "GPU latent handoff returned failure");
            return false;
        }
    }
    context.encode_blob_allocator->clear();
    context.encode_staging_allocator->clear();
    context.dit_blob_allocator->clear();
    context.dit_staging_allocator->clear();

    ncnn::Mat reconstruction;
    {
        ncnn::Net decode;
        std::fprintf(stderr, "stage=load-decode\n");
        if (!load_vae(decode, context.graphs.vae_decode_stem, context.vkdev, context.decode_blob_allocator,
                      context.decode_staging_allocator.get(), true))
        {
            stage_error("load-decode", "ncnn graph load returned failure");
            return false;
        }
        std::fprintf(stderr, "stage=vae-decode\n");
        ncnn::Extractor extractor = decode.create_extractor();
        extractor.set_light_mode(false);
        if (extractor.input("in0", decode_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
        {
            stage_error("vae-decode", "ncnn extraction returned failure");
            return false;
        }
    }
    if (!reconstruction_to_rgb(reconstruction, plan, output))
    {
        error = "stage=output-postprocess failed";
        return false;
    }

    decode_latent_gpu.release();
    return true;
}

#endif

} // namespace

struct ImageInferenceSession::Impl
{
#if NCNN_VULKAN
    VulkanInferenceContext context;
#endif
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
                                 std::uint32_t memory_budget_mib)
{
    error.clear();
#if NCNN_VULKAN
    if (plan.image_width <= 0 || plan.image_height <= 0 || plan.latent_width <= 0 || plan.latent_height <= 0)
    {
        error = "input image or resolution plan is invalid";
        return false;
    }
    std::unique_ptr<Impl> candidate(new Impl);
    if (!initialize_vulkan_context(graphs, plan, gpu_id, memory_budget_mib, candidate->context, error))
        return false;
    session.impl_ = std::move(candidate);
    return true;
#else
    (void)graphs;
    (void)plan;
    (void)gpu_id;
    (void)memory_budget_mib;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

bool ImageInferenceSession::run_frame(const RgbImage& input, RgbImage& output, std::string& error) const
{
    output = RgbImage();
    error.clear();
#if NCNN_VULKAN
    if (!impl_)
    {
        error = "inference session is not open";
        return false;
    }
    if (!valid_rgb_image(input))
    {
        error = "input image or resolution plan is invalid";
        return false;
    }
    return run_vulkan_image_frame(input, impl_->context.plan, impl_->context, output, error);
#else
    (void)input;
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
                         std::uint32_t memory_budget_mib)
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
    if (!ImageInferenceSession::open(graphs, plan, gpu_id, session, error, memory_budget_mib))
        return false;
    return session.run_frame(input, output, error);
#else
    (void)graphs;
    (void)input;
    (void)plan;
    (void)gpu_id;
    (void)memory_budget_mib;
    error = "image inference requires a Vulkan-enabled build";
    return false;
#endif
}

} // namespace seedvr2
