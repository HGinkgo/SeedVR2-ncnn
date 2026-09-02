#include <cmath>
#include <cstdio>

#include "command.h"
#include "dit/dit_stack.h"
#include "gpu.h"

namespace
{

constexpr int kLatentChannels = 16;
constexpr int kLatentSize = 16;
constexpr int kPatchSize = 2;
constexpr int kTokenWidth = 8;

float noise_value(int channel, int row, int column)
{
    return static_cast<float>(channel * 10000 + row * 100 + column);
}

float condition_value(int channel, int row, int column)
{
    return -noise_value(channel, row, column) - 0.5f;
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

bool upload(const ncnn::Mat& source, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, ncnn::VkMat& target)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_upload(source, target, opt);
    return compute.submit_and_wait() == 0;
}

bool download(const ncnn::VkMat& source, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, ncnn::Mat& target)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(source, target, opt);
    return compute.submit_and_wait() == 0;
}

bool near(float actual, float expected)
{
    return std::fabs(actual - expected) <= 1.e-5f;
}

} // namespace

int main()
{
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
    seedvr2::ResolutionPlan square;
    if (!seedvr2::ResolutionPlan::from_explicit(128, 128, square))
    {
        std::fprintf(stderr, "stage=square-plan failed\n");
        return 1;
    }

    ncnn::Mat noise(kLatentSize, kLatentSize, 1, kLatentChannels);
    ncnn::Mat condition(kLatentSize, kLatentSize, 1, kLatentChannels);
    for (int channel = 0; channel < kLatentChannels; channel++)
        for (int row = 0; row < kLatentSize; row++)
            for (int column = 0; column < kLatentSize; column++)
            {
                noise.channel(channel).row(row)[column] = noise_value(channel, row, column);
                condition.channel(channel).row(row)[column] = condition_value(channel, row, column);
            }

    ncnn::VkMat noise_gpu;
    ncnn::VkMat condition_gpu;
    if (!upload(noise, vkdev, opt, noise_gpu) || !upload(condition, vkdev, opt, condition_gpu))
    {
        std::fprintf(stderr, "stage=upload failed\n");
        return 1;
    }

    ncnn::VkMat input_patches_gpu;
    if (!seedvr2::make_dit_input_patches_gpu(noise_gpu, condition_gpu, square, vkdev, blob_allocator,
                                              staging_allocator, input_patches_gpu))
    {
        std::fprintf(stderr, "stage=input-patchify failed\n");
        return 1;
    }
    ncnn::Mat input_patches;
    if (!download(input_patches_gpu, vkdev, opt, input_patches) || input_patches.dims != 2 ||
        input_patches.w != 132 || input_patches.h != 64 || input_patches.elempack != 1)
    {
        std::fprintf(stderr, "stage=input-patch-download failed\n");
        return 1;
    }
    for (int patch_row = 0; patch_row < kTokenWidth; patch_row++)
        for (int patch_column = 0; patch_column < kTokenWidth; patch_column++)
        {
            const float* patch = input_patches.row(patch_row * kTokenWidth + patch_column);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < 33; channel++)
                    {
                        const int feature = (dy * kPatchSize + dx) * 33 + channel;
                        const int row = patch_row * kPatchSize + dy;
                        const int column = patch_column * kPatchSize + dx;
                        const float expected = channel < kLatentChannels
                                                   ? noise_value(channel, row, column)
                                                   : channel < 2 * kLatentChannels
                                                         ? condition_value(channel - kLatentChannels, row, column)
                                                         : 1.f;
                        if (!near(patch[feature], expected))
                        {
                            std::fprintf(stderr, "input patch mismatch token=%d feature=%d actual=%g expected=%g\n",
                                         patch_row * kTokenWidth + patch_column, feature, patch[feature], expected);
                            return 1;
                        }
                    }
        }

    ncnn::VkMat output_patches_gpu;
    if (!seedvr2::patch_latent_for_dit_output_gpu(noise_gpu, square, vkdev, blob_allocator, staging_allocator,
                                                   output_patches_gpu))
    {
        std::fprintf(stderr, "stage=output-patchify failed\n");
        return 1;
    }
    ncnn::VkMat restored_gpu;
    if (!seedvr2::unpatch_dit_output_gpu(output_patches_gpu, square, vkdev, blob_allocator, staging_allocator,
                                          restored_gpu))
    {
        std::fprintf(stderr, "stage=unpatch failed\n");
        return 1;
    }
    ncnn::Mat restored;
    if (!download(restored_gpu, vkdev, opt, restored) || restored.dims != 3 || restored.w != kLatentSize ||
        restored.h != kLatentSize || restored.c != kLatentChannels || restored.elempack != 1)
    {
        std::fprintf(stderr, "stage=restored-download failed\n");
        return 1;
    }
    for (int channel = 0; channel < kLatentChannels; channel++)
        for (int row = 0; row < kLatentSize; row++)
            for (int column = 0; column < kLatentSize; column++)
                if (!near(restored.channel(channel).row(row)[column], noise_value(channel, row, column)))
                {
                    std::fprintf(stderr, "output patch mismatch channel=%d row=%d column=%d\n", channel, row,
                                 column);
                    return 1;
                }

    seedvr2::ResolutionPlan widescreen;
    if (!seedvr2::ResolutionPlan::from_explicit(720, 1280, widescreen))
    {
        std::fprintf(stderr, "stage=dynamic-plan failed\n");
        return 1;
    }
    ncnn::Mat dynamic_noise(widescreen.latent_width, widescreen.latent_height, 1, kLatentChannels);
    ncnn::Mat dynamic_condition(widescreen.latent_width, widescreen.latent_height, 1, kLatentChannels);
    for (int channel = 0; channel < kLatentChannels; channel++)
        for (int row = 0; row < widescreen.latent_height; row++)
            for (int column = 0; column < widescreen.latent_width; column++)
            {
                dynamic_noise.channel(channel).row(row)[column] = noise_value(channel, row, column);
                dynamic_condition.channel(channel).row(row)[column] = condition_value(channel, row, column);
            }
    ncnn::VkMat dynamic_noise_gpu;
    ncnn::VkMat dynamic_condition_gpu;
    if (!upload(dynamic_noise, vkdev, opt, dynamic_noise_gpu) ||
        !upload(dynamic_condition, vkdev, opt, dynamic_condition_gpu))
    {
        std::fprintf(stderr, "stage=dynamic-upload failed\n");
        return 1;
    }
    ncnn::VkMat dynamic_input_patches_gpu;
    if (!seedvr2::make_dit_input_patches_gpu(dynamic_noise_gpu, dynamic_condition_gpu, widescreen, vkdev,
                                              blob_allocator, staging_allocator, dynamic_input_patches_gpu))
    {
        std::fprintf(stderr, "stage=dynamic-input-patchify failed\n");
        return 1;
    }
    ncnn::Mat dynamic_input_patches;
    if (!download(dynamic_input_patches_gpu, vkdev, opt, dynamic_input_patches) ||
        dynamic_input_patches.dims != 2 || dynamic_input_patches.w != 132 ||
        dynamic_input_patches.h != widescreen.video_tokens)
    {
        std::fprintf(stderr, "stage=dynamic-input-shape failed\n");
        return 1;
    }
    ncnn::VkMat dynamic_output_patches_gpu;
    ncnn::VkMat dynamic_restored_gpu;
    if (!seedvr2::patch_latent_for_dit_output_gpu(dynamic_noise_gpu, widescreen, vkdev, blob_allocator,
                                                   staging_allocator, dynamic_output_patches_gpu) ||
        !seedvr2::unpatch_dit_output_gpu(dynamic_output_patches_gpu, widescreen, vkdev, blob_allocator,
                                           staging_allocator, dynamic_restored_gpu))
    {
        std::fprintf(stderr, "stage=dynamic-unpatch failed\n");
        return 1;
    }
    ncnn::Mat dynamic_restored;
    if (!download(dynamic_restored_gpu, vkdev, opt, dynamic_restored) || dynamic_restored.dims != 3 ||
        dynamic_restored.w != widescreen.latent_width || dynamic_restored.h != widescreen.latent_height ||
        dynamic_restored.c != kLatentChannels)
    {
        std::fprintf(stderr, "stage=dynamic-restored-shape failed\n");
        return 1;
    }
    if (!near(dynamic_restored.channel(7).row(31)[47], dynamic_noise.channel(7).row(31)[47]))
    {
        std::fprintf(stderr, "stage=dynamic-restored-value failed\n");
        return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    std::puts("seedvr2-dit-patch-vulkan: ok");
    return 0;
}
