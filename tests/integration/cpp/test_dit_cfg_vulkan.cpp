#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "conditioning/conditioning.h"
#include "command.h"
#include "dit/dit_stack.h"
#include "gpu.h"
#include "sampler/sampler.h"
#include "sampler/vulkan_sampler.h"

namespace
{

constexpr int kLatentChannels = 16;
constexpr int kLatentSize = 16;
constexpr int kOutputWidth = 64;
constexpr int kOutputHeight = 64;

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

bool download(const ncnn::VkMat& source, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, ncnn::Mat& result)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(source, result, opt);
    return compute.submit_and_wait() == 0;
}

bool to_vector(const ncnn::Mat& value, std::vector<float>& result)
{
    if (!finite(value) || value.dims != 2 || value.w != kOutputWidth || value.h != kOutputHeight ||
        value.elemsize != 4u)
        return false;
    const float* data = static_cast<const float*>(value.data);
    result.assign(data, data + value.total());
    return true;
}

float max_error(const std::vector<float>& actual, const std::vector<float>& expected)
{
    if (actual.size() != expected.size())
        return std::numeric_limits<float>::infinity();
    float error = 0.f;
    for (size_t index = 0; index < actual.size(); index++)
        error = std::max(error, std::fabs(actual[index] - expected[index]));
    return error;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::fprintf(stderr,
                     "usage: test_dit_cfg_vulkan <positive-stack-dir> <negative-stack-dir> "
                     "<positive-condition-f32> <negative-condition-f32>\n");
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device available\n");
        return 1;
    }
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Mat positive_condition;
    ncnn::Mat negative_condition;
    if (!seedvr2::load_conditioning_f32(argv[3], 58, positive_condition) ||
        !seedvr2::load_conditioning_f32(argv[4], 64, negative_condition))
    {
        std::fprintf(stderr, "stage=conditioning-load failed\n");
        return 1;
    }

    ncnn::Mat latent(kLatentSize, kLatentSize, 1, kLatentChannels);
    latent.fill(0.f);
    seedvr2::ResolutionPlan square;
    if (!seedvr2::ResolutionPlan::from_explicit(128, 128, square))
    {
        std::fprintf(stderr, "stage=square-plan failed\n");
        return 1;
    }
    ncnn::VkMat positive_output_gpu;
    ncnn::VkMat negative_output_gpu;
    if (!seedvr2::run_dit_stack_gpu(latent, positive_condition, 500.f, argv[1], square, vkdev, blob_allocator,
                                    staging_allocator, positive_output_gpu))
    {
        std::fprintf(stderr, "stage=positive-dit-stack failed\n");
        return 1;
    }
    if (!seedvr2::run_dit_stack_gpu(latent, negative_condition, 500.f, argv[2], square, vkdev, blob_allocator,
                                     staging_allocator, negative_output_gpu))
    {
        std::fprintf(stderr, "stage=negative-dit-stack failed\n");
        return 1;
    }

    ncnn::Mat sample(kOutputWidth, kOutputHeight);
    for (size_t index = 0; index < sample.total(); index++)
        sample[index] = 0.01f * static_cast<float>(index % 17);
    ncnn::VkMat sample_gpu;
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
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(sample, sample_gpu, opt);
        if (compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "stage=sample-upload failed\n");
            return 1;
        }
    }

    ncnn::VkMat updated_gpu;
    if (!seedvr2::apply_cfg_euler_vulkan(positive_output_gpu, negative_output_gpu, sample_gpu, 7.5f, -0.5f,
                                         vkdev, blob_allocator, staging_allocator, updated_gpu))
    {
        std::fprintf(stderr,
                     "stage=cfg-euler failed positive=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d,elemsize=%zu) "
                     "negative=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d,elemsize=%zu) "
                     "sample=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d,elemsize=%zu)\n",
                     positive_output_gpu.dims, positive_output_gpu.w, positive_output_gpu.h, positive_output_gpu.d,
                     positive_output_gpu.c, positive_output_gpu.elempack, positive_output_gpu.elemsize,
                     negative_output_gpu.dims, negative_output_gpu.w, negative_output_gpu.h, negative_output_gpu.d,
                     negative_output_gpu.c, negative_output_gpu.elempack, negative_output_gpu.elemsize,
                     sample_gpu.dims, sample_gpu.w, sample_gpu.h, sample_gpu.d, sample_gpu.c, sample_gpu.elempack,
                     sample_gpu.elemsize);
        return 1;
    }

    ncnn::Option download_opt;
    download_opt.use_vulkan_compute = true;
    download_opt.use_packing_layout = false;
    download_opt.use_fp16_packed = false;
    download_opt.use_fp16_storage = false;
    download_opt.use_fp16_arithmetic = false;
    download_opt.blob_vkallocator = blob_allocator;
    download_opt.workspace_vkallocator = blob_allocator;
    download_opt.staging_vkallocator = staging_allocator;
    ncnn::Mat positive_output;
    ncnn::Mat negative_output;
    ncnn::Mat updated;
    if (!download(positive_output_gpu, vkdev, download_opt, positive_output) ||
        !download(negative_output_gpu, vkdev, download_opt, negative_output) ||
        !download(updated_gpu, vkdev, download_opt, updated))
    {
        std::fprintf(stderr, "stage=download failed\n");
        return 1;
    }

    std::vector<float> positive_values;
    std::vector<float> negative_values;
    std::vector<float> sample_values;
    std::vector<float> updated_values;
    if (!to_vector(positive_output, positive_values) || !to_vector(negative_output, negative_values) ||
        !to_vector(sample, sample_values) || !to_vector(updated, updated_values))
    {
        std::fprintf(stderr, "stage=tensor-validation failed\n");
        return 1;
    }

    std::vector<float> guided;
    std::vector<float> expected;
    if (!seedvr2::classifier_free_guidance(positive_values, negative_values, 7.5f, 0.f, guided) ||
        !seedvr2::euler_v_lerp_step(sample_values, guided, 1000.f, 500.f, 1000.f, expected))
    {
        std::fprintf(stderr, "stage=cpu-reference failed\n");
        return 1;
    }
    const float error = max_error(updated_values, expected);
    if (!std::isfinite(error) || error > 1.e-4f)
    {
        std::fprintf(stderr, "CFG/Euler Vulkan mismatch max_abs_error=%g\n", error);
        return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    std::printf("seedvr2-dit-cfg-vulkan: ok max_abs_error=%g\n", error);
    return 0;
}
