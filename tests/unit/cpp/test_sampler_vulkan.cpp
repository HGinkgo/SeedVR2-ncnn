#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "command.h"
#include "gpu.h"
#include "sampler/sampler.h"
#include "sampler/vulkan_sampler.h"

namespace
{

bool download(const ncnn::VkMat& source, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, ncnn::Mat& result)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(source, result, opt);
    return compute.submit_and_wait() == 0;
}

float max_error(const ncnn::Mat& actual, const std::vector<float>& expected)
{
    if (actual.empty() || actual.total() != expected.size())
        return INFINITY;
    const float* data = static_cast<const float*>(actual.data);
    float error = 0.f;
    for (size_t index = 0; index < expected.size(); index++)
        error = std::max(error, std::fabs(data[index] - expected[index]));
    return error;
}

void print_vector(const char* name, const std::vector<float>& values)
{
    std::fprintf(stderr, "%s:", name);
    for (const float value : values)
        std::fprintf(stderr, " %g", value);
    std::fprintf(stderr, "\n");
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
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    ncnn::Mat positive(4, 4);
    ncnn::Mat negative(4, 4);
    ncnn::Mat sample(4, 4);
    std::vector<float> positive_values(positive.total());
    std::vector<float> negative_values(negative.total());
    std::vector<float> sample_values(sample.total());
    for (size_t index = 0; index < sample.total(); index++)
    {
        positive_values[index] = 0.1f * static_cast<float>(index + 1);
        negative_values[index] = -0.05f * static_cast<float>(index % 7);
        sample_values[index] = 0.01f * static_cast<float>((index * 3) % 11) - 0.05f;
        positive[index] = positive_values[index];
        negative[index] = negative_values[index];
        sample[index] = sample_values[index];
    }

    ncnn::VkMat positive_packed;
    ncnn::VkMat negative_packed;
    ncnn::VkMat positive_gpu;
    ncnn::VkMat negative_gpu;
    ncnn::VkMat sample_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(positive, positive_packed, opt);
        compute.record_upload(negative, negative_packed, opt);
        compute.record_upload(sample, sample_gpu, opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }
    if (positive_packed.elempack != 4 || negative_packed.elempack != 4 || sample_gpu.elempack != 4)
        return 1;
    {
        ncnn::VkCompute unpack(vkdev);
        vkdev->convert_packing(positive_packed, positive_gpu, 1, unpack, opt);
        vkdev->convert_packing(negative_packed, negative_gpu, 1, unpack, opt);
        if (unpack.submit_and_wait() != 0 || positive_gpu.elempack != 1 || negative_gpu.elempack != 1)
            return 1;
    }
    ncnn::VkMat updated_gpu;
    if (!seedvr2::apply_cfg_euler_vulkan(positive_gpu, negative_gpu, sample_gpu, 7.5f, -0.5f, vkdev,
                                         blob_allocator, staging_allocator, updated_gpu))
        return 1;

    ncnn::Mat updated;
    if (!download(updated_gpu, vkdev, opt, updated))
        return 1;
    std::vector<float> guided;
    std::vector<float> expected;
    if (!seedvr2::classifier_free_guidance(positive_values, negative_values, 7.5f, 0.f, guided) ||
        !seedvr2::euler_v_lerp_step(sample_values, guided, 1000.f, 500.f, 1000.f, expected))
        return 1;
    const float error = max_error(updated, expected);
    if (!std::isfinite(error) || error > 1.e-5f)
    {
        std::fprintf(stderr, "Vulkan sampler mismatch max_abs_error=%g\n", error);
        print_vector("expected", expected);
        return 1;
    }

    ncnn::VkMat endpoint_gpu;
    if (!seedvr2::apply_cfg_v_lerp_endpoint_vulkan(positive_gpu, sample_gpu, vkdev, blob_allocator,
                                                    staging_allocator, endpoint_gpu))
        return 1;
    ncnn::Mat endpoint;
    if (!download(endpoint_gpu, vkdev, opt, endpoint))
        return 1;
    std::vector<float> endpoint_expected;
    if (!seedvr2::euler_v_lerp_endpoint(sample_values, positive_values, 1000.f, 1000.f, endpoint_expected))
        return 1;
    const float endpoint_error = max_error(endpoint, endpoint_expected);
    if (!std::isfinite(endpoint_error) || endpoint_error > 1.e-5f)
    {
        std::fprintf(stderr, "Vulkan endpoint mismatch max_abs_error=%g\n", endpoint_error);
        return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    std::printf("seedvr2-sampler-vulkan: ok max_abs_error=%g endpoint_error=%g\n", error, endpoint_error);
    return 0;
}
