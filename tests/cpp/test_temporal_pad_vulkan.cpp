#include <cmath>
#include <cstdio>

#include "command.h"
#include "gpu.h"
#include "layer/convolution3d.h"
#include "vae/causal_conv3d.h"
#include "vae/temporal_pad.h"

namespace
{

bool matches_expected(const ncnn::Mat& output)
{
    if (output.dims != 4 || output.w != 3 || output.h != 2 || output.d != 4 || output.c != 2)
        return false;

    for (int channel = 0; channel < output.c; channel++)
        for (int depth = 0; depth < output.d; depth++)
            for (int row = 0; row < output.h; row++)
                for (int column = 0; column < output.w; column++)
                {
                    const int source_depth = depth < 2 ? 0 : depth - 2;
                    const float expected = static_cast<float>(100 * channel + 10 * source_depth + 3 * row + column);
                    const float actual = output.channel(channel).depth(depth).row(row)[column];
                    if (std::fabs(actual - expected) > 1e-6f)
                    {
                        std::fprintf(stderr, "mismatch c=%d d=%d y=%d x=%d actual=%g expected=%g\n",
                                     channel, depth, row, column, actual, expected);
                        return false;
                    }
                }
    return true;
}

bool causal_conv3d_matches_cpu_reference(ncnn::VulkanDevice* vkdev)
{
    ncnn::Mat input(2, 2, 2, 1);
    for (int depth = 0; depth < input.d; depth++)
        for (int row = 0; row < input.h; row++)
            for (int column = 0; column < input.w; column++)
                input.depth(depth).row(row)[column] = static_cast<float>(100 * depth + 10 * row + column);

    ncnn::ParamDict convolution_params;
    convolution_params.set(0, 2);
    convolution_params.set(1, 3);
    convolution_params.set(11, 3);
    convolution_params.set(21, 3);
    convolution_params.set(2, 1);
    convolution_params.set(12, 1);
    convolution_params.set(22, 1);
    convolution_params.set(3, 1);
    convolution_params.set(13, 1);
    convolution_params.set(23, 1);
    convolution_params.set(4, 1);
    convolution_params.set(14, 1);
    convolution_params.set(24, 0);
    convolution_params.set(5, 1);
    convolution_params.set(6, 54);
    convolution_params.set(9, 0);

    ncnn::Mat weights(54);
    for (int index = 0; index < 54; index++)
        weights[index] = static_cast<float>((index % 5) - 2) * 0.125f;
    ncnn::Mat bias(2);
    bias[0] = 0.25f;
    bias[1] = -0.5f;

    ncnn::Convolution3D reference_convolution;
    if (reference_convolution.load_param(convolution_params) != 0)
        return false;
    reference_convolution.weight_data = weights;
    reference_convolution.bias_data = bias;

    ncnn::ParamDict pad_params;
    pad_params.set(0, 2);
    SeedVR2TemporalPad temporal_pad;
    if (temporal_pad.load_param(pad_params) != 0)
        return false;

    ncnn::Option cpu_options;
    ncnn::Mat padded;
    ncnn::Mat expected;
    if (temporal_pad.forward(input, padded, cpu_options) != 0 ||
        reference_convolution.forward(padded, expected, cpu_options) != 0)
        return false;

    ncnn::ParamDict causal_params = convolution_params;
    causal_params.set(31, 2);
    SeedVR2CausalConv3D causal_convolution;
    if (causal_convolution.load_param(causal_params) != 0)
        return false;
    causal_convolution.weight_data = weights;
    causal_convolution.bias_data = bias;
    causal_convolution.vkdev = vkdev;

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

    ncnn::VkTransfer transfer(vkdev);
    if (causal_convolution.upload_model(transfer, opt) != 0 || transfer.submit_and_wait() != 0 ||
        causal_convolution.create_pipeline(opt) != 0)
        return false;

    ncnn::VkMat input_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(input, input_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return false;
    }

    ncnn::VkMat output_gpu;
    ncnn::Mat actual;
    ncnn::VkCompute compute(vkdev);
    if (causal_convolution.forward(input_gpu, output_gpu, compute, opt) != 0)
        return false;
    compute.record_download(output_gpu, actual, opt);
    const int submit_ret = compute.submit_and_wait();
    causal_convolution.destroy_pipeline(opt);
    input_gpu.release();
    output_gpu.release();
    blob_allocator->clear();
    staging_allocator->clear();
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (submit_ret != 0 || expected.total() != actual.total())
        return false;
    for (size_t index = 0; index < expected.total(); index++)
        if (std::fabs(expected[index] - actual[index]) > 1e-4f)
            return false;
    return true;
}

} // namespace

int main()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;

    ncnn::Mat input(3, 2, 2, 2);
    for (int channel = 0; channel < input.c; channel++)
        for (int depth = 0; depth < input.d; depth++)
            for (int row = 0; row < input.h; row++)
                for (int column = 0; column < input.w; column++)
                    input.channel(channel).depth(depth).row(row)[column] =
                        static_cast<float>(100 * channel + 10 * depth + 3 * row + column);

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

    ncnn::ParamDict params;
    params.set(0, 2);
    SeedVR2TemporalPad pad;
    pad.vkdev = vkdev;
    if (pad.load_param(params) != 0 || pad.create_pipeline(opt) != 0)
        return 1;

    ncnn::VkMat input_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(input, input_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat output_gpu;
    ncnn::Mat output;
    ncnn::VkCompute compute(vkdev);
    if (pad.forward(input_gpu, output_gpu, compute, opt) != 0)
        return 1;
    compute.record_download(output_gpu, output, opt);
    const int submit_ret = compute.submit_and_wait();
    pad.destroy_pipeline(opt);
    input_gpu.release();
    output_gpu.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (submit_ret != 0 || !matches_expected(output))
    {
        std::fprintf(stderr, "temporal pad submit=%d dims=%d w=%d h=%d d=%d c=%d pack=%d empty=%d\n",
                     submit_ret, output.dims, output.w, output.h, output.d, output.c,
                     output.elempack, output.empty());
        return 1;
    }

    if (!causal_conv3d_matches_cpu_reference(vkdev))
        return 1;

    std::puts("seedvr2-temporal-pad-vulkan: ok");
    return 0;
}
