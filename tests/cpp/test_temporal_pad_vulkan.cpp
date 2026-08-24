#include <cmath>
#include <cstdio>

#include "command.h"
#include "gpu.h"
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
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (submit_ret != 0 || !matches_expected(output))
    {
        std::fprintf(stderr, "temporal pad submit=%d dims=%d w=%d h=%d d=%d c=%d pack=%d empty=%d\n",
                     submit_ret, output.dims, output.w, output.h, output.d, output.c,
                     output.elempack, output.empty());
        return 1;
    }

    std::puts("seedvr2-temporal-pad-vulkan: ok");
    return 0;
}
