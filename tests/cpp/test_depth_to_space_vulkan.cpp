#include <cmath>
#include <cstdio>

#include "command.h"
#include "gpu.h"
#include "vae/depth_to_space.h"

namespace
{

bool matches_expected(const ncnn::Mat& output)
{
    if (output.dims != 4 || output.w != 4 || output.h != 4 || output.d != 2 || output.c != 1)
    {
        std::fprintf(stderr, "shape dims=%d w=%d h=%d d=%d c=%d n=%d\n", output.dims, output.w, output.h, output.d, output.c, output.n);
        return false;
    }
    for (int depth = 0; depth < output.d; depth++)
        for (int row = 0; row < output.h; row++)
            for (int column = 0; column < output.w; column++)
            {
                const int source_channel = ((row % 2) * 2 + (column % 2)) * 2 + (depth % 2);
                const float expected = static_cast<float>(100 * source_channel + 10 * (row / 2) + column / 2);
                if (std::fabs(output.channel(0).depth(depth).row(row)[column] - expected) > 1e-6f)
                {
                    std::fprintf(stderr, "mismatch d=%d y=%d x=%d got=%f expected=%f source=%d\n", depth, row, column,
                                 output.channel(0).depth(depth).row(row)[column], expected, source_channel);
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

    ncnn::Mat input(2, 2, 1, 8);
    for (int channel = 0; channel < input.c; channel++)
        for (int row = 0; row < input.h; row++)
            for (int column = 0; column < input.w; column++)
                input.channel(channel).row(row)[column] = static_cast<float>(100 * channel + 10 * row + column);

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
    params.set(1, 2);
    params.set(2, 2);
    SeedVR2DepthToSpace layer;
    layer.vkdev = vkdev;
    if (layer.load_param(params) != 0 || layer.create_pipeline(opt) != 0)
        return 1;

    ncnn::VkMat input_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(input, input_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat input_pack1;
    ncnn::VkMat output_gpu;
    ncnn::Mat output;
    ncnn::VkCompute compute(vkdev);
    vkdev->convert_packing(input_gpu, input_pack1, 1, compute, opt);
    const int forward_ret = layer.forward(input_pack1, output_gpu, compute, opt);
    if (forward_ret != 0)
    {
        std::fprintf(stderr, "forward ret=%d\n", forward_ret);
        return 1;
    }
    compute.record_download(output_gpu, output, opt);
    const int submit_ret = compute.submit_and_wait();
    layer.destroy_pipeline(opt);
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (submit_ret != 0)
    {
        std::fprintf(stderr, "submit ret=%d\n", submit_ret);
        return 1;
    }
    if (!matches_expected(output))
        return 1;

    std::puts("seedvr2-depth-to-space-vulkan: ok");
    return 0;
}
