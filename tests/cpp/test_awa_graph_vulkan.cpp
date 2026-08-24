#include <cmath>
#include <cstdio>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"
#include "net.h"

namespace
{

void fill_inputs(ncnn::Mat& video, ncnn::Mat& text)
{
    constexpr int source_tokens = 2 * 19 * 23;
    constexpr int text_tokens = 5;
    constexpr int heads = 2;
    constexpr int head_dim = 3;
    video.create(head_dim, heads, 3, source_tokens);
    text.create(head_dim, heads, 3, text_tokens);
    for (int token = 0; token < source_tokens; token++)
    {
        float* data = static_cast<float*>(video.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 17 + value) % 101) / 100.f;
    }
    for (int token = 0; token < text_tokens; token++)
    {
        float* data = static_cast<float*>(text.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 23 + value) % 79) / 79.f;
    }
}

bool finite_logical(const ncnn::Mat& value)
{
    for (int batch = 0; batch < value.n; batch++)
    {
        const ncnn::Mat batch_value = value.batch(batch);
        for (int channel = 0; channel < value.c; channel++)
        {
            const ncnn::Mat channel_value = batch_value.channel(channel);
            for (int depth = 0; depth < (value.dims == 4 ? value.d : 1); depth++)
            {
                const ncnn::Mat depth_value = value.dims == 4 ? channel_value.depth(depth) : channel_value;
                for (int row = 0; row < value.h; row++)
                    for (int col = 0; col < value.w; col++)
                        if (!std::isfinite(depth_value.row(row)[col]))
                            return false;
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: test_awa_graph_vulkan <param> <bin>\n");
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.blob_vkallocator = blob_allocator;
    net.opt.workspace_vkallocator = blob_allocator;
    net.opt.staging_vkallocator = staging_allocator;
    register_seedvr2_awa_layers(net);
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0)
        return 1;

    ncnn::Mat video;
    ncnn::Mat text;
    fill_inputs(video, text);
    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(video, video_gpu, net.opt);
        upload.record_upload(text, text_gpu, net.opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", video_gpu) != 0 || extractor.input("in1", text_gpu) != 0)
        return 1;

    ncnn::VkMat output_video_gpu;
    ncnn::VkMat output_text_gpu;
    ncnn::VkCompute compute(vkdev);
    if (extractor.extract("out0", output_video_gpu, compute) != 0 ||
        extractor.extract("out1", output_text_gpu, compute) != 0)
        return 1;
    ncnn::Mat output_video;
    ncnn::Mat output_text;
    compute.record_download(output_video_gpu, output_video, net.opt);
    compute.record_download(output_text_gpu, output_text, net.opt);
    if (compute.submit_and_wait() != 0)
        return 1;

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    const bool video_shape = output_video.dims == 4 && output_video.w == 3 && output_video.h == 2 &&
                             output_video.d == 23 && output_video.c == 19 && output_video.n == 2;
    const bool text_shape = output_text.dims == 3 && output_text.w == 3 && output_text.h == 2 &&
                            output_text.c == 5;
    if (!video_shape || !text_shape || !finite_logical(output_video) || !finite_logical(output_text))
    {
        std::fprintf(stderr, "unexpected Vulkan graph output shape or value\n");
        return 1;
    }

    std::puts("seedvr2-awa-graph-vulkan: ok");
    return 0;
}
