#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"

namespace
{

bool compare(const ncnn::Mat& lhs, const ncnn::Mat& rhs, float tolerance)
{
    if (lhs.dims != rhs.dims || lhs.w != rhs.w || lhs.h != rhs.h || lhs.d != rhs.d || lhs.c != rhs.c || lhs.n != rhs.n)
        return false;
    for (int batch = 0; batch < lhs.n; batch++)
    {
        const ncnn::Mat lhs_batch = lhs.batch(batch);
        const ncnn::Mat rhs_batch = rhs.batch(batch);
        for (int channel = 0; channel < lhs.c; channel++)
        {
            const ncnn::Mat lhs_channel = lhs_batch.channel(channel);
            const ncnn::Mat rhs_channel = rhs_batch.channel(channel);
            for (int depth = 0; depth < (lhs.dims == 4 ? lhs.d : 1); depth++)
            {
                const ncnn::Mat lhs_depth = lhs.dims == 4 ? lhs_channel.depth(depth) : lhs_channel;
                const ncnn::Mat rhs_depth = rhs.dims == 4 ? rhs_channel.depth(depth) : rhs_channel;
                for (int row = 0; row < lhs.h; row++)
                    for (int col = 0; col < lhs.w; col++)
                    {
                        const float a = lhs_depth.row(row)[col];
                        const float b = rhs_depth.row(row)[col];
                        if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(a - b) > tolerance)
                            return false;
                    }
            }
        }
    }
    return true;
}

void make_inputs(ncnn::Mat& video, ncnn::Mat& text)
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

int run_vulkan(const ncnn::Mat& packed, const ncnn::Option& base_opt, ncnn::Mat& output_video, ncnn::Mat& output_text)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return -1;

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    ncnn::Option opt = base_opt;
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
    params.set(1, 19);
    params.set(2, 23);
    params.set(3, 4);
    params.set(4, 3);
    params.set(5, 3);
    params.set(6, 5);
    params.set(7, 1);

    SeedVR2AWAUnpack unpack;
    if (unpack.load_param(params) != 0)
        return -1;
    unpack.vkdev = vkdev;
    if (unpack.create_pipeline(opt) != 0)
        return -1;

    {
        ncnn::VkTransfer transfer(vkdev);
        if (unpack.upload_model(transfer, opt) != 0 || transfer.submit_and_wait() != 0)
            return -1;
    }

    ncnn::VkMat packed_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(packed, packed_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return -1;
    }

    std::vector<ncnn::VkMat> top_blobs;
    ncnn::VkCompute compute(vkdev);
    if (unpack.forward({packed_gpu}, top_blobs, compute, opt) != 0 || top_blobs.size() != 2)
        return -1;
    compute.record_download(top_blobs[0], output_video, opt);
    compute.record_download(top_blobs[1], output_text, opt);
    const int ret = compute.submit_and_wait();
    unpack.destroy_pipeline(opt);
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return ret;
}

int run_pack_vulkan(const ncnn::Mat& video, const ncnn::Mat& text, const ncnn::Option& base_opt,
                    ncnn::Mat& output_packed, ncnn::Mat& output_cu)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return -1;

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    ncnn::Option opt = base_opt;
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
    params.set(1, 19);
    params.set(2, 23);
    params.set(3, 4);
    params.set(4, 3);
    params.set(5, 3);
    params.set(6, 5);
    params.set(7, 1);

    SeedVR2AWAPack pack;
    if (pack.load_param(params) != 0)
        return -1;
    pack.vkdev = vkdev;
    if (pack.create_pipeline(opt) != 0)
        return -1;
    {
        ncnn::VkTransfer transfer(vkdev);
        if (pack.upload_model(transfer, opt) != 0 || transfer.submit_and_wait() != 0)
            return -1;
    }

    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(video, video_gpu, opt);
        upload.record_upload(text, text_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return -1;
    }

    std::vector<ncnn::VkMat> top_blobs;
    ncnn::VkCompute compute(vkdev);
    if (pack.forward({video_gpu, text_gpu}, top_blobs, compute, opt) != 0 || top_blobs.size() != 2)
        return -1;
    compute.record_download(top_blobs[0], output_packed, opt);
    compute.record_download(top_blobs[1], output_cu, opt);
    const int ret = compute.submit_and_wait();
    pack.destroy_pipeline(opt);
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return ret;
}

} // namespace

int main()
{
    ncnn::Mat video;
    ncnn::Mat text;
    make_inputs(video, text);

    ncnn::ParamDict params;
    params.set(0, 2);
    params.set(1, 19);
    params.set(2, 23);
    params.set(3, 4);
    params.set(4, 3);
    params.set(5, 3);
    params.set(6, 5);
    params.set(7, 1);

    SeedVR2AWAPack pack;
    if (pack.load_param(params) != 0)
        return 1;
    ncnn::Option opt;
    std::vector<ncnn::Mat> pack_outputs;
    if (pack.forward({video, text}, pack_outputs, opt) != 0)
        return 1;

    const ncnn::Mat& packed_qkv = pack_outputs[0];
    ncnn::Mat packed(packed_qkv.w, packed_qkv.h / 3, packed_qkv.c);
    for (int token = 0; token < packed.c; token++)
        std::memcpy(packed.channel(token).data, packed_qkv.channel(token).data,
                    static_cast<size_t>(packed.w * packed.h) * sizeof(float));

    SeedVR2AWAUnpack unpack_cpu;
    if (unpack_cpu.load_param(params) != 0)
        return 1;
    std::vector<ncnn::Mat> cpu_outputs;
    if (unpack_cpu.forward({packed}, cpu_outputs, opt) != 0 || cpu_outputs.size() != 2)
        return 1;

    ncnn::Mat gpu_video;
    ncnn::Mat gpu_text;
    ncnn::Mat gpu_packed;
    ncnn::Mat gpu_cu;
    if (run_pack_vulkan(video, text, opt, gpu_packed, gpu_cu) != 0 ||
        !compare(pack_outputs[0], gpu_packed, 1e-3f) || !compare(pack_outputs[1], gpu_cu, 1e-3f))
    {
        std::fprintf(stderr, "CPU/Vulkan Pack outputs differ\n");
        return 1;
    }
    if (run_vulkan(packed, opt, gpu_video, gpu_text) != 0)
    {
        std::fprintf(stderr, "Vulkan Unpack execution failed\n");
        return 1;
    }
    if (!compare(cpu_outputs[0], gpu_video, 1e-3f) || !compare(cpu_outputs[1], gpu_text, 1e-3f))
    {
        std::fprintf(stderr, "CPU/Vulkan Unpack outputs differ\n");
        return 1;
    }

    std::puts("seedvr2-awa-unpack-vulkan: ok");
    return 0;
}
