#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"

namespace
{

struct AwaConfig
{
    int source_t;
    int source_h;
    int source_w;
    int windows_t;
    int windows_h;
    int windows_w;
    int text_tokens;
    int shifted;
    bool verify_token_batch_vulkan;
};

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

ncnn::ParamDict make_params(const AwaConfig& config)
{
    ncnn::ParamDict params;
    params.set(0, config.source_t);
    params.set(1, config.source_h);
    params.set(2, config.source_w);
    params.set(3, config.windows_t);
    params.set(4, config.windows_h);
    params.set(5, config.windows_w);
    params.set(6, config.text_tokens);
    params.set(7, config.shifted);
    return params;
}

void make_inputs(const AwaConfig& config, ncnn::Mat& video, ncnn::Mat& text)
{
    const int source_tokens = config.source_t * config.source_h * config.source_w;
    constexpr int heads = 2;
    constexpr int head_dim = 3;
    video.create(head_dim, heads, 3, source_tokens);
    text.create(head_dim, heads, 3, config.text_tokens);
    for (int token = 0; token < source_tokens; token++)
    {
        float* data = static_cast<float*>(video.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 17 + value) % 101) / 100.f;
    }
    for (int token = 0; token < config.text_tokens; token++)
    {
        float* data = static_cast<float*>(text.channel(token).data);
        for (int value = 0; value < 3 * heads * head_dim; value++)
            data[value] = static_cast<float>((token * 23 + value) % 79) / 79.f;
    }
}

bool make_token_batched_qkv(const ncnn::Mat& source, ncnn::Mat& destination)
{
    if (source.dims != 4 || source.d != 3 || source.n != 1)
        return false;

    destination.create(source.w, source.h, source.d, source.elemsize, source.elempack, source.c);
    if (destination.empty())
        return false;

    const size_t qkv_size = static_cast<size_t>(source.w) * source.h * sizeof(float);
    for (int token = 0; token < source.c; token++)
        for (int qkv = 0; qkv < source.d; qkv++)
            std::memcpy(destination.batch(token).channel(qkv).data,
                        source.channel(token).depth(qkv).data, qkv_size);
    return true;
}

int run_vulkan(const AwaConfig& config, const ncnn::Mat& packed, const ncnn::Option& base_opt,
               ncnn::Mat& output_video, ncnn::Mat& output_text)
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

    const ncnn::ParamDict params = make_params(config);

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
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(packed, packed_gpu, opt, false);
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

int run_pack_vulkan(const AwaConfig& config, const ncnn::Mat& video, const ncnn::Mat& text,
                    const ncnn::Option& base_opt, ncnn::Mat& output_packed, ncnn::Mat& output_cu)
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

    const ncnn::ParamDict params = make_params(config);

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
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(video, video_gpu, opt, false);
        upload.record_upload(text, text_gpu, opt, false);
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

bool verify_config(const AwaConfig& config)
{
    ncnn::Mat video;
    ncnn::Mat text;
    make_inputs(config, video, text);

    const ncnn::ParamDict params = make_params(config);

    SeedVR2AWAPack pack;
    if (pack.load_param(params) != 0)
        return false;
    ncnn::Option opt;
    std::vector<ncnn::Mat> pack_outputs;
    if (pack.forward({video, text}, pack_outputs, opt) != 0)
        return false;

    ncnn::Mat video_batched;
    ncnn::Mat text_batched;
    std::vector<ncnn::Mat> batched_pack_outputs;
    if (!make_token_batched_qkv(video, video_batched) || !make_token_batched_qkv(text, text_batched) ||
        pack.forward({video_batched, text_batched}, batched_pack_outputs, opt) != 0 ||
        batched_pack_outputs.size() != 2 || !compare(pack_outputs[0], batched_pack_outputs[0], 1e-6f) ||
        !compare(pack_outputs[1], batched_pack_outputs[1], 1e-6f))
    {
        std::fprintf(stderr, "Token-batch CPU Pack output differs from token-channel layout\n");
        return false;
    }

    const ncnn::Mat& packed_qkv = pack_outputs[0];
    ncnn::Mat packed(packed_qkv.w, packed_qkv.h / 3, packed_qkv.c);
    for (int token = 0; token < packed.c; token++)
        std::memcpy(packed.channel(token).data, packed_qkv.channel(token).data,
                    static_cast<size_t>(packed.w * packed.h) * sizeof(float));

    SeedVR2AWAUnpack unpack_cpu;
    if (unpack_cpu.load_param(params) != 0)
        return false;
    std::vector<ncnn::Mat> cpu_outputs;
    if (unpack_cpu.forward({packed}, cpu_outputs, opt) != 0 || cpu_outputs.size() != 2)
        return false;

    ncnn::Mat gpu_video;
    ncnn::Mat gpu_text;
    ncnn::Mat gpu_packed;
    ncnn::Mat gpu_cu;
    if (run_pack_vulkan(config, video, text, opt, gpu_packed, gpu_cu) != 0 ||
        !compare(pack_outputs[0], gpu_packed, 1e-3f) || !compare(pack_outputs[1], gpu_cu, 1e-3f))
    {
        std::fprintf(stderr, "CPU/Vulkan Pack outputs differ\n");
        return false;
    }
    if (config.verify_token_batch_vulkan &&
        (run_pack_vulkan(config, video_batched, text_batched, opt, gpu_packed, gpu_cu) != 0 ||
         !compare(pack_outputs[0], gpu_packed, 1e-3f) || !compare(pack_outputs[1], gpu_cu, 1e-3f)))
    {
        std::fprintf(stderr, "Token-batch CPU/Vulkan Pack outputs differ\n");
        return false;
    }
    if (run_vulkan(config, packed, opt, gpu_video, gpu_text) != 0)
    {
        std::fprintf(stderr, "Vulkan Unpack execution failed\n");
        return false;
    }
    if (!compare(cpu_outputs[0], gpu_video, 1e-3f) || !compare(cpu_outputs[1], gpu_text, 1e-3f))
    {
        std::fprintf(stderr, "CPU/Vulkan Unpack outputs differ\n");
        return false;
    }

    return true;
}

} // namespace

int main()
{
    const AwaConfig legacy = {2, 19, 23, 4, 3, 3, 5, 1, true};
    const AwaConfig target_unshifted = {1, 45, 80, 4, 3, 3, 58, 0, false};
    const AwaConfig target_shifted = {1, 45, 80, 4, 3, 3, 58, 1, false};
    if (!verify_config(legacy) || !verify_config(target_unshifted) || !verify_config(target_shifted))
        return 1;

    std::puts("seedvr2-awa-unpack-vulkan: ok");
    return 0;
}
