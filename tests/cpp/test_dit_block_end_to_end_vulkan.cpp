#include <cmath>
#include <cstdio>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"
#include "net.h"

namespace
{

constexpr int kVideoTokens = 3600;
constexpr int kTextTokens = 58;
constexpr int kHiddenDim = 2560;
constexpr int kEmbeddingDim = 15360;

void fill_input(ncnn::Mat& value, int tokens, float offset)
{
    value.create(kHiddenDim, static_cast<size_t>(4u), 1, tokens);
    for (int token = 0; token < tokens; token++)
    {
        float* data = static_cast<float*>(value.batch(token).data);
        for (int feature = 0; feature < kHiddenDim; feature++)
            data[feature] = offset + static_cast<float>((token * 31 + feature * 17) % 1009) / 1009.f;
    }
}

bool is_finite(const ncnn::Mat& value)
{
    for (int batch = 0; batch < value.n; batch++)
    {
        const float* data = static_cast<const float*>(value.batch(batch).data);
        for (size_t index = 0; index < value.total(); index++)
            if (!std::isfinite(data[index]))
                return false;
    }
    return true;
}

bool is_token_batch(const ncnn::Mat& value, int tokens)
{
    return value.dims == 1 && value.w == kHiddenDim && value.h == 1 && value.d == 1 &&
           value.c == 1 && value.n == tokens && value.elemsize == 4u && value.elempack == 1;
}

int run_block(ncnn::Net& net, ncnn::VulkanDevice* vkdev, const ncnn::Mat& video,
              const ncnn::Mat& text, const ncnn::Mat& embedding, ncnn::Mat& video_output,
              ncnn::Mat& text_output)
{
    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    ncnn::VkMat embedding_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(video, video_gpu, net.opt);
        upload.record_upload(text, text_gpu, net.opt);
        upload.record_upload(embedding, embedding_gpu, net.opt);
        if (upload.submit_and_wait() != 0)
            return -1;
    }

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", video_gpu) != 0 || extractor.input("in1", text_gpu) != 0 ||
        extractor.input("in2", embedding_gpu) != 0)
        return -1;

    ncnn::VkMat video_output_gpu;
    ncnn::VkMat text_output_gpu;
    ncnn::VkCompute compute(vkdev);
    const int video_extract_result = extractor.extract("out0", video_output_gpu, compute);
    const int text_extract_result =
        video_extract_result == 0 ? extractor.extract("out1", text_output_gpu, compute) : -1;
    if (video_extract_result != 0 || text_extract_result != 0)
    {
        std::fprintf(stderr, "output extraction failed: out0=%d out1=%d\n", video_extract_result,
                     text_extract_result);
        return -1;
    }
    compute.record_download(video_output_gpu, video_output, net.opt);
    compute.record_download(text_output_gpu, text_output, net.opt);
    return compute.submit_and_wait();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: test_dit_block_end_to_end_vulkan <param> <bin>\n");
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

    int result = 1;
    {
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
        {
            std::fprintf(stderr, "failed to load DiT graph\n");
        }
        else
        {
            ncnn::Mat video;
            ncnn::Mat text;
            ncnn::Mat embedding(kEmbeddingDim, 1);
            fill_input(video, kVideoTokens, -0.5f);
            fill_input(text, kTextTokens, 0.25f);
            float* embedding_data = static_cast<float*>(embedding.data);
            for (int feature = 0; feature < kEmbeddingDim; feature++)
                embedding_data[feature] = static_cast<float>((feature * 13) % 997) / 997.f;

            ncnn::Mat video_output;
            ncnn::Mat text_output;
            if (run_block(net, vkdev, video, text, embedding, video_output, text_output) != 0)
            {
                std::fprintf(stderr, "Vulkan DiT graph execution failed\n");
            }
            else if (!is_token_batch(video_output, kVideoTokens) ||
                     !is_token_batch(text_output, kTextTokens) || !is_finite(video_output) ||
                     !is_finite(text_output))
            {
                std::fprintf(stderr, "unexpected Vulkan DiT output layout or values\n");
            }
            else
            {
                result = 0;
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (result == 0)
        std::puts("seedvr2-dit-block-end-to-end-vulkan: ok");
    return result;
}
