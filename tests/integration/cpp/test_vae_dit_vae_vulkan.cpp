#include <cmath>
#include <cstdio>
#include <cstring>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"
#include "net.h"
#include "vae/temporal_pad.h"

namespace
{

constexpr int kLatentChannels = 16;
constexpr int kLatentSize = 16;
constexpr int kPatchSize = 2;
constexpr int kPatchGrid = kLatentSize / kPatchSize;
constexpr int kVideoTokens = kPatchGrid * kPatchGrid;
constexpr int kVideoPatchWidth = 33 * kPatchSize * kPatchSize;
constexpr int kTextTokens = 58;
constexpr int kTextInputWidth = 5120;
constexpr int kHiddenWidth = 2560;
constexpr int kEmbeddingWidth = 15360;

bool configure(ncnn::Net& net, ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
               ncnn::VkAllocator* staging_allocator)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.blob_vkallocator = blob_allocator;
    net.opt.workspace_vkallocator = blob_allocator;
    net.opt.staging_vkallocator = staging_allocator;
    return true;
}

bool load_vae(ncnn::Net& net, const char* param, const char* model,
              ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
              ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_vae_layers(net);
    return net.load_param(param) == 0 && net.load_model(model) == 0;
}

bool load_awa(ncnn::Net& net, const char* param, const char* model,
             ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
             ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_awa_layers(net);
    return net.load_param(param) == 0 && net.load_model(model) == 0;
}

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

ncnn::Mat make_matrix(int width, int rows)
{
    ncnn::Mat value(width, rows);
    value.fill(0.f);
    return value;
}

bool matrix_to_batch(const ncnn::Mat& matrix, ncnn::Mat& batch)
{
    if (matrix.dims != 2 || matrix.w <= 0 || matrix.h <= 0)
        return false;
    batch.create(matrix.w, static_cast<size_t>(4u), 1, matrix.h);
    for (int row = 0; row < matrix.h; row++)
        std::memcpy(batch.batch(row).data, matrix.row(row), static_cast<size_t>(matrix.w) * sizeof(float));
    return true;
}

bool batch_to_matrix(const ncnn::Mat& batch, ncnn::Mat& matrix)
{
    if (batch.dims == 2)
    {
        matrix = batch.clone();
        return true;
    }
    if (batch.dims != 1 || batch.w <= 0 || batch.n <= 0)
        return false;
    matrix.create(batch.w, batch.n);
    for (int row = 0; row < batch.n; row++)
        std::memcpy(matrix.row(row), batch.batch(row).data, static_cast<size_t>(batch.w) * sizeof(float));
    return true;
}

bool make_patch_input(const ncnn::Mat& latent, ncnn::Mat& patches)
{
    if (latent.empty() || latent.total() != kLatentChannels * kLatentSize * kLatentSize)
        return false;
    patches = make_matrix(kVideoPatchWidth, kVideoTokens);
    for (int patch_y = 0; patch_y < kPatchGrid; patch_y++)
        for (int patch_x = 0; patch_x < kPatchGrid; patch_x++)
        {
            const int token = patch_y * kPatchGrid + patch_x;
            float* output = patches.row(token);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < kLatentChannels; channel++)
                    {
                        const int offset = ((dy * kPatchSize + dx) * kLatentChannels) + channel;
                        output[kLatentChannels * kPatchSize * kPatchSize + offset] =
                            latent.channel(channel).row(patch_y * kPatchSize + dy)[patch_x * kPatchSize + dx];
                    }
            output[kVideoPatchWidth - 1] = 1.f;
        }
    return true;
}

bool unpatch_output(const ncnn::Mat& patches, ncnn::Mat& latent)
{
    if (patches.empty() || patches.total() != static_cast<size_t>(kVideoTokens) * 64u)
        return false;
    latent.create(kLatentSize, kLatentSize, 1, kLatentChannels);
    for (int patch_y = 0; patch_y < kPatchGrid; patch_y++)
        for (int patch_x = 0; patch_x < kPatchGrid; patch_x++)
        {
            const int token = patch_y * kPatchGrid + patch_x;
            const float* input = patches.dims == 2 ? patches.row(token) :
                static_cast<const float*>(patches.batch(token).data);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < kLatentChannels; channel++)
                    {
                        const int offset = ((dy * kPatchSize + dx) * kLatentChannels) + channel;
                        latent.channel(channel).row(patch_y * kPatchSize + dy)[patch_x * kPatchSize + dx] =
                            input[offset];
                    }
        }
    return finite(latent);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 11)
    {
        std::fprintf(stderr,
                     "usage: test_vae_dit_vae_vulkan <encode.param> <encode.bin> "
                     "<patch-in.param> <patch-in.bin> <block.param> <block.bin> "
                     "<patch-out.param> <patch-out.bin> <decode.param> <decode.bin>\n");
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net encode;
    ncnn::Net patch_in;
    ncnn::Net block;
    ncnn::Net patch_out;
    ncnn::Net decode;
    if (!load_vae(encode, argv[1], argv[2], vkdev, blob_allocator, staging_allocator) ||
        !load_awa(patch_in, argv[3], argv[4], vkdev, blob_allocator, staging_allocator) ||
        !load_awa(block, argv[5], argv[6], vkdev, blob_allocator, staging_allocator) ||
        !load_awa(patch_out, argv[7], argv[8], vkdev, blob_allocator, staging_allocator) ||
        !load_vae(decode, argv[9], argv[10], vkdev, blob_allocator, staging_allocator))
    {
        std::fprintf(stderr, "failed to load VAE/DiT bridge graphs\n");
        return 1;
    }

    ncnn::Mat sample(128, 128, 1, 3);
    sample.fill(0.f);
    ncnn::VkMat sample_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(sample, sample_gpu, encode.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat latent_gpu;
    {
        std::fprintf(stderr, "stage=vae-encode\n");
        ncnn::Extractor extractor = encode.create_extractor();
        if (extractor.input("in0", sample_gpu) != 0)
            return 1;
        ncnn::VkCompute compute(vkdev);
        if (extractor.extract("out0", latent_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Mat latent;
    {
        std::fprintf(stderr, "stage=download-latent\n");
        ncnn::VkCompute compute(vkdev);
        compute.record_download(latent_gpu, latent, encode.opt);
        if (compute.submit_and_wait() != 0 || !finite(latent))
            return 1;
    }

    ncnn::Mat patches;
    if (!make_patch_input(latent, patches))
    {
        std::fprintf(stderr, "unexpected VAE latent layout\n");
        return 1;
    }
    ncnn::Mat text = make_matrix(kTextInputWidth, kTextTokens);
    ncnn::Mat embedding = make_matrix(kEmbeddingWidth, 1);
    float* embedding_data = static_cast<float*>(embedding.data);
    for (int index = 0; index < kEmbeddingWidth; index++)
        embedding_data[index] = static_cast<float>((index * 13) % 997) / 997.f;

    ncnn::VkMat video_tokens_gpu;
    ncnn::VkMat text_tokens_gpu;
    {
        std::fprintf(stderr, "stage=patch-in\n");
        ncnn::Extractor extractor = patch_in.create_extractor();
        if (extractor.input("in0", patches) != 0 || extractor.input("in1", text) != 0)
            return 1;
        ncnn::VkCompute compute(vkdev);
        if (extractor.extract("out0", video_tokens_gpu, compute) != 0 ||
            extractor.extract("out1", text_tokens_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Mat video_tokens_matrix;
    ncnn::Mat text_tokens_matrix;
    {
        std::fprintf(stderr, "stage=download-patch-in\n");
        ncnn::VkCompute compute(vkdev);
        compute.record_download(video_tokens_gpu, video_tokens_matrix, patch_in.opt);
        compute.record_download(text_tokens_gpu, text_tokens_matrix, patch_in.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }
    ncnn::Mat video_tokens_batch;
    ncnn::Mat text_tokens_batch;
    if (!matrix_to_batch(video_tokens_matrix, video_tokens_batch) ||
        !matrix_to_batch(text_tokens_matrix, text_tokens_batch))
        return 1;
    ncnn::VkMat video_tokens_matrix_gpu;
    ncnn::VkMat text_tokens_matrix_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(video_tokens_batch, video_tokens_matrix_gpu, block.opt);
        compute.record_upload(text_tokens_batch, text_tokens_matrix_gpu, block.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat block_video_gpu;
    {
        std::fprintf(stderr, "stage=dit-block\n");
        ncnn::Extractor extractor = block.create_extractor();
        if (extractor.input("in0", video_tokens_matrix_gpu) != 0 ||
            extractor.input("in1", text_tokens_matrix_gpu) != 0 || extractor.input("in2", embedding) != 0)
            return 1;
        ncnn::VkCompute compute(vkdev);
        const int out0 = extractor.extract("out0", block_video_gpu, compute);
        const int submit = out0 == 0 ? compute.submit_and_wait() : -1;
        if (out0 != 0 || submit != 0)
        {
            std::fprintf(stderr, "dit-block extract out0=%d submit=%d\n", out0, submit);
            return 1;
        }
    }

    ncnn::Mat block_video;
    {
        std::fprintf(stderr, "stage=download-block\n");
        ncnn::VkCompute compute(vkdev);
        compute.record_download(block_video_gpu, block_video, block.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }
    ncnn::Mat block_video_matrix;
    if (!batch_to_matrix(block_video, block_video_matrix))
        return 1;
    ncnn::VkMat block_video_matrix_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(block_video_matrix, block_video_matrix_gpu, patch_out.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat latent_patches_gpu;
    {
        std::fprintf(stderr, "stage=patch-out\n");
        ncnn::Extractor extractor = patch_out.create_extractor();
        if (extractor.input("in0", block_video_matrix_gpu) != 0)
            return 1;
        ncnn::VkCompute compute(vkdev);
        if (extractor.extract("out0", latent_patches_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Mat latent_patches;
    {
        std::fprintf(stderr, "stage=download-patch-out\n");
        ncnn::VkCompute compute(vkdev);
        compute.record_download(latent_patches_gpu, latent_patches, patch_out.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }
    ncnn::Mat decoded_latent;
    if (!unpatch_output(latent_patches, decoded_latent))
        return 1;

    ncnn::VkMat decoded_latent_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(decoded_latent, decoded_latent_gpu, decode.opt);
        if (compute.submit_and_wait() != 0)
            return 1;
    }
    ncnn::Mat reconstruction;
    {
        std::fprintf(stderr, "stage=vae-decode\n");
        ncnn::Extractor extractor = decode.create_extractor();
        if (extractor.input("in0", decoded_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
            return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (reconstruction.empty() || reconstruction.dims != 3 || reconstruction.w != 128 ||
        reconstruction.h != 128 || reconstruction.c != 3 || !finite(reconstruction))
        return 1;

    std::puts("seedvr2-vae-dit-vae-vulkan: ok");
    return 0;
}
