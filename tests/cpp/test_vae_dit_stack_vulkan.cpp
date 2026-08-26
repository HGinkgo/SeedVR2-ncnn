#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include "awa/awa_layers.h"
#include "command.h"
#include "conditioning/conditioning.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"
#include "vae/temporal_pad.h"

namespace
{

constexpr int kLatentChannels = 16;
constexpr int kLatentSize = 16;
constexpr int kPatchSize = 2;
constexpr int kVideoTokens = 64;
constexpr int kVideoPatchWidth = 132;
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
    net.set_vulkan_device(vkdev);
    return true;
}

bool load_vae(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
              ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_vae_layers(net);
    return net.load_param((stem + ".ncnn.param").c_str()) == 0 &&
           net.load_model((stem + ".ncnn.bin").c_str()) == 0;
}

bool load_graph(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
                ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_awa_layers(net);
    return net.load_param((stem + ".ncnn.param").c_str()) == 0 &&
           net.load_model((stem + ".ncnn.bin").c_str()) == 0;
}

bool load_packing_graph(ncnn::Net& net, ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
                        ncnn::VkAllocator* staging_allocator)
{
    static const char kPackingParam[] =
        "7767517\n"
        "2 2\n"
        "Input in0 0 1 in0\n"
        "Packing unpack 1 1 in0 out0 0=1\n";
    configure(net, vkdev, blob_allocator, staging_allocator);
    if (net.load_param_mem(kPackingParam) != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    return net.load_model(model_reader) == 0;
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

bool unpack_to_pack1(ncnn::Net& net, const ncnn::VkMat& packed, ncnn::VulkanDevice* vkdev,
                     ncnn::VkMat& unpacked)
{
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    return extractor.input("in0", packed) == 0 && extractor.extract("out0", unpacked, compute) == 0 &&
           compute.submit_and_wait() == 0;
}

bool matrix_to_batch_gpu(const ncnn::VkMat& matrix, int rows, ncnn::VulkanDevice* vkdev,
                         const ncnn::Option& opt, ncnn::VkAllocator* allocator, ncnn::VkMat& batch)
{
    if (matrix.empty() || matrix.dims != 2 || matrix.h != rows || matrix.w <= 0 || matrix.elempack != 1)
        return false;
    ncnn::VkMat source = matrix;
    source.dims = 1;
    source.h = 1;
    source.d = 1;
    source.c = 1;
    source.cstep = static_cast<size_t>(matrix.w);
#if NCNN_BATCH
    source.n = rows;
    source.nstep = source.cstep;
#endif
    batch.create(matrix.w, matrix.elemsize, 1, rows, allocator);
    if (batch.empty())
        return false;
    ncnn::VkCompute compute(vkdev);
    for (int row = 0; row < rows; row++)
    {
        ncnn::VkMat destination = batch.batch(row);
        compute.record_clone(source.batch(row), destination, opt);
    }
    return compute.submit_and_wait() == 0;
}

bool batch_to_matrix_gpu(const ncnn::VkMat& batch, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt,
                         ncnn::VkAllocator* allocator, ncnn::VkMat& matrix)
{
    if (batch.empty() || batch.dims != 1 || batch.elempack != 1 || batch.n <= 1 || batch.w <= 0)
        return false;
    matrix.create(batch.w, batch.n, batch.elemsize, 1, allocator);
    if (matrix.empty())
        return false;

    ncnn::VkCompute compute(vkdev);
    for (int row = 0; row < batch.n; row++)
    {
        ncnn::VkMat source = batch.batch(row);
        ncnn::VkMat destination = matrix;
        destination.dims = 1;
        destination.w = matrix.w;
        destination.h = 1;
        destination.d = 1;
        destination.c = 1;
        destination.cstep = static_cast<size_t>(matrix.w);
        destination.n = 1;
        destination.nstep = destination.cstep;
        destination.offset = matrix.offset + static_cast<size_t>(row) * matrix.w * matrix.elemsize;
        compute.record_clone(source, destination, opt);
    }
    return compute.submit_and_wait() == 0;
}

bool make_patches(const ncnn::Mat& latent, ncnn::Mat& patches)
{
    if (latent.empty() || (latent.dims != 3 && latent.dims != 4) || latent.d != 1 || latent.c != kLatentChannels ||
        latent.w != kLatentSize || latent.h != kLatentSize || latent.elemsize != 4u)
        return false;
    patches.create(kVideoPatchWidth, kVideoTokens);
    if (patches.empty())
        return false;
    patches.fill(0.f);
    for (int patch_y = 0; patch_y < kLatentSize / kPatchSize; patch_y++)
        for (int patch_x = 0; patch_x < kLatentSize / kPatchSize; patch_x++)
        {
            const int token = patch_y * (kLatentSize / kPatchSize) + patch_x;
            float* output = patches.row(token);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < kLatentChannels; channel++)
                    {
                        const int offset = kLatentChannels * (dy * kPatchSize + dx) + channel;
                        output[kLatentChannels * kPatchSize * kPatchSize + offset] =
                            latent.channel(channel).row(patch_y * kPatchSize + dy)[patch_x * kPatchSize + dx];
                    }
            output[kVideoPatchWidth - 1] = 1.f;
        }
    return true;
}

bool unpatch(const ncnn::Mat& patches, ncnn::Mat& latent)
{
    if (patches.empty() || patches.dims != 2 || patches.w != 64 || patches.h != kVideoTokens ||
        patches.elemsize != 4u)
        return false;
    latent.create(kLatentSize, kLatentSize, 1, kLatentChannels);
    if (latent.empty())
        return false;
    for (int patch_y = 0; patch_y < kLatentSize / kPatchSize; patch_y++)
        for (int patch_x = 0; patch_x < kLatentSize / kPatchSize; patch_x++)
        {
            const int token = patch_y * (kLatentSize / kPatchSize) + patch_x;
            const float* input = patches.row(token);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < kLatentChannels; channel++)
                    {
                        const int offset = kLatentChannels * (dy * kPatchSize + dx) + channel;
                        latent.channel(channel).row(patch_y * kPatchSize + dy)[patch_x * kPatchSize + dx] =
                            input[offset];
                    }
        }
    return finite(latent);
}

bool download(const ncnn::VkMat& source, ncnn::Mat& destination, ncnn::VulkanDevice* vkdev,
              const ncnn::Option& opt)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_download(source, destination, opt);
    return compute.submit_and_wait() == 0;
}

bool run_dit_stack(const ncnn::Mat& latent_input, const ncnn::Mat& text, float timestep_value,
                   const std::string& stack_dir, ncnn::VulkanDevice* vkdev,
                   ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator,
                   ncnn::Mat& latent_output)
{
    ncnn::Mat patches;
    if (!make_patches(latent_input, patches))
        return false;
    if (text.dims != 2 || text.w != kTextInputWidth || (text.h != 58 && text.h != 64) || text.elemsize != 4u)
        return false;

    ncnn::Mat timestep(1);
    timestep[0] = timestep_value;
    ncnn::Net dit_input;
    ncnn::Net dit_embedding;
    ncnn::Net packing;
    if (!load_graph(dit_input, stack_dir + "/dit_input", vkdev, blob_allocator, staging_allocator) ||
        !load_graph(dit_embedding, stack_dir + "/dit_embedding", vkdev, blob_allocator, staging_allocator) ||
        !load_packing_graph(packing, vkdev, blob_allocator, staging_allocator))
        return false;

    ncnn::VkMat video_packed;
    ncnn::VkMat text_packed;
    {
        ncnn::Extractor extractor = dit_input.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", patches) != 0 || extractor.input("in1", text) != 0 ||
            extractor.extract("out0", video_packed, compute) != 0 ||
            extractor.extract("out1", text_packed, compute) != 0 || compute.submit_and_wait() != 0)
            return false;
    }
    ncnn::VkMat video_matrix;
    ncnn::VkMat text_matrix;
    if (!unpack_to_pack1(packing, video_packed, vkdev, video_matrix) ||
        !unpack_to_pack1(packing, text_packed, vkdev, text_matrix))
        return false;

    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    if (!matrix_to_batch_gpu(video_matrix, kVideoTokens, vkdev, dit_input.opt, blob_allocator, video_gpu) ||
        !matrix_to_batch_gpu(text_matrix, text.h, vkdev, dit_input.opt, blob_allocator, text_gpu))
        return false;

    ncnn::VkMat embedding_packed;
    {
        ncnn::Extractor extractor = dit_embedding.create_extractor();
        ncnn::VkCompute compute(vkdev);
        ncnn::VkMat timestep_gpu;
        compute.record_upload(timestep, timestep_gpu, dit_embedding.opt);
        if (compute.submit_and_wait() != 0 || extractor.input("in0", timestep_gpu) != 0)
            return false;
        ncnn::VkCompute forward(vkdev);
        if (extractor.extract("out0", embedding_packed, forward) != 0 || forward.submit_and_wait() != 0)
            return false;
    }
    ncnn::VkMat embedding_gpu;
    if (!unpack_to_pack1(packing, embedding_packed, vkdev, embedding_gpu))
        return false;

    for (int block_index = 0; block_index < 32; block_index++)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "%s/dit_block_%02d", stack_dir.c_str(), block_index);
        ncnn::Net block;
        if (!load_graph(block, name, vkdev, blob_allocator, staging_allocator))
            return false;
        ncnn::Extractor extractor = block.create_extractor();
        extractor.set_light_mode(false);
        ncnn::VkMat next_video;
        ncnn::VkMat next_text;
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", video_gpu) != 0 || extractor.input("in1", text_gpu) != 0 ||
            extractor.input("in2", embedding_gpu) != 0 || extractor.extract("out0", next_video, compute) != 0 ||
            extractor.extract("out1", next_text, compute) != 0 || compute.submit_and_wait() != 0)
            return false;
        video_gpu = next_video;
        text_gpu = next_text;
    }

    ncnn::Net dit_output;
    if (!load_graph(dit_output, stack_dir + "/dit_output", vkdev, blob_allocator, staging_allocator))
        return false;
    ncnn::VkMat video_unpacked_final;
    if (!unpack_to_pack1(packing, video_gpu, vkdev, video_unpacked_final))
        return false;
    ncnn::VkMat video_matrix_final;
    if (!batch_to_matrix_gpu(video_unpacked_final, vkdev, dit_output.opt, blob_allocator, video_matrix_final))
        return false;
    ncnn::VkMat output_packed;
    {
        ncnn::Extractor extractor = dit_output.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", video_matrix_final) != 0 || extractor.input("in1", embedding_gpu) != 0 ||
            extractor.extract("out0", output_packed, compute) != 0 || compute.submit_and_wait() != 0)
            return false;
    }
    ncnn::VkMat output_matrix_gpu;
    if (!unpack_to_pack1(packing, output_packed, vkdev, output_matrix_gpu))
        return false;
    ncnn::Mat output_matrix;
    if (!download(output_matrix_gpu, output_matrix, vkdev, dit_output.opt))
        return false;
    return unpatch(output_matrix, latent_output);
}

float l1_difference(const ncnn::Mat& lhs, const ncnn::Mat& rhs)
{
    if (lhs.empty() || rhs.empty() || lhs.total() != rhs.total())
        return std::numeric_limits<float>::infinity();
    const float* lhs_data = static_cast<const float*>(lhs.data);
    const float* rhs_data = static_cast<const float*>(rhs.data);
    double sum = 0.0;
    for (size_t index = 0; index < lhs.total(); index++)
        sum += std::fabs(static_cast<double>(lhs_data[index]) - static_cast<double>(rhs_data[index]));
    return static_cast<float>(sum / static_cast<double>(lhs.total()));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6)
    {
        std::fprintf(stderr,
                     "usage: test_vae_dit_stack_vulkan <encode-stem> <stack-dir> <decode-stem> <condition-f32> [text-tokens]\n");
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
    const std::string encode_stem = argv[1];
    const std::string stack_dir = argv[2];
    const std::string decode_stem = argv[3];
    const std::string condition_path = argv[4];
    const int text_tokens = argc == 6 ? std::atoi(argv[5]) : 58;
    if (text_tokens != 58 && text_tokens != 64)
    {
        std::fprintf(stderr, "text-tokens must be 58 or 64\n");
        return 2;
    }

    ncnn::Net encode;
    ncnn::Net decode;
    if (!load_vae(encode, encode_stem, vkdev, blob_allocator, staging_allocator))
    {
        std::fprintf(stderr, "failed to load VAE or packing graph\n");
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
        ncnn::Extractor extractor = encode.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", sample_gpu) != 0 || extractor.extract("out0", latent_gpu, compute) != 0 ||
            compute.submit_and_wait() != 0)
            return 1;
    }
    ncnn::Mat latent;
    if (!download(latent_gpu, latent, vkdev, encode.opt))
    {
        std::fprintf(stderr, "stage=vae-encode-download failed\n");
        return 1;
    }
    latent_gpu = ncnn::VkMat();
    sample_gpu = ncnn::VkMat();
    encode.clear();

    ncnn::Mat text;
    if (!seedvr2::load_conditioning_f32(condition_path.c_str(), text_tokens, text))
    {
        std::fprintf(stderr, "stage=conditioning-load failed path=%s\n", condition_path.c_str());
        return 1;
    }
    ncnn::Mat real_latent;
    if (!run_dit_stack(latent, text, 0.f, stack_dir, vkdev, blob_allocator, staging_allocator, real_latent))
    {
        std::fprintf(stderr, "stage=real-conditioning-dit failed\n");
        return 1;
    }
    ncnn::Mat zero_text(kTextInputWidth, text.h);
    zero_text.fill(0.f);
    ncnn::Mat zero_latent;
    if (!run_dit_stack(latent, zero_text, 0.f, stack_dir, vkdev, blob_allocator, staging_allocator, zero_latent))
    {
        std::fprintf(stderr, "stage=zero-conditioning-dit failed\n");
        return 1;
    }
    const float condition_delta = l1_difference(real_latent, zero_latent);
    if (!std::isfinite(condition_delta) || condition_delta <= 1.0e-3f)
    {
        std::fprintf(stderr, "stage=conditioning-effect failed l1=%g\n", condition_delta);
        return 1;
    }
    latent = real_latent;

    if (!load_vae(decode, decode_stem, vkdev, blob_allocator, staging_allocator))
    {
        std::fprintf(stderr, "stage=load-decode failed\n");
        return 1;
    }

    ncnn::VkMat output_latent_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(latent, output_latent_gpu, decode.opt);
        if (compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "stage=decode-upload failed\n");
            return 1;
        }
    }
    ncnn::Mat reconstruction;
    {
        ncnn::Extractor extractor = decode.create_extractor();
        if (extractor.input("in0", output_latent_gpu) != 0 || extractor.extract("out0", reconstruction) != 0)
        {
            std::fprintf(stderr, "stage=vae-decode failed\n");
            return 1;
        }
    }
    if (!finite(reconstruction) || reconstruction.dims != 3 || reconstruction.w != 128 ||
        reconstruction.h != 128 || reconstruction.c != 3)
    {
        std::fprintf(stderr, "stage=decode-download-or-shape failed dims=%d w=%d h=%d c=%d\n", reconstruction.dims,
                     reconstruction.w, reconstruction.h, reconstruction.c);
        return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    std::puts("seedvr2-vae-dit-stack-vulkan: ok");
    return 0;
}
