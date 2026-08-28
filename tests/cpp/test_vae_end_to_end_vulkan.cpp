#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "allocator.h"
#include "command.h"
#include "gpu.h"
#include "net.h"
#include "vae/temporal_pad.h"
#include "vulkan/transient_staging_allocator.h"

namespace
{

ncnn::Option make_vulkan_option(ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    return opt;
}

bool load_graph(ncnn::Net& net, const char* param_path, const char* model_path,
                ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
                ncnn::VkAllocator* staging_allocator, bool low_memory_cpu = false)
{
    net.opt = make_vulkan_option(blob_allocator, staging_allocator);
    net.opt.use_local_pool_allocator = !low_memory_cpu;
    net.set_vulkan_device(vkdev);
    register_seedvr2_vae_layers(net);
    return net.load_param(param_path) == 0 && net.load_model(model_path) == 0;
}

bool clone_to_allocator(const ncnn::VkMat& source, ncnn::VkMat& destination, ncnn::VulkanDevice* vkdev,
                        ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    ncnn::VkCompute compute(vkdev);
    compute.record_clone(source, destination, make_vulkan_option(blob_allocator, staging_allocator));
    return !destination.empty() && compute.submit_and_wait() == 0;
}

bool parse_dimension(const char* text, int& value)
{
    char* end = 0;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
        return false;

    value = static_cast<int>(parsed);
    return true;
}

bool finite(const ncnn::Mat& values)
{
    const float* data = values;
    for (size_t index = 0; index < values.total(); index++)
    {
        if (!std::isfinite(data[index]))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 7)
    {
        std::fprintf(stderr,
                     "usage: test_vae_end_to_end_vulkan <encode.param> <encode.bin> <decode.param> <decode.bin> [height width]\n");
        return 2;
    }

    int sample_height = 128;
    int sample_width = 128;
    if (argc == 7 &&
        (!parse_dimension(argv[5], sample_height) || !parse_dimension(argv[6], sample_width)))
    {
        std::fprintf(stderr, "height and width must be positive integers\n");
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;
    std::fprintf(stderr, "vulkan-heap-budget-mib=%u\n", vkdev->get_heap_budget());
    ncnn::VkAllocator* encode_blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* encode_staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net encode;
    std::fprintf(stderr, "stage=load-encode\n");
    if (!load_graph(encode, argv[1], argv[2], vkdev, encode_blob_allocator, encode_staging_allocator))
        return 1;

    ncnn::Mat sample(sample_width, sample_height, 1, 3);
    sample.fill(0.f);
    ncnn::VkMat sample_gpu;
    {
        std::fprintf(stderr, "stage=upload-sample\n");
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(sample, sample_gpu, encode.opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::VkMat latent_gpu;
    {
        ncnn::Extractor encode_extractor = encode.create_extractor();
        if (encode_extractor.input("in0", sample_gpu) != 0)
            return 1;
        std::fprintf(stderr, "stage=encode-extract\n");
        ncnn::VkCompute compute(vkdev);
        if (encode_extractor.extract("out0", latent_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    // Decode is the peak-memory phase at large resolutions. Release the
    // completed encoder's graph, input, and extractor before allocating it.
    sample_gpu.release();
    encode.clear();

    ncnn::VkAllocator* decode_blob_allocator = vkdev->acquire_blob_allocator();
    seedvr2::TransientVkStagingAllocator decode_staging_allocator(vkdev);
    ncnn::VkMat decode_latent_gpu;
    std::fprintf(stderr, "stage=handoff-latent\n");
    if (!clone_to_allocator(latent_gpu, decode_latent_gpu, vkdev, decode_blob_allocator, &decode_staging_allocator))
        return 1;

    latent_gpu.release();
    encode_blob_allocator->clear();
    encode_staging_allocator->clear();
    vkdev->reclaim_blob_allocator(encode_blob_allocator);
    vkdev->reclaim_staging_allocator(encode_staging_allocator);

    ncnn::Net decode;
    std::fprintf(stderr, "stage=load-decode\n");
    if (!load_graph(decode, argv[3], argv[4], vkdev, decode_blob_allocator, &decode_staging_allocator, true))
        return 1;

    ncnn::Mat reconstruction;
    {
        ncnn::Extractor decode_extractor = decode.create_extractor();
        if (decode_extractor.input("in0", decode_latent_gpu) != 0)
            return 1;
        std::fprintf(stderr, "stage=decode-extract\n");
        // The graph ends in the custom causal Conv3D layer. Host extraction
        // keeps the final image boundary explicit while convolution runs on
        // the Vulkan path.
        if (decode_extractor.extract("out0", reconstruction) != 0)
        {
            return 1;
        }
    }

    if (reconstruction.empty() || reconstruction.dims != 3 || reconstruction.w != sample_width ||
        reconstruction.h != sample_height || reconstruction.c != 3 ||
        reconstruction.total() != static_cast<size_t>(3) * sample_width * sample_height ||
        !finite(reconstruction))
    {
        std::fprintf(stderr, "unexpected Vulkan reconstruction shape\n");
        return 1;
    }

    decode_latent_gpu.release();
    decode.clear();
    decode_blob_allocator->clear();
    decode_staging_allocator.clear();
    vkdev->reclaim_blob_allocator(decode_blob_allocator);

    std::puts("seedvr2-vae-end-to-end-vulkan: ok");
    return 0;
}
