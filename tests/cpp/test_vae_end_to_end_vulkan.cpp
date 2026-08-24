#include <cstdio>

#include "command.h"
#include "gpu.h"
#include "net.h"
#include "vae/temporal_pad.h"

namespace
{

bool load_graph(ncnn::Net& net, const char* param_path, const char* model_path,
                ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
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
    register_seedvr2_vae_layers(net);
    return net.load_param(param_path) == 0 && net.load_model(model_path) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: test_vae_end_to_end_vulkan <encode.param> <encode.bin> <decode.param> <decode.bin>\n");
        return 2;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 1;
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net encode;
    ncnn::Net decode;
    std::fprintf(stderr, "stage=load-graphs\n");
    if (!load_graph(encode, argv[1], argv[2], vkdev, blob_allocator, staging_allocator) ||
        !load_graph(decode, argv[3], argv[4], vkdev, blob_allocator, staging_allocator))
        return 1;

    ncnn::Mat sample(128, 128, 1, 3);
    sample.fill(0.f);
    ncnn::VkMat sample_gpu;
    {
        std::fprintf(stderr, "stage=upload-sample\n");
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(sample, sample_gpu, encode.opt);
        if (upload.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Extractor encode_extractor = encode.create_extractor();
    if (encode_extractor.input("in0", sample_gpu) != 0)
        return 1;
    ncnn::VkMat latent_gpu;
    {
        std::fprintf(stderr, "stage=encode-extract\n");
        ncnn::VkCompute compute(vkdev);
        if (encode_extractor.extract("out0", latent_gpu, compute) != 0 || compute.submit_and_wait() != 0)
            return 1;
    }

    ncnn::Extractor decode_extractor = decode.create_extractor();
    if (decode_extractor.input("in0", latent_gpu) != 0)
        return 1;
    ncnn::Mat reconstruction;
    {
        std::fprintf(stderr, "stage=decode-extract\n");
        // The graph ends in Convolution3D, which is CPU-only in this ncnn
        // revision. Host extraction handles the GPU-to-CPU boundary and
        // avoids treating the CPU result as an empty VkMat.
        if (decode_extractor.extract("out0", reconstruction) != 0)
            return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (reconstruction.empty() || reconstruction.dims != 3 || reconstruction.w != 128 ||
        reconstruction.h != 128 || reconstruction.c != 3 || reconstruction.total() != 3 * 128 * 128)
    {
        std::fprintf(stderr, "unexpected Vulkan reconstruction shape\n");
        return 1;
    }

    std::puts("seedvr2-vae-end-to-end-vulkan: ok");
    return 0;
}
