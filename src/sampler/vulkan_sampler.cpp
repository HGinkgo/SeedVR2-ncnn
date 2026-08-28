#include "sampler/vulkan_sampler.h"

#include <cmath>
#include <cstdio>

#include "datareader.h"
#include "net.h"

namespace seedvr2
{

bool apply_cfg_v_lerp_endpoint_vulkan(const ncnn::VkMat& positive_output,
                                      const ncnn::VkMat& sample,
                                      ncnn::VulkanDevice* vkdev,
                                      ncnn::VkAllocator* blob_allocator,
                                      ncnn::VkAllocator* staging_allocator,
                                      ncnn::VkMat& endpoint_sample)
{
    if (!vkdev || !blob_allocator || !staging_allocator || positive_output.empty() || sample.empty() ||
        positive_output.elempack != 1)
        return false;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    const ncnn::VkMat* sample_input = &sample;
    ncnn::VkMat unpacked_sample;
    if (sample.elempack == 4)
    {
        ncnn::VkCompute unpack(vkdev);
        vkdev->convert_packing(sample, unpacked_sample, 1, unpack, opt);
        if (unpacked_sample.empty() || unpack.submit_and_wait() != 0)
            return false;
        sample_input = &unpacked_sample;
    }
    if (sample_input->elempack != 1 || positive_output.dims != sample_input->dims ||
        positive_output.w != sample_input->w || positive_output.h != sample_input->h ||
        positive_output.d != sample_input->d || positive_output.c != sample_input->c)
        return false;

    ncnn::Net net;
    static const char kParam[] =
        "7767517\n"
        "3 3\n"
        "Input prediction 0 1 prediction\n"
        "Input sample 0 1 sample\n"
        "BinaryOp endpoint 2 1 sample prediction endpoint 0=1\n";
    net.opt = opt;
    net.set_vulkan_device(vkdev);
    if (net.load_param_mem(kParam) != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    if (net.load_model(model_reader) != 0)
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat graph_output;
    if (extractor.input("prediction", positive_output) != 0 || extractor.input("sample", *sample_input) != 0 ||
        extractor.extract("endpoint", graph_output, compute) != 0 || compute.submit_and_wait() != 0)
        return false;

    ncnn::VkCompute clone(vkdev);
    clone.record_clone(graph_output, endpoint_sample, net.opt);
    return clone.submit_and_wait() == 0;
}

bool apply_cfg_euler_vulkan(const ncnn::VkMat& positive_output,
                            const ncnn::VkMat& negative_output,
                            const ncnn::VkMat& sample,
                            float cfg_scale,
                            float normalized_delta,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& updated_sample)
{
    if (!vkdev || !blob_allocator || !staging_allocator || positive_output.empty() || negative_output.empty() ||
        sample.empty() || positive_output.dims != negative_output.dims ||
        positive_output.w != negative_output.w || positive_output.h != negative_output.h ||
        positive_output.d != negative_output.d || positive_output.c != negative_output.c ||
        positive_output.elempack != 1 || negative_output.elempack != 1 ||
        !std::isfinite(cfg_scale) || !std::isfinite(normalized_delta))
        return false;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    const ncnn::VkMat* sample_input = &sample;
    ncnn::VkMat unpacked_sample;
    if (sample.elempack == 4)
    {
        ncnn::VkCompute unpack(vkdev);
        vkdev->convert_packing(sample, unpacked_sample, 1, unpack, opt);
        if (unpacked_sample.empty() || unpack.submit_and_wait() != 0)
            return false;
        sample_input = &unpacked_sample;
    }
    if (sample_input->elempack != 1 || positive_output.dims != sample_input->dims ||
        positive_output.w != sample_input->w || positive_output.h != sample_input->h ||
        positive_output.d != sample_input->d || positive_output.c != sample_input->c)
        return false;

    char param[1024];
    const int length = std::snprintf(
        param, sizeof(param),
        "7767517\n"
        "8 8\n"
        "Input positive 0 1 positive\n"
        "Input negative 0 1 negative\n"
        "Input sample 0 1 sample\n"
        "BinaryOp difference 2 1 positive negative difference 0=1\n"
        "BinaryOp scaled 1 1 difference scaled 0=2 1=1 2=%g\n"
        "BinaryOp guided 2 1 negative scaled guided 0=0\n"
        "BinaryOp derivative 1 1 guided derivative 0=2 1=1 2=%g\n"
        "BinaryOp updated 2 1 sample derivative updated 0=0\n",
        static_cast<double>(cfg_scale), static_cast<double>(normalized_delta));
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(param))
        return false;

    ncnn::Net net;
    net.opt = opt;
    net.set_vulkan_device(vkdev);
    if (net.load_param_mem(param) != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    if (net.load_model(model_reader) != 0)
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat graph_output;
    if (extractor.input("positive", positive_output) != 0 || extractor.input("negative", negative_output) != 0 ||
        extractor.input("sample", *sample_input) != 0 || extractor.extract("updated", graph_output, compute) != 0 ||
        compute.submit_and_wait() != 0)
        return false;

    ncnn::VkCompute clone(vkdev);
    clone.record_clone(graph_output, updated_sample, net.opt);
    return clone.submit_and_wait() == 0;
}

} // namespace seedvr2
