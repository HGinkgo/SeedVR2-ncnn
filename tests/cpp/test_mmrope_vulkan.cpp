#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"
#include "net.h"

namespace
{

constexpr int kHeads = 1;
constexpr int kHeadDim = 8;
constexpr int kRopeDim = 6;
constexpr int kWindowVideoTokens = 45 * 40;
constexpr int kTextTokens = 3;
constexpr int kWindowTokens = kWindowVideoTokens + kTextTokens;
constexpr int kSequenceTokens = 2 * kWindowTokens;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

ncnn::ParamDict make_params()
{
    ncnn::ParamDict params;
    params.set(0, 1);
    params.set(1, 45);
    params.set(2, 80);
    params.set(3, 1);
    params.set(4, 1);
    params.set(5, 2);
    params.set(6, kTextTokens);
    params.set(7, 0);
    params.set(8, kRopeDim);
    return params;
}

ncnn::Mat make_input()
{
    ncnn::Mat input(kHeadDim, 3 * kHeads, kSequenceTokens);
    for (int token = 0; token < kSequenceTokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            for (int feature = 0; feature < kHeadDim; feature++)
                input.channel(token).row(qkv)[feature] =
                    token * 0.01f + qkv * 0.1f + feature * 0.01f;
    return input;
}

float frequency_for(int token, int feature)
{
    const int within_window = token % kWindowTokens;
    if (within_window >= kWindowVideoTokens)
        return static_cast<float>(within_window - kWindowVideoTokens);
    if (feature / 2 == 0)
        return static_cast<float>(kTextTokens);
    if (feature / 2 == 1)
        return static_cast<float>(within_window / 40);
    return static_cast<float>(within_window % 40);
}

float expected_value(const ncnn::Mat& input, int token, int qkv, int feature)
{
    const float value = input.channel(token).row(qkv)[feature];
    if (qkv == 2 || feature >= kRopeDim)
        return value;
    const float angle = frequency_for(token, feature);
    const int paired_feature = feature % 2 == 0 ? feature + 1 : feature - 1;
    const float paired = input.channel(token).row(qkv)[paired_feature];
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    return feature % 2 == 0 ? value * cosine - paired * sine : value * cosine + paired * sine;
}

void require_matches_reference(const ncnn::Mat& input, const ncnn::Mat& output)
{
    require(output.dims == 3 && output.w == kHeadDim && output.h == 3 * kHeads &&
                output.c == kSequenceTokens,
            "MMRoPE output layout");
    for (int token = 0; token < kSequenceTokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            for (int feature = 0; feature < kHeadDim; feature++)
            {
                const float actual = output.channel(token).row(qkv)[feature];
                const float expected = expected_value(input, token, qkv, feature);
                if (std::fabs(actual - expected) >= 2e-5f)
                {
                    std::fprintf(stderr, "MMRoPE mismatch token=%d qkv=%d feature=%d expected=%f actual=%f\n",
                                 token, qkv, feature, expected, actual);
                    require(false, "MMRoPE CPU reference value");
                }
            }
}

int run_vulkan(const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return -1;
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

    SeedVR2MMRoPE rope;
    if (rope.load_param(make_params()) != 0)
        return -1;
    rope.vkdev = vkdev;
    if (rope.create_pipeline(opt) != 0)
        return -1;
    {
        ncnn::VkTransfer transfer(vkdev);
        if (rope.upload_model(transfer, opt) != 0 || transfer.submit_and_wait() != 0)
            return -1;
    }

    ncnn::VkMat input_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(input, input_gpu, opt);
        if (upload.submit_and_wait() != 0)
            return -1;
    }
    std::vector<ncnn::VkMat> outputs;
    ncnn::VkCompute compute(vkdev);
    const int forward_result = rope.forward({input_gpu}, outputs, compute, opt);
    if (forward_result == 0 && outputs.size() == 1)
        compute.record_download(outputs[0], output, opt);
    const int submit_result = forward_result == 0 && outputs.size() == 1 ? compute.submit_and_wait() : -1;
    rope.destroy_pipeline(opt);
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return submit_result;
}

int run_vulkan_via_net(const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return -1;
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    int result = -1;
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
        net.set_vulkan_device(vkdev);
        register_seedvr2_awa_layers(net);

        FILE* param = std::tmpfile();
        FILE* model = std::tmpfile();
        if (!param || !model)
        {
            if (param)
                std::fclose(param);
            if (model)
                std::fclose(model);
        }
        else
        {
            std::fprintf(param,
                         "7767517\n"
                         "2 2\n"
                         "Input input 0 1 in\n"
                         "SeedVR2MMRoPE mmrope 1 1 in out "
                         "0=1 1=45 2=80 3=1 4=1 5=2 6=3 7=0 8=6\n");
            std::rewind(param);
            std::rewind(model);
            if (net.load_param(param) == 0 && net.load_model(model) == 0)
            {
                ncnn::VkMat input_gpu;
                ncnn::VkCompute upload(vkdev);
                upload.record_upload(input, input_gpu, net.opt);
                if (upload.submit_and_wait() == 0)
                {
                    ncnn::Extractor extractor = net.create_extractor();
                    extractor.set_light_mode(false);
                    ncnn::VkMat output_gpu;
                    ncnn::VkCompute compute(vkdev);
                    if (extractor.input("in", input_gpu) == 0 &&
                        extractor.extract("out", output_gpu, compute) == 0)
                    {
                        compute.record_download(output_gpu, output, net.opt);
                        result = compute.submit_and_wait();
                    }
                }
            }
            std::fclose(param);
            std::fclose(model);
        }
    }
    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    return result;
}

} // namespace

int main()
{
    const ncnn::Mat input = make_input();
    SeedVR2MMRoPE rope;
    require(rope.load_param(make_params()) == 0, "MMRoPE parameters load");
    ncnn::Option option;
    std::vector<ncnn::Mat> outputs;
    require(rope.forward({input}, outputs, option) == 0 && outputs.size() == 1,
            "MMRoPE CPU forward");
    require_matches_reference(input, outputs[0]);

    ncnn::Mat vulkan_output;
    require(run_vulkan(input, vulkan_output) == 0, "MMRoPE Vulkan forward");
    require_matches_reference(input, vulkan_output);

    ncnn::Mat network_output;
    require(run_vulkan_via_net(input, network_output) == 0, "MMRoPE Vulkan Net forward");
    require_matches_reference(input, network_output);
    for (int token = 0; token < kSequenceTokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            for (int feature = 0; feature < kHeadDim; feature++)
                require(std::fabs(outputs[0].channel(token).row(qkv)[feature] -
                                      vulkan_output.channel(token).row(qkv)[feature]) < 2e-5f,
                        "MMRoPE CPU/Vulkan parity");

    std::puts("seedvr2-mmrope-vulkan: ok");
    return 0;
}
