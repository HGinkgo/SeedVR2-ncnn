#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"

namespace
{

constexpr int kSourceT = 1;
constexpr int kSourceH = 2;
constexpr int kSourceW = 2;
constexpr int kTextTokens = 2;
constexpr int kHeads = 2;
constexpr int kHeadDim = 3;
constexpr int kWindowTokens = 1 + kTextTokens;
constexpr int kWindowCount = 4;
constexpr int kSequenceTokens = kWindowCount * kWindowTokens;

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
    params.set(0, kSourceT);
    params.set(1, kSourceH);
    params.set(2, kSourceW);
    params.set(3, 1);
    // The reference scheduler first rescales the spatial plane to 60x60.
    // Sixty partitions map the 2x2 test plane back to one token per window.
    params.set(4, 60);
    params.set(5, 60);
    params.set(6, kTextTokens);
    params.set(7, 0);
    return params;
}

ncnn::Mat make_input()
{
    ncnn::Mat input(kHeadDim, 3 * kHeads, kSequenceTokens);
    for (int token = 0; token < kSequenceTokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            for (int head = 0; head < kHeads; head++)
                for (int feature = 0; feature < kHeadDim; feature++)
                    input.channel(token).row(qkv * kHeads + head)[feature] =
                        0.01f * static_cast<float>(17 * token + 5 * qkv + 3 * head + feature + 1);
    return input;
}

ncnn::Mat attention_reference(const ncnn::Mat& input)
{
    ncnn::Mat output(kHeadDim, kHeads, kSequenceTokens);
    const float scale = 1.f / std::sqrt(static_cast<float>(kHeadDim));
    for (int window = 0; window < kWindowCount; window++)
    {
        const int offset = window * kWindowTokens;
        for (int head = 0; head < kHeads; head++)
            for (int query = 0; query < kWindowTokens; query++)
            {
                float scores[kWindowTokens];
                float max_score = -INFINITY;
                for (int key = 0; key < kWindowTokens; key++)
                {
                    float dot = 0.f;
                    for (int feature = 0; feature < kHeadDim; feature++)
                        dot += input.channel(offset + query).row(head)[feature] *
                               input.channel(offset + key).row(kHeads + head)[feature];
                    scores[key] = dot * scale;
                    max_score = std::max(max_score, scores[key]);
                }
                float total = 0.f;
                for (int key = 0; key < kWindowTokens; key++)
                {
                    scores[key] = std::exp(scores[key] - max_score);
                    total += scores[key];
                }
                for (int feature = 0; feature < kHeadDim; feature++)
                {
                    float value = 0.f;
                    for (int key = 0; key < kWindowTokens; key++)
                        value += scores[key] / total *
                                 input.channel(offset + key).row(2 * kHeads + head)[feature];
                    output.channel(offset + query).row(head)[feature] = value;
                }
            }
    }
    return output;
}

void require_matches(const ncnn::Mat& expected, const ncnn::Mat& actual, float tolerance)
{
    require(actual.dims == 3 && actual.w == kHeadDim && actual.h == kHeads &&
                actual.c == kSequenceTokens,
            "attention output layout");
    for (int token = 0; token < kSequenceTokens; token++)
        for (int head = 0; head < kHeads; head++)
            for (int feature = 0; feature < kHeadDim; feature++)
            {
                const float a = expected.channel(token).row(head)[feature];
                const float b = actual.channel(token).row(head)[feature];
                if (!std::isfinite(b) || std::fabs(a - b) > tolerance)
                {
                    std::fprintf(stderr,
                                 "attention mismatch token=%d head=%d feature=%d expected=%f actual=%f\n",
                                 token, head, feature, a, b);
                    require(false, "attention reference value");
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

    int result = -1;
    SeedVR2WindowAttention attention;
    if (attention.load_param(make_params()) != 0)
        std::fprintf(stderr, "attention Vulkan: load_param failed\n");
    else
    {
        attention.vkdev = vkdev;
        if (attention.create_pipeline(opt) != 0)
            std::fprintf(stderr, "attention Vulkan: create_pipeline failed\n");
        else
        {
            ncnn::VkMat input_gpu;
            ncnn::VkTransfer upload(vkdev);
            upload.record_upload(input, input_gpu, opt, false);
            if (upload.submit_and_wait() != 0)
                std::fprintf(stderr, "attention Vulkan: upload failed\n");
            else
            {
                std::vector<ncnn::VkMat> outputs;
                ncnn::VkCompute compute(vkdev);
                const int forward_result = attention.forward({input_gpu}, outputs, compute, opt);
                if (forward_result != 0 || outputs.size() != 1)
                    std::fprintf(stderr, "attention Vulkan: forward failed ret=%d outputs=%zu\n",
                                 forward_result, outputs.size());
                else
                {
                    compute.record_download(outputs[0], output, opt);
                    result = compute.submit_and_wait();
                    if (result != 0)
                        std::fprintf(stderr, "attention Vulkan: submit failed ret=%d\n", result);
                }
            }
            attention.destroy_pipeline(opt);
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
    const ncnn::Mat expected = attention_reference(input);

    SeedVR2WindowAttention attention;
    require(attention.load_param(make_params()) == 0, "attention parameters load");
    ncnn::Option option;
    std::vector<ncnn::Mat> outputs;
    require(attention.forward({input}, outputs, option) == 0 && outputs.size() == 1,
            "attention CPU forward");
    require_matches(expected, outputs[0], 1e-5f);

    ncnn::Mat vulkan_output;
    require(run_vulkan(input, vulkan_output) == 0, "attention Vulkan forward");
    require_matches(expected, vulkan_output, 2e-5f);

    std::puts("seedvr2-awa-attention-vulkan: ok");
    return 0;
}
