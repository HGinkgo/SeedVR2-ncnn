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

struct RealScaleConfig
{
    int source_t;
    int source_h;
    int source_w;
    int windows_t;
    int windows_h;
    int windows_w;
    int text_tokens;
    int heads;
    int head_dim;
    bool shifted;
};

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

ncnn::ParamDict make_params(const RealScaleConfig& config)
{
    ncnn::ParamDict params;
    params.set(0, config.source_t);
    params.set(1, config.source_h);
    params.set(2, config.source_w);
    params.set(3, config.windows_t);
    params.set(4, config.windows_h);
    params.set(5, config.windows_w);
    params.set(6, config.text_tokens);
    params.set(7, config.shifted ? 1 : 0);
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

ncnn::Mat make_real_scale_input(int sequence_tokens, const RealScaleConfig& config)
{
    ncnn::Mat input(config.head_dim, 3 * config.heads, sequence_tokens);
    for (int token = 0; token < sequence_tokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            for (int head = 0; head < config.heads; head++)
                for (int feature = 0; feature < config.head_dim; feature++)
                {
                    const int value = (token * 31 + qkv * 19 + head * 13 + feature * 7) % 997;
                    input.channel(token).row(qkv * config.heads + head)[feature] =
                        (static_cast<float>(value) - 498.f) / 997.f;
                }
    return input;
}

bool matches_real_scale_samples(const ncnn::Mat& input, const ncnn::Mat& output,
                                const RealScaleConfig& config, float tolerance)
{
    const std::vector<seedvr2::AwaWindow> windows = seedvr2::make_awa_windows(
        config.source_t, config.source_h, config.source_w, config.windows_t, config.windows_h, config.windows_w,
        config.shifted);
    std::vector<int> offsets;
    offsets.reserve(windows.size() + 1);
    offsets.push_back(0);
    for (const seedvr2::AwaWindow& window : windows)
    {
        const int video_tokens = (window.t1 - window.t0) * (window.h1 - window.h0) * (window.w1 - window.w0);
        offsets.push_back(offsets.back() + video_tokens + config.text_tokens);
    }
    if (output.dims != 3 || output.w != config.head_dim || output.h != config.heads || output.c != offsets.back())
        return false;

    const float scale = 1.f / std::sqrt(static_cast<float>(config.head_dim));
    for (size_t window_index = 0; window_index < windows.size(); window_index++)
    {
        const int window_start = offsets[window_index];
        const int window_length = offsets[window_index + 1] - window_start;
        const int query = window_start + window_length / 2;
        for (int head = 0; head < config.heads; head++)
        {
            std::vector<float> scores(static_cast<size_t>(window_length));
            float maximum = -INFINITY;
            for (int key = 0; key < window_length; key++)
            {
                float dot = 0.f;
                const float* query_data = static_cast<const float*>(input.channel(query).row(head));
                const float* key_data = static_cast<const float*>(input.channel(window_start + key).row(config.heads + head));
                for (int feature = 0; feature < config.head_dim; feature++)
                    dot += query_data[feature] * key_data[feature];
                scores[key] = dot * scale;
                maximum = std::max(maximum, scores[key]);
            }
            float normalizer = 0.f;
            for (float& score : scores)
            {
                score = std::exp(score - maximum);
                normalizer += score;
            }
            for (int feature = 0; feature < config.head_dim; feature++)
            {
                float expected = 0.f;
                for (int key = 0; key < window_length; key++)
                {
                    const float* value = static_cast<const float*>(
                        input.channel(window_start + key).row(2 * config.heads + head));
                    expected += scores[key] / normalizer * value[feature];
                }
                const float actual = output.channel(query).row(head)[feature];
                if (!std::isfinite(actual) || std::fabs(expected - actual) > tolerance)
                {
                    std::fprintf(stderr,
                                 "real-scale attention mismatch shifted=%d window=%zu token=%d head=%d feature=%d "
                                 "expected=%f actual=%f\n",
                                 config.shifted, window_index, query, head, feature, expected, actual);
                    return false;
                }
            }
        }
    }
    return true;
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

int run_vulkan(const ncnn::Mat& input, const ncnn::ParamDict& params, ncnn::Mat& output)
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
    if (attention.load_param(params) != 0)
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

int run_vulkan(const ncnn::Mat& input, ncnn::Mat& output)
{
    return run_vulkan(input, make_params(), output);
}

void verify_real_scale_attention(bool shifted)
{
    const RealScaleConfig config = {1, 45, 80, 4, 3, 3, 58, 20, 128, shifted};
    const std::vector<seedvr2::AwaWindow> windows = seedvr2::make_awa_windows(
        config.source_t, config.source_h, config.source_w, config.windows_t, config.windows_h, config.windows_w,
        config.shifted);
    int sequence_tokens = 0;
    for (const seedvr2::AwaWindow& window : windows)
        sequence_tokens += (window.t1 - window.t0) * (window.h1 - window.h0) * (window.w1 - window.w0) +
                           config.text_tokens;
    const ncnn::Mat input = make_real_scale_input(sequence_tokens, config);
    ncnn::Mat output;
    require(run_vulkan(input, make_params(config), output) == 0, "real-scale attention Vulkan forward");
    require(matches_real_scale_samples(input, output, config, 2e-3f), "real-scale attention reference values");
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

    verify_real_scale_attention(false);
    verify_real_scale_attention(true);

    std::puts("seedvr2-awa-attention-vulkan: ok");
    return 0;
}
