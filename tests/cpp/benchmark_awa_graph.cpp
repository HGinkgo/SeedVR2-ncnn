#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "awa/awa_benchmark.h"
#include "awa/awa_layers.h"
#include "command.h"
#include "gpu.h"
#include "net.h"

namespace
{

void fill_inputs(const SeedVR2AwaBenchmarkShape& shape, ncnn::Mat& video, ncnn::Mat& text)
{
    const int source_tokens = shape.source_t * shape.source_h * shape.source_w;
    video.create(shape.head_dim, shape.heads, 3, source_tokens);
    text.create(shape.head_dim, shape.heads, 3, shape.text_tokens);
    for (int token = 0; token < source_tokens; token++)
    {
        float* data = static_cast<float*>(video.channel(token).data);
        for (int value = 0; value < 3 * shape.heads * shape.head_dim; value++)
            data[value] = static_cast<float>((token * 17 + value) % 101) / 100.f;
    }
    for (int token = 0; token < shape.text_tokens; token++)
    {
        float* data = static_cast<float*>(text.channel(token).data);
        for (int value = 0; value < 3 * shape.heads * shape.head_dim; value++)
            data[value] = static_cast<float>((token * 23 + value) % 79) / 79.f;
    }
}

bool finite_logical(const ncnn::Mat& value)
{
    for (int batch = 0; batch < value.n; batch++)
    {
        const ncnn::Mat batch_value = value.batch(batch);
        for (int channel = 0; channel < value.c; channel++)
        {
            const ncnn::Mat channel_value = batch_value.channel(channel);
            for (int depth = 0; depth < (value.dims == 4 ? value.d : 1); depth++)
            {
                const ncnn::Mat depth_value = value.dims == 4 ? channel_value.depth(depth) : channel_value;
                for (int row = 0; row < value.h; row++)
                    for (int col = 0; col < value.w; col++)
                        if (!std::isfinite(depth_value.row(row)[col]))
                            return false;
            }
        }
    }
    return true;
}

bool valid_outputs(const SeedVR2AwaBenchmarkShape& shape, const ncnn::Mat& video, const ncnn::Mat& text)
{
    const bool video_shape = video.dims == 4 && video.w == shape.head_dim && video.h == shape.heads &&
                             video.d == shape.source_w && video.c == shape.source_h && video.n == shape.source_t;
    const bool text_shape = text.dims == 3 && text.w == shape.head_dim && text.h == shape.heads &&
                            text.c == shape.text_tokens;
    return video_shape && text_shape && finite_logical(video) && finite_logical(text);
}

bool load_graph(ncnn::Net& net, const char* param_path, const char* bin_path)
{
    register_seedvr2_awa_layers(net);
    if (net.load_param(param_path) != 0 || net.load_model(bin_path) != 0)
    {
        std::fprintf(stderr, "failed to load graph\n");
        return false;
    }
    return true;
}

int run_cpu(const SeedVR2AwaBenchmarkShape& shape, ncnn::Net& net, const ncnn::Mat& video,
            const ncnn::Mat& text, bool validate)
{
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(true);
    const int video_input_ret = extractor.input("in0", video);
    const int text_input_ret = extractor.input("in1", text);
    if (video_input_ret != 0 || text_input_ret != 0)
    {
        std::fprintf(stderr, "cpu input failed: in0=%d in1=%d\n", video_input_ret, text_input_ret);
        return -1;
    }

    ncnn::Mat output_video;
    ncnn::Mat output_text;
    const int video_output_ret = extractor.extract("out0", output_video);
    const int text_output_ret = extractor.extract("out1", output_text);
    if (video_output_ret != 0 || text_output_ret != 0)
    {
        std::fprintf(stderr, "cpu extract failed: out0=%d out1=%d\n", video_output_ret, text_output_ret);
        return -1;
    }
    if (validate && !valid_outputs(shape, output_video, output_text))
    {
        std::fprintf(stderr, "cpu output validation failed\n");
        return -1;
    }
    return 0;
}

struct VulkanTiming
{
    double compute_ms = 0.0;
    double download_ms = 0.0;
};

int run_vulkan(const SeedVR2AwaBenchmarkShape& shape, ncnn::Net& net, ncnn::VulkanDevice* vkdev,
               const ncnn::VkMat& video, const ncnn::VkMat& text, bool validate, VulkanTiming& timing)
{
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", video) != 0 || extractor.input("in1", text) != 0)
        return -1;

    ncnn::VkMat output_video_gpu;
    ncnn::VkMat output_text_gpu;
    ncnn::VkCompute compute(vkdev);
    const auto compute_start = std::chrono::steady_clock::now();
    if (extractor.extract("out0", output_video_gpu, compute) != 0 ||
        extractor.extract("out1", output_text_gpu, compute) != 0)
        return -1;
    if (compute.submit_and_wait() != 0)
        return -1;
    const auto compute_end = std::chrono::steady_clock::now();

    ncnn::Mat output_video;
    ncnn::Mat output_text;
    ncnn::VkCompute download(vkdev);
    const auto download_start = std::chrono::steady_clock::now();
    download.record_download(output_video_gpu, output_video, net.opt);
    download.record_download(output_text_gpu, output_text, net.opt);
    if (download.submit_and_wait() != 0)
        return -1;
    const auto download_end = std::chrono::steady_clock::now();
    timing.compute_ms = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
    timing.download_ms = std::chrono::duration<double, std::milli>(download_end - download_start).count();
    if (validate && !valid_outputs(shape, output_video, output_text))
        return -1;
    return 0;
}

bool measure_cpu(const SeedVR2AwaBenchmarkShape& shape, ncnn::Net& net, const ncnn::Mat& video,
                 const ncnn::Mat& text, int warmup, int runs, double& median_ms)
{
    for (int index = 0; index < warmup; index++)
    {
        if (run_cpu(shape, net, video, text, true) != 0)
        {
            std::fprintf(stderr, "cpu warmup failed\n");
            return false;
        }
    }

    std::vector<double> samples;
    samples.reserve(runs);
    for (int index = 0; index < runs; index++)
    {
        const auto start = std::chrono::steady_clock::now();
        const int ret = run_cpu(shape, net, video, text, false);
        const auto end = std::chrono::steady_clock::now();
        if (ret != 0)
        {
            std::fprintf(stderr, "cpu measurement failed\n");
            return false;
        }
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    median_ms = seedvr2_awa_median_ms(samples);
    return true;
}

bool measure_vulkan(const SeedVR2AwaBenchmarkShape& shape, ncnn::Net& net, ncnn::VulkanDevice* vkdev,
                    const ncnn::VkMat& video, const ncnn::VkMat& text, int warmup, int runs,
                    double& compute_median_ms, double& download_median_ms, double& total_median_ms)
{
    for (int index = 0; index < warmup; index++)
    {
        VulkanTiming timing;
        if (run_vulkan(shape, net, vkdev, video, text, true, timing) != 0)
        {
            std::fprintf(stderr, "vulkan warmup failed\n");
            return false;
        }
    }

    std::vector<double> compute_samples;
    std::vector<double> download_samples;
    std::vector<double> total_samples;
    compute_samples.reserve(runs);
    download_samples.reserve(runs);
    total_samples.reserve(runs);
    for (int index = 0; index < runs; index++)
    {
        VulkanTiming timing;
        if (run_vulkan(shape, net, vkdev, video, text, false, timing) != 0)
        {
            std::fprintf(stderr, "vulkan measurement failed\n");
            return false;
        }
        compute_samples.push_back(timing.compute_ms);
        download_samples.push_back(timing.download_ms);
        total_samples.push_back(timing.compute_ms + timing.download_ms);
    }
    compute_median_ms = seedvr2_awa_median_ms(compute_samples);
    download_median_ms = seedvr2_awa_median_ms(download_samples);
    total_median_ms = seedvr2_awa_median_ms(total_samples);
    return true;
}

bool parse_positive_int(const char* value, int& parsed)
{
    char* end = 0;
    const long result = std::strtol(value, &end, 10);
    if (!value[0] || !end || *end || result < 1 || result > 10000)
        return false;
    parsed = static_cast<int>(result);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 6)
    {
        std::fprintf(stderr,
                     "usage: seedvr2-awa-graph-benchmark <param> <bin> [t,h,w,heads,head_dim,text] [warmup] [runs]\n");
        return 2;
    }

    SeedVR2AwaBenchmarkShape shape = {2, 19, 23, 2, 3, 5};
    int next_argument = 3;
    if (argc > next_argument && std::strchr(argv[next_argument], ','))
    {
        if (!seedvr2_awa_parse_shape(argv[next_argument], shape))
        {
            std::fprintf(stderr, "shape must be t,h,w,heads,head_dim,text with positive integers\n");
            return 2;
        }
        next_argument++;
    }
    if (argc > next_argument + 2)
    {
        std::fprintf(stderr, "too many benchmark arguments\n");
        return 2;
    }
    int warmup = 5;
    int runs = 25;
    if ((argc > next_argument && !parse_positive_int(argv[next_argument], warmup)) ||
        (argc > next_argument + 1 && !parse_positive_int(argv[next_argument + 1], runs)))
    {
        std::fprintf(stderr, "warmup and runs must be integers in [1, 10000]\n");
        return 2;
    }

    ncnn::Mat video;
    ncnn::Mat text;
    fill_inputs(shape, video, text);

    ncnn::Net cpu_net;
    cpu_net.opt.use_packing_layout = false;
    cpu_net.opt.num_threads = 1;
    if (!load_graph(cpu_net, argv[1], argv[2]))
        return 1;

    double cpu_median_ms = 0.0;
    if (!measure_cpu(shape, cpu_net, video, text, warmup, runs, cpu_median_ms))
        return 1;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device\n");
        return 1;
    }
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Net vulkan_net;
    vulkan_net.opt.use_vulkan_compute = true;
    vulkan_net.opt.use_packing_layout = false;
    vulkan_net.opt.use_fp16_packed = false;
    vulkan_net.opt.use_fp16_storage = false;
    vulkan_net.opt.use_fp16_arithmetic = false;
    vulkan_net.opt.blob_vkallocator = blob_allocator;
    vulkan_net.opt.workspace_vkallocator = blob_allocator;
    vulkan_net.opt.staging_vkallocator = staging_allocator;
    if (!load_graph(vulkan_net, argv[1], argv[2]))
        return 1;

    double vulkan_compute_median_ms = 0.0;
    double vulkan_download_median_ms = 0.0;
    double vulkan_total_median_ms = 0.0;
    {
        ncnn::VkMat video_gpu;
        ncnn::VkMat text_gpu;
        ncnn::VkCompute upload(vkdev);
        upload.record_upload(video, video_gpu, vulkan_net.opt);
        upload.record_upload(text, text_gpu, vulkan_net.opt);
        if (upload.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "Vulkan input upload failed\n");
            return 1;
        }
        if (!measure_vulkan(shape, vulkan_net, vkdev, video_gpu, text_gpu, warmup, runs,
                            vulkan_compute_median_ms, vulkan_download_median_ms, vulkan_total_median_ms))
            return 1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    std::printf("benchmark=awa-fixed-graph\n");
    std::printf("shape=t%d,h%d,w%d,heads%d,head_dim%d,text%d\n", shape.source_t, shape.source_h, shape.source_w,
                shape.heads, shape.head_dim, shape.text_tokens);
    std::printf("warmup=%d runs=%d\n", warmup, runs);
    std::printf("backend=cpu median_ms=%.3f\n", cpu_median_ms);
    std::printf("backend=vulkan_compute median_ms=%.3f\n", vulkan_compute_median_ms);
    std::printf("backend=vulkan_download_sync median_ms=%.3f\n", vulkan_download_median_ms);
    std::printf("backend=vulkan_total median_ms=%.3f\n", vulkan_total_median_ms);
    std::printf("speedup=%.3fx\n", cpu_median_ms / vulkan_total_median_ms);
    return 0;
}
