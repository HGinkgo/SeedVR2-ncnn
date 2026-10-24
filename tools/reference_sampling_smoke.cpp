// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

// Test-only closed-loop comparison with a separately imported CPU reference.
// Uses the production patching, DiT, timestep, CFG/Euler and unpatching code.
// Sessions are released between positive/negative FP32 evaluations to fit one
// 24 GiB GPU. No reference latent is fed back after the initial state.

#include "dit/dit_stack.h"
#include "dit/dit_stack_internal.h"
#include "sampler/sampler.h"
#include "sampler/vulkan_sampler.h"
#include "vae/static_vae_graph.h"
#include "vae/temporal_pad.h"
#include "command.h"
#include "net.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace
{
void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

struct Record
{
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};
using Records = std::map<std::string, Record>;

template <typename T> void read(std::ifstream& input, T& value)
{
    require(static_cast<bool>(input.read(reinterpret_cast<char*>(&value), sizeof(value))), "truncated fixture");
}

Records load(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open fixture");
    char magic[8];
    std::uint32_t version, count;
    read(input, magic); read(input, version); read(input, count);
    require(std::memcmp(magic, "SVR2F32\0", 8) == 0 && version == 1 && count <= 64, "invalid fixture header");
    Records records;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        std::uint16_t length;
        std::uint8_t rank, reserved;
        std::uint64_t elements, dims[4];
        read(input, length); read(input, rank); read(input, reserved); read(input, elements); read(input, dims);
        require(length > 0 && length <= 128 && rank > 0 && rank <= 4 && reserved == 0 && elements <= 1048576,
                "invalid fixture record header");
        std::uint64_t product = 1;
        for (int axis = 0; axis < 4; ++axis)
        {
            if (axis < rank)
            {
                require(dims[axis] > 0 && dims[axis] <= 1048576 / product, "invalid fixture dimensions");
                product *= dims[axis];
            }
            else require(dims[axis] == 0, "nonzero reserved fixture dimension");
        }
        require(product == elements, "fixture element count mismatch");
        std::string name(length, '\0');
        require(static_cast<bool>(input.read(name.data(), length)), "truncated fixture name");
        Record record;
        record.shape.assign(dims, dims + rank);
        record.values.resize(elements);
        require(static_cast<bool>(input.read(reinterpret_cast<char*>(record.values.data()), elements * sizeof(float))),
                "truncated fixture values");
        require(std::all_of(record.values.begin(), record.values.end(), [](float x) { return std::isfinite(x); }),
                "non-finite fixture values");
        require(records.emplace(name, std::move(record)).second, "duplicate fixture name");
    }
    require(input.peek() == std::char_traits<char>::eof(), "unexpected fixture trailing bytes");
    return records;
}

const Record& get(const Records& records, const std::string& name, const std::vector<std::uint64_t>& shape)
{
    const auto p = records.find(name);
    require(p != records.end() && p->second.shape == shape, "missing or invalid record: " + name);
    return p->second;
}

ncnn::Mat latent(const Record& record)
{
    ncnn::Mat result(32, 32, 1, 16);
    require(!result.empty(), "latent allocation failed");
    // Copy logical channels separately; do not assume channel padding.
    for (int c = 0; c < 16; ++c)
        std::copy_n(record.values.data() + c * 1024, 1024, static_cast<float*>(result.channel(c)));
    return result;
}

ncnn::Mat text(const Record& record)
{
    ncnn::Mat result(5120, static_cast<int>(record.shape[0]), sizeof(float));
    require(!result.empty(), "text allocation failed");
    std::copy(record.values.begin(), record.values.end(), static_cast<float*>(result.data));
    return result;
}

bool compare_cpu(const std::string& name, const ncnn::Mat& cpu, const Record& expected,
                 const char* prefix, double fixed_limit = -1)
{
    require(!cpu.empty() && cpu.elempack == 1 && cpu.elemsize == sizeof(float),
            "output precision contract violated: " + name);
    const bool channels = expected.shape.size() == 4;
    require(channels ? ((cpu.dims == 3 || cpu.dims == 4) && cpu.c == expected.shape[0]
                            && cpu.d == expected.shape[1] && cpu.h == expected.shape[2] && cpu.w == expected.shape[3])
                     : (expected.shape.size() == 2 && cpu.dims == 2
                            && cpu.w == expected.shape[1] && cpu.h == expected.shape[0]),
            "output shape mismatch: " + name);
    const std::size_t area = static_cast<std::size_t>(cpu.w) * cpu.h * cpu.d;
    double squared = 0, maximum = 0, peak = 0;
    for (std::size_t i = 0; i < expected.values.size(); ++i)
    {
        const float actual = channels ? static_cast<const float*>(cpu.channel(i / area))[i % area]
                                      : static_cast<const float*>(cpu.data)[i];
        require(std::isfinite(actual), "non-finite native output: " + name);
        const double difference = std::fabs(static_cast<double>(actual) - expected.values[i]);
        maximum = std::max(maximum, difference);
        squared += difference * difference;
        peak = std::max(peak, std::fabs(static_cast<double>(expected.values[i])));
    }
    const double limit = fixed_limit >= 0 ? fixed_limit : 5e-4 + 5e-5 * peak;
    std::cout << prefix << " {\"stage\":\"" << name << "\",\"max_abs\":" << maximum
              << ",\"rmse\":" << std::sqrt(squared / expected.values.size())
              << ",\"reference_peak\":" << peak << ",\"limit\":" << limit
              << ",\"pass\":" << (maximum <= limit ? "true" : "false") << "}" << std::endl;
    return maximum <= limit;
}

void save(const ncnn::Mat& value, const std::string& path)
{
    require(!std::filesystem::exists(path), "refusing to overwrite native output: " + path);
    require(!value.empty() && value.elempack == 1 && value.elemsize == sizeof(float), "invalid output precision");
    std::ofstream stream(path, std::ios::binary);
    const std::size_t area = static_cast<std::size_t>(value.w) * value.h * value.d;
    for (int c = 0; c < value.c; ++c)
        stream.write(static_cast<const char*>(value.channel(c).data), area * sizeof(float));
    require(static_cast<bool>(stream), "cannot write native output: " + path);
}

struct Context
{
    ncnn::VulkanDevice* device;
    ncnn::VkAllocator *blob, *staging;
    ncnn::Net transfer;
    explicit Context(int gpu)
    {
        require(gpu >= 0 && gpu < ncnn::get_gpu_count(), "requested Vulkan GPU is unavailable");
        device = ncnn::get_gpu_device(gpu);
        blob = device->acquire_blob_allocator();
        staging = device->acquire_staging_allocator();
        require(blob && staging, "Vulkan allocator acquisition failed");
        seedvr2::configure_dit_vulkan_net_fp32(transfer, device, blob, staging, nullptr);
    }
    ~Context()
    {
        transfer.clear();
        device->reclaim_blob_allocator(blob);
        device->reclaim_staging_allocator(staging);
    }
    ncnn::VkMat upload(const ncnn::Mat& cpu)
    {
        ncnn::VkMat gpu;
        ncnn::VkCompute compute(device);
        compute.record_upload(cpu, gpu, transfer.opt);
        require(compute.submit_and_wait() == 0 && !gpu.empty(), "upload failed");
        return gpu;
    }
    ncnn::Mat download(const ncnn::VkMat& gpu)
    {
        ncnn::Mat cpu;
        ncnn::VkCompute compute(device);
        compute.record_download(gpu, cpu, transfer.opt);
        require(compute.submit_and_wait() == 0 && !cpu.empty() && cpu.elempack == 1 && cpu.elemsize == sizeof(float),
                "download failed or precision contract violated");
        return cpu;
    }
    bool compare(const std::string& name, const ncnn::VkMat& gpu, const Record& expected)
    {
        return compare_cpu(name, download(gpu), expected, "sampling_stage");
    }
};

ncnn::Mat vae(const std::string& model, const char* stage, const ncnn::Mat& input, int gpu)
{
    const auto started = std::chrono::steady_clock::now();
    // Separate contexts release VAE allocations before DiT and DiT before
    // decode. Only CPU handoff tensors survive these scopes on one GPU.
    Context context(gpu);
    ncnn::Net net;
    seedvr2::configure_dit_vulkan_net_fp32(net, context.device, context.blob, context.staging, nullptr);
    register_seedvr2_vae_layers(net, true);
    seedvr2::PreparedVaeGraph graph;
    std::string error;
    require(seedvr2::prepare_vae_graph(std::filesystem::path(model) / (std::string("vae_") + stage),
                256, 256, 0, graph, error, std::string(stage) == "decode"), "VAE graph preparation failed: " + error);
    require((graph.param_contents.empty() ? net.load_param(graph.param_path.string().c_str())
                                        : net.load_param_mem(graph.param_contents.c_str())) == 0
                && net.load_model(graph.model_path.string().c_str()) == 0, "VAE load failed");
    ncnn::Mat output;
    {
        ncnn::Extractor extractor = net.create_extractor();
        require(extractor.input("in0", input) == 0 && extractor.extract("out0", output) == 0, "VAE execution failed");
    }
    std::cout << "image_runtime {\"stage\":\"vae_" << stage << "\",\"wall_seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << "}" << std::endl;
    return output;
}

bool run(const Records& records, const std::string& positive, const std::string& negative, int gpu,
         const ncnn::Mat* native_condition = nullptr, ncnn::Mat* final_latent = nullptr)
{
    const auto& time_record = records.at("timesteps");
    const int steps = static_cast<int>(time_record.values.size());
    require(time_record.shape == std::vector<std::uint64_t>{static_cast<std::uint64_t>(steps)} && steps >= 1 && steps <= 4,
            "only 1..4 fixed uniform-trailing steps are reviewed");
    require(time_record.values == seedvr2::uniform_trailing_timesteps(1000.f, steps), "sampling schedule differs");
    const float scale = get(records, "cfg_scale", {1}).values[0];
    require(scale > 1.f && scale <= 20.f, "this comparison requires CFG in (1,20]");
    const auto& noise = get(records, "initial_noise", {16, 1, 32, 32});
    const auto& condition = get(records, "condition", {16, 1, 32, 32});
    const auto& pos_text = get(records, "input_text_positive", {58, 5120});
    const auto& neg_text = get(records, "input_text_negative", {64, 5120});
    require(records.size() == static_cast<std::size_t>(6 + 3 * steps), "unexpected fixture records");
    for (int i = 0; i < steps; ++i)
    {
        get(records, "step_" + std::to_string(i) + "_positive", {256, 64});
        get(records, "step_" + std::to_string(i) + "_negative", {256, 64});
        get(records, "step_" + std::to_string(i) + "_latent", {16, 1, 32, 32});
    }
    seedvr2::ResolutionPlan plan;
    require(seedvr2::ResolutionPlan::from_explicit(256, 256, plan), "resolution plan failed");
    Context context(gpu);
    const ncnn::Mat text_cpu[] = {text(pos_text), text(neg_text)};
    ncnn::VkMat current = context.upload(latent(noise));
    ncnn::VkMat cond = context.upload(native_condition ? *native_condition : latent(condition));
    bool passed = true;
    for (int i = 0; i < steps; ++i)
    {
        const std::string prefix = "step_" + std::to_string(i);
        ncnn::VkMat input, prediction[2], patches, updated, next;
        require(seedvr2::make_dit_input_patches_gpu(current, cond, plan, context.device, context.blob,
                                                    context.staging, input, nullptr, nullptr, true), "input patching failed");
        float timestep;
        require(seedvr2::transform_timestep(time_record.values[i], 1000.f, 1, 32, 32, timestep), "timestep transform failed");
        for (int branch = 0; branch < 2; ++branch)
        {
            seedvr2::DitStackSession session;
            require(seedvr2::DitStackSession::open(branch == 0 ? positive : negative, plan, context.device,
                                                    context.blob, context.staging, session, nullptr, nullptr, true),
                    "DiT session open failed");
            require(session.run(input, text_cpu[branch], timestep, plan, prediction[branch]), "DiT execution failed");
            session.clear();
            passed = context.compare(prefix + (branch == 0 ? "_positive" : "_negative"), prediction[branch],
                                     records.at(prefix + (branch == 0 ? "_positive" : "_negative"))) && passed;
        }
        require(seedvr2::patch_latent_for_dit_output_gpu(current, plan, context.device, context.blob, context.staging,
                                                         patches, nullptr, nullptr, true), "sample patching failed");
        const float next_time = i + 1 < steps ? time_record.values[i + 1] : 0.f;
        require(seedvr2::apply_cfg_euler_vulkan(prediction[0], prediction[1], patches, scale,
                                               (next_time - time_record.values[i]) / 1000.f, context.device,
                                               context.blob, context.staging, updated, nullptr, true), "CFG/Euler failed");
        require(seedvr2::unpatch_dit_output_gpu(updated, plan, context.device, context.blob, context.staging,
                                                next, nullptr, nullptr, true), "latent unpatching failed");
        passed = context.compare(prefix + "_latent", next, records.at(prefix + "_latent")) && passed;
        current = std::move(next);
    }
    if (final_latent) *final_latent = context.download(current);
    return passed;
}
} // namespace

int main(int argc, char** argv)
{
    const bool image_chain = argc == 9 && std::string(argv[5]) == "--image-fixture"
                                     && std::string(argv[7]) == "--output-prefix";
    if (argc != 5 && !image_chain)
    {
        std::cerr << "usage: " << argv[0] << " <positive-stack> <negative-stack> <sampling.f32> <gpu-id>"
                  << " [--image-fixture <image.f32> --output-prefix <path>]\n";
        return 2;
    }
    try
    {
        const Records records = load(argv[3]);
        std::size_t consumed;
        const int gpu = std::stoi(argv[4], &consumed);
        require(consumed == std::strlen(argv[4]) && gpu >= 0, "invalid GPU id");
        Records image_records;
        if (image_chain)
        {
            image_records = load(argv[6]);
            require(image_records.size() == 2, "image fixture requires exactly input and reconstruction");
            get(image_records, "input", {3, 1, 256, 256});
            get(image_records, "reconstruction", {3, 1, 256, 256});
        }
        require(ncnn::create_gpu_instance() == 0, "Vulkan initialization failed");
        struct Guard { ~Guard() { ncnn::destroy_gpu_instance(); } } guard;
        const auto started = std::chrono::steady_clock::now();
        bool passed;
        if (image_chain)
        {
            ncnn::Mat input(256, 256, 1, 3);
            require(!input.empty(), "image allocation failed");
            for (int c = 0; c < 3; ++c)
                std::copy_n(image_records.at("input").values.data() + c * 65536, 65536,
                            static_cast<float*>(input.channel(c)));
            const std::string prefix = argv[8];
            ncnn::Mat condition = vae(argv[1], "encode", input, gpu), endpoint;
            passed = compare_cpu("vae_encode", condition, get(records, "condition", {16, 1, 32, 32}), "image_stage", 1e-3);
            save(condition, prefix + ".condition.f32");
            passed = run(records, argv[1], argv[2], gpu, &condition, &endpoint) && passed;
            save(endpoint, prefix + ".latent.f32");
            ncnn::Mat reconstruction = vae(argv[1], "decode", endpoint, gpu);
            passed = compare_cpu("vae_decode", reconstruction, image_records.at("reconstruction"), "image_stage", 1e-3) && passed;
            save(reconstruction, prefix + ".reconstruction.f32");
        }
        else passed = run(records, argv[1], argv[2], gpu);
        struct rusage usage {};
        require(getrusage(RUSAGE_SELF, &usage) == 0, "cannot read peak RSS");
        std::cout << "sampling_runtime {\"wall_seconds\":"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                  << ",\"peak_rss_mib\":" << usage.ru_maxrss / 1024.0 << "}" << std::endl;
        return passed ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "reference-sampling-smoke: " << error.what() << '\n';
        return 1;
    }
}
