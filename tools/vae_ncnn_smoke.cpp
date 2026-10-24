// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 SeedVR2-ncnn contributors

// Validate the packaged VAE encode/decode graphs against the portable
// float32 reference emitted by tools/reference/seedvr2_baseline.py.

#include "net.h"
#include "mat.h"
#include "platform.h"
#include "vae/static_vae_graph.h"
#include "vae/temporal_pad.h"

#if NCNN_VULKAN
#include "allocator.h"
#include "gpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace
{

struct TensorRecord final
{
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

template <typename T>
bool read_scalar(std::ifstream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(stream);
}

bool read_golden(const std::string& path, std::map<std::string, TensorRecord>& records)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        std::cerr << "cannot open golden file: " << path << "\n";
        return false;
    }
    char magic[8] = {};
    std::uint32_t version = 0;
    std::uint32_t record_count = 0;
    if (!stream.read(magic, sizeof(magic)) || !read_scalar(stream, version) || !read_scalar(stream, record_count) ||
        std::memcmp(magic, "SVR2F32\0", sizeof(magic)) != 0 || version != 1)
    {
        std::cerr << "invalid portable golden header\n";
        return false;
    }
    for (std::uint32_t index = 0; index < record_count; ++index)
    {
        std::uint16_t name_length = 0;
        std::uint8_t rank = 0;
        std::uint8_t reserved = 0;
        std::uint64_t element_count = 0;
        std::uint64_t dimensions[4] = {};
        if (!read_scalar(stream, name_length) || !read_scalar(stream, rank) || !read_scalar(stream, reserved) ||
            !read_scalar(stream, element_count) || !stream.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions)) ||
            rank > 4)
            return false;
        std::string name(name_length, '\0');
        if (!stream.read(name.data(), static_cast<std::streamsize>(name.size())) || name.empty() ||
            records.find(name) != records.end())
            return false;
        TensorRecord record;
        record.shape.assign(dimensions, dimensions + rank);
        record.values.resize(static_cast<std::size_t>(element_count));
        if (element_count > 0 &&
            !stream.read(reinterpret_cast<char*>(record.values.data()),
                         static_cast<std::streamsize>(element_count * sizeof(float))))
            return false;
        records.emplace(std::move(name), std::move(record));
    }
    return true;
}

const TensorRecord* find_record(const std::map<std::string, TensorRecord>& records, const char* name)
{
    const auto iterator = records.find(name);
    return iterator == records.end() ? nullptr : &iterator->second;
}

bool mat_shape(const TensorRecord& record, std::uint64_t channels, std::uint64_t depth,
               std::uint64_t height, std::uint64_t width)
{
    return record.shape == std::vector<std::uint64_t>{channels, depth, height, width} &&
           record.values.size() == static_cast<std::size_t>(channels * depth * height * width);
}

bool fill_mat(const TensorRecord& record, ncnn::Mat& output)
{
    if (record.shape.size() != 4 || record.shape[0] == 0 || record.shape[1] == 0 || record.shape[2] == 0 ||
        record.shape[3] == 0 || record.shape[0] > static_cast<std::uint64_t>(INT32_MAX) ||
        record.shape[1] > static_cast<std::uint64_t>(INT32_MAX) || record.shape[2] > static_cast<std::uint64_t>(INT32_MAX) ||
        record.shape[3] > static_cast<std::uint64_t>(INT32_MAX))
        return false;
    const int channels = static_cast<int>(record.shape[0]);
    const int depth = static_cast<int>(record.shape[1]);
    const int height = static_cast<int>(record.shape[2]);
    const int width = static_cast<int>(record.shape[3]);
    output.create(width, height, depth, channels, sizeof(float));
    if (output.empty() || output.total() != record.values.size())
        return false;
    std::memcpy(output.data, record.values.data(), record.values.size() * sizeof(float));
    return true;
}

bool compare(const ncnn::Mat& actual, const TensorRecord& expected, const char* name, float limit)
{
    const std::vector<std::uint64_t> actual_shape = {
        static_cast<std::uint64_t>(actual.c), static_cast<std::uint64_t>(actual.d),
        static_cast<std::uint64_t>(actual.h), static_cast<std::uint64_t>(actual.w)};
    if (actual.elemsize != sizeof(float) || actual.elempack != 1 || actual_shape != expected.shape ||
        static_cast<std::size_t>(actual.w) * actual.h * actual.d * actual.c != expected.values.size())
    {
        std::cerr << name << " shape mismatch actual=" << actual.w << "x" << actual.h << "x" << actual.d << "x"
                  << actual.c << " expected=";
        for (std::uint64_t dimension : expected.shape)
            std::cerr << dimension << " ";
        std::cerr << "\n";
        return false;
    }
    const std::size_t plane = static_cast<std::size_t>(actual.w) * actual.h * actual.d;
    float max_abs = 0.f;
    float max_rel = 0.f;
    long double sum_squared = 0.0;
    std::uint64_t fingerprint = UINT64_C(14695981039346656037);
    for (std::size_t index = 0; index < expected.values.size(); ++index)
    {
        const float value = static_cast<const float*>(actual.channel(index / plane).data)[index % plane];
        if (!std::isfinite(value) || !std::isfinite(expected.values[index]))
            return false;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int shift = 0; shift < 32; shift += 8)
        {
            fingerprint ^= (bits >> shift) & 0xffu;
            fingerprint *= UINT64_C(1099511628211);
        }
        const float difference = std::fabs(value - expected.values[index]);
        max_abs = std::max(max_abs, difference);
        max_rel = std::max(max_rel, difference / std::max(1.0e-12f, std::fabs(expected.values[index])));
        sum_squared += static_cast<long double>(difference) * difference;
    }
    const double rmse = expected.values.empty() ? 0.0 : std::sqrt(static_cast<double>(sum_squared / expected.values.size()));
    std::cout << name << " max_abs=" << std::setprecision(9) << max_abs << " max_rel=" << max_rel
              << " rmse=" << rmse << " fnv1a64=" << std::hex << fingerprint << std::dec << "\n";
    return std::isfinite(max_abs) && max_abs <= limit;
}

bool load_graph(ncnn::Net& net, const seedvr2::PreparedVaeGraph& graph)
{
    register_seedvr2_vae_layers(net, net.opt.use_vulkan_compute &&
                               !net.opt.use_bf16_storage && !net.opt.use_fp16_storage);
    const int param_result = graph.param_contents.empty()
                                 ? net.load_param(graph.param_path.string().c_str())
                                 : net.load_param_mem(graph.param_contents.c_str());
    return param_result == 0 && net.load_model(graph.model_path.string().c_str()) == 0;
}

#if NCNN_VULKAN
struct GpuInstanceGuard final
{
    bool active = false;

    ~GpuInstanceGuard()
    {
        if (active)
            ncnn::destroy_gpu_instance();
    }
};

// No checkpoint: compare ordinary Vulkan Conv3D with a double-precision
// oracle, covering asymmetric padding/stride/dilation and cancellation.
bool check_conv3d(int gpu_id)
{
    auto* device = ncnn::get_gpu_device(gpu_id);
    auto* blob = device->acquire_blob_allocator();
    auto* staging = device->acquire_staging_allocator();
    bool passed = blob && staging;
    for (int test = 0; passed && test < 2; ++test)
    {
        const int w = test ? 1 : 7, h = test ? 1 : 6, d = test ? 1 : 5;
        const int channels = test ? 314 : 5, outputs = test ? 1 : 3;
        const int kw = test ? 1 : 3, kh = test ? 1 : 2, kd = test ? 1 : 2;
        const int dw = test ? 1 : 2, dd = test ? 1 : 2, stride = test ? 1 : 2;
        const int left = test ? 0 : 1, right = test ? 0 : 2, top = test ? 0 : 1;
        const int front = test ? 0 : 2, behind = test ? 0 : 1;
        const int ow = (w + left + right - dw * (kw - 1) - 1) / stride + 1;
        const int oh = (h + top - kh) / stride + 1;
        const int od = d + front + behind - dd * (kd - 1);
        ncnn::Mat input(w, h, d, channels, 4u);
        ncnn::Mat weights[] = {ncnn::Mat(outputs * channels * kw * kh * kd), ncnn::Mat(outputs)};
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < w * h * d; ++i)
                static_cast<float*>(input.channel(c).data)[i] = test
                    ? (c < 100 ? 10000.f : c < 214 ? 1.f : -10000.f)
                    : static_cast<float>((i * 17 + c * 13) % 31 - 15) / 7.f;
        for (int i = 0; i < weights[0].w; ++i)
            weights[0][i] = test ? 1.f / 314.f : static_cast<float>(static_cast<int>(i * 11 % 29) - 14) / 19.f;
        weights[1].fill(test ? 0.f : .125f);
        TensorRecord expected{{static_cast<std::uint64_t>(outputs), static_cast<std::uint64_t>(od),
                               static_cast<std::uint64_t>(oh), static_cast<std::uint64_t>(ow)}, {}};
        for (int o = 0; o < outputs; ++o)
            for (int z = 0; z < od; ++z)
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x)
                    {
                        double sum = weights[1][o];
                        for (int c = 0; c < channels; ++c)
                            for (int kz = 0; kz < kd; ++kz)
                                for (int ky = 0; ky < kh; ++ky)
                                    for (int kx = 0; kx < kw; ++kx)
                                    {
                                        const int iz = z + kz * dd - front;
                                        const int iy = y * stride + ky - top;
                                        const int ix = x * stride + kx * dw - left;
                                        if (iz >= 0 && iz < d && iy >= 0 && iy < h && ix >= 0 && ix < w)
                                        {
                                            const int wi = (((o * channels + c) * kd + kz) * kh + ky) * kw + kx;
                                            sum += static_cast<double>(input.channel(c).depth(iz).row(iy)[ix]) * weights[0][wi];
                                        }
                                    }
                        expected.values.push_back(static_cast<float>(sum));
                    }
        ncnn::Net net;
        net.set_vulkan_device(device);
        net.opt.use_vulkan_compute = true;
        net.opt.use_packing_layout = false;
        net.opt.use_bf16_packed = net.opt.use_bf16_storage = false;
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = false;
        net.opt.blob_vkallocator = net.opt.workspace_vkallocator = blob;
        net.opt.staging_vkallocator = staging;
        register_seedvr2_vae_layers(net, true);
        const std::string param = "7767517\n2 2\nInput input 0 1 in\nConvolution3D conv 1 1 in out 0=" +
            std::to_string(outputs) + " 1=" + std::to_string(kw) + " 11=" + std::to_string(kh) +
            " 21=" + std::to_string(kd) + " 2=" + std::to_string(dw) + " 12=1 22=" + std::to_string(dd) +
            " 3=" + std::to_string(stride) + " 13=" + std::to_string(stride) + " 23=1 4=" +
            std::to_string(left) + " 15=" + std::to_string(right) + " 14=" + std::to_string(top) +
            " 16=0 24=" + std::to_string(front) + " 17=" + std::to_string(behind) +
            " 5=1 6=" + std::to_string(weights[0].w) + "\n";
        // ncnn FP32 weight tag, followed by contiguous weights and bias.
        std::vector<unsigned char> model(4 + (weights[0].w + outputs) * sizeof(float), 0);
        std::memcpy(model.data() + 4, weights[0].data, weights[0].w * sizeof(float));
        std::memcpy(model.data() + 4 + weights[0].w * sizeof(float), weights[1].data, outputs * sizeof(float));
        const int param_status = net.load_param_mem(param.c_str());
        const std::size_t consumed = param_status == 0 ? net.load_model(model.data()) : 0;
        passed = param_status == 0 && consumed == model.size();
        if (!passed)
            std::cerr << "conv3d contract load failed param=" << param_status << " model_bytes=" << consumed
                      << " expected=" << model.size() << "\n";
        if (passed)
        {
            ncnn::Mat actual;
            auto ex = net.create_extractor();
            passed = ex.input("in", input) == 0 && ex.extract("out", actual) == 0 &&
                     compare(actual, expected, test ? "conv3d_cancellation_fp64" : "conv3d_geometry_fp64", 2e-5f);
        }
    }
    if (blob) device->reclaim_blob_allocator(blob);
    if (staging) device->reclaim_staging_allocator(staging);
    return passed;
}
#endif

void usage(const char* program)
{
    std::cerr << "usage: " << program
              << " <model-dir> <vae_golden.f32> [cpu|vulkan] [gpu-id] [--max-abs value]"
                 " [--max-encode-abs value] [--max-decode-abs value] [--repeat count] [--fp32]\n";
}

bool parse_nonnegative_int(const char* text, int& value)
{
    try
    {
        std::size_t consumed = 0;
        value = std::stoi(text, &consumed);
        return consumed == std::string(text).size() && value >= 0;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_nonnegative_float(const char* text, float& value)
{
    try
    {
        std::size_t consumed = 0;
        value = std::stof(text, &consumed);
        return consumed == std::string(text).size() && std::isfinite(value) && value >= 0.f;
    }
    catch (...)
    {
        return false;
    }
}

} // namespace

int main(int argc, char** argv)
{
#if NCNN_VULKAN
    if (argc >= 2 && std::string(argv[1]) == "--check-conv3d")
    {
        int gpu = 0;
        if (argc > 3 || (argc == 3 && !parse_nonnegative_int(argv[2], gpu))) return 2;
        GpuInstanceGuard guard;
        if (ncnn::create_gpu_instance() != 0) return 1;
        guard.active = true;
        return gpu < ncnn::get_gpu_count() && check_conv3d(gpu) ? 0 : 1;
    }
#endif
    if (argc < 3)
    {
        usage(argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string golden_path = argv[2];
    std::string mode = "cpu";
    int gpu_id = 0;
    // Vulkan follows the production BF16-storage path.  Compare against the
    // FP32 reference with stage-specific absolute-error limits; callers can
    // tighten these with --max-encode-abs/--max-decode-abs.
    float max_encode_abs_limit = 1.0e-1f;
    float max_decode_abs_limit = 2.5e-1f;
    int repeat_count = 1;
    bool fp32_storage = false;
    bool gpu_id_seen = false;
#if NCNN_VULKAN
    GpuInstanceGuard gpu_instance_guard;
#endif
    int argument_index = 3;
    while (argument_index < argc)
    {
        const std::string argument = argv[argument_index];
        if (argument == "--fp32")
        {
            fp32_storage = true;
            ++argument_index;
        }
        else if (argument == "--repeat")
        {
            if (argument_index + 1 >= argc || !parse_nonnegative_int(argv[argument_index + 1], repeat_count) ||
                repeat_count < 1)
            {
                usage(argv[0]);
                return 2;
            }
            argument_index += 2;
        }
        else if (argument == "--max-abs")
        {
            float value = 0.f;
            if (argument_index + 1 >= argc || !parse_nonnegative_float(argv[argument_index + 1], value))
            {
                usage(argv[0]);
                return 2;
            }
            max_encode_abs_limit = value;
            max_decode_abs_limit = value;
            argument_index += 2;
        }
        else if (argument == "--max-encode-abs")
        {
            if (argument_index + 1 >= argc || !parse_nonnegative_float(argv[argument_index + 1], max_encode_abs_limit))
            {
                usage(argv[0]);
                return 2;
            }
            argument_index += 2;
        }
        else if (argument == "--max-decode-abs")
        {
            if (argument_index + 1 >= argc || !parse_nonnegative_float(argv[argument_index + 1], max_decode_abs_limit))
            {
                usage(argv[0]);
                return 2;
            }
            argument_index += 2;
        }
        else if (argument == "cpu" || argument == "vulkan")
        {
            mode = argument;
            ++argument_index;
        }
        else if (!gpu_id_seen && parse_nonnegative_int(argument.c_str(), gpu_id))
        {
            gpu_id_seen = true;
            ++argument_index;
        }
        else
        {
            usage(argv[0]);
            return 2;
        }
    }
    if (fp32_storage && mode != "vulkan")
    {
        std::cerr << "--fp32 selects Vulkan storage; CPU already uses FP32\n";
        return 2;
    }
    if (mode == "vulkan")
    {
#if !NCNN_VULKAN
        std::cerr << "this runner was built without NCNN_VULKAN\n";
        return 1;
#else
        if (ncnn::create_gpu_instance() != 0 || gpu_id >= ncnn::get_gpu_count())
        {
            std::cerr << "requested Vulkan device is unavailable: " << gpu_id << "\n";
            return 1;
        }
        gpu_instance_guard.active = true;
#endif
    }

    std::map<std::string, TensorRecord> records;
    if (!read_golden(golden_path, records))
        return 1;
    const TensorRecord* input_record = find_record(records, "input");
    const TensorRecord* latent_record = find_record(records, "latent");
    const TensorRecord* decode_latent_record = find_record(records, "decode_latent");
    const TensorRecord* reconstruction_record = find_record(records, "reconstruction");
    if (!input_record || !latent_record || !reconstruction_record || !mat_shape(*input_record, 3, 1, 256, 256) ||
        !mat_shape(*latent_record, 16, 1, 32, 32) || !mat_shape(*reconstruction_record, 3, 1, 256, 256))
    {
        std::cerr << "golden records do not match the fixed 256x256 VAE contract\n";
        return 1;
    }
    if (decode_latent_record && !mat_shape(*decode_latent_record, 16, 1, 32, 32))
    {
        std::cerr << "decode_latent does not match the fixed 256x256 VAE contract\n";
        return 1;
    }

    ncnn::Mat input;
    if (!fill_mat(*input_record, input))
        return 1;
    seedvr2::PreparedVaeGraph encode_graph;
    seedvr2::PreparedVaeGraph decode_graph;
    std::string error;
    const std::filesystem::path root(model_dir);
    if (!seedvr2::prepare_vae_graph(root / "vae_encode", 256, 256, 0, encode_graph, error) ||
        !seedvr2::prepare_vae_graph(root / "vae_decode", 256, 256, 0, decode_graph, error, true))
    {
        std::cerr << "failed to prepare VAE graph: " << error << "\n";
        return 1;
    }

    ncnn::Net encode;
    ncnn::Net decode;
#if NCNN_VULKAN
    ncnn::VkAllocator* encode_blob = nullptr;
    ncnn::VkAllocator* encode_staging = nullptr;
    ncnn::VkAllocator* decode_blob = nullptr;
    ncnn::VkAllocator* decode_staging = nullptr;
    ncnn::VulkanDevice* vkdev = nullptr;
    if (mode == "vulkan")
    {
        vkdev = ncnn::get_gpu_device(gpu_id);
        encode_blob = vkdev->acquire_blob_allocator();
        encode_staging = vkdev->acquire_staging_allocator();
        decode_blob = vkdev->acquire_blob_allocator();
        decode_staging = vkdev->acquire_staging_allocator();
        for (ncnn::VkAllocator* allocator : {encode_blob, encode_staging, decode_blob, decode_staging})
            if (!allocator)
                return 1;
        encode.opt.use_vulkan_compute = true;
        decode.opt.use_vulkan_compute = true;
        encode.opt.use_packing_layout = false;
        decode.opt.use_packing_layout = false;
        // Default matches production. FP32 isolates graph accuracy from
        // activation-storage rounding without changing packaged weights.
        encode.opt.use_bf16_packed = !fp32_storage;
        decode.opt.use_bf16_packed = !fp32_storage;
        encode.opt.use_bf16_storage = !fp32_storage;
        decode.opt.use_bf16_storage = !fp32_storage;
        encode.opt.use_fp16_packed = false;
        decode.opt.use_fp16_packed = false;
        encode.opt.use_fp16_storage = false;
        decode.opt.use_fp16_storage = false;
        encode.opt.use_fp16_arithmetic = false;
        decode.opt.use_fp16_arithmetic = false;
        encode.opt.blob_vkallocator = encode_blob;
        encode.opt.workspace_vkallocator = encode_blob;
        encode.opt.staging_vkallocator = encode_staging;
        decode.opt.blob_vkallocator = decode_blob;
        decode.opt.workspace_vkallocator = decode_blob;
        decode.opt.staging_vkallocator = decode_staging;
        encode.set_vulkan_device(vkdev);
        decode.set_vulkan_device(vkdev);
    }
#endif
    if (!load_graph(encode, encode_graph) || !load_graph(decode, decode_graph))
    {
        std::cerr << "failed to load VAE graph\n";
        return 1;
    }

    ncnn::Mat latent;
    {
        ncnn::Extractor extractor = encode.create_extractor();
        if (extractor.input("in0", input) != 0 || extractor.extract("out0", latent) != 0)
        {
            std::cerr << "VAE encode failed\n";
            return 1;
        }
    }
    const bool encode_ok = compare(latent, *latent_record, "vae_encode", max_encode_abs_limit);

    // A full sampling golden decodes the CFG/Euler endpoint, which differs
    // from the encoded condition. Legacy reconstruction fixtures omit this
    // record and continue decoding this runner's own encoder output.
    ncnn::Mat decode_input = latent;
    if (decode_latent_record && !fill_mat(*decode_latent_record, decode_input))
        return 1;
    std::cout << "vae_decode_input=" << (decode_latent_record ? "golden_sampler_endpoint" : "native_encoder") << "\n";
    ncnn::Mat reconstruction;
    {
        ncnn::Extractor extractor = decode.create_extractor();
        const int input_result = extractor.input("in0", decode_input);
        const int extract_result = input_result == 0 ? extractor.extract("out0", reconstruction) : -1;
        if (input_result != 0 || extract_result != 0)
        {
            std::cerr << "VAE decode failed input_result=" << input_result
                      << " extract_result=" << extract_result << " latent_dims=" << decode_input.dims
                      << " latent_shape=" << decode_input.w << "x" << decode_input.h << "x" << decode_input.d << "x" << decode_input.c
                      << " latent_elemsize=" << decode_input.elemsize << "\n";
            return 1;
        }
    }
    const char* decode_name = decode_latent_record ? "vae_decode_from_sampler_endpoint" : "vae_decode_from_native_latent";
    const bool decode_ok = compare(reconstruction, *reconstruction_record, decode_name, max_decode_abs_limit);
    bool repeat_ok = true;
    if (repeat_count > 1)
    {
        // Keep decoder input fixed independently of repeated encoder output,
        // so a failure identifies the first nondeterministic VAE boundary.
        const float* latent_values = reinterpret_cast<const float*>(latent.data);
        const float* reconstruction_values = reinterpret_cast<const float*>(reconstruction.data);
        const TensorRecord initial_latent{latent_record->shape,
                                         {latent_values, latent_values + latent.total()}};
        const TensorRecord initial_reconstruction{reconstruction_record->shape,
                                                 {reconstruction_values, reconstruction_values + reconstruction.total()}};
        for (int iteration = 1; iteration < repeat_count; ++iteration)
        {
            ncnn::Mat repeated_latent;
            ncnn::Extractor encode_extractor = encode.create_extractor();
            if (encode_extractor.input("in0", input) != 0 || encode_extractor.extract("out0", repeated_latent) != 0)
                return 1;
            const std::string encode_name = "vae_encode_repeat_" + std::to_string(iteration);
            repeat_ok = compare(repeated_latent, initial_latent, encode_name.c_str(), 0.f) && repeat_ok;

            ncnn::Mat repeated_reconstruction;
            ncnn::Extractor decode_extractor = decode.create_extractor();
            if (decode_extractor.input("in0", decode_input) != 0 || decode_extractor.extract("out0", repeated_reconstruction) != 0)
                return 1;
            const std::string decode_name = "vae_decode_fixed_input_repeat_" + std::to_string(iteration);
            repeat_ok = compare(repeated_reconstruction, initial_reconstruction, decode_name.c_str(), 0.f) && repeat_ok;
        }
    }
    std::cout << "mode=" << mode << " storage=" << (mode == "cpu" || fp32_storage ? "fp32" : "bf16")
              << " contract=256x256 result="
              << (encode_ok && decode_ok && repeat_ok ? "pass" : "fail") << "\n";
#if NCNN_VULKAN
    // Release graph-owned GPU tensors before returning the temporary allocators.
    encode.clear();
    decode.clear();
    if (encode_blob)
        vkdev->reclaim_blob_allocator(encode_blob);
    if (encode_staging)
        vkdev->reclaim_staging_allocator(encode_staging);
    if (decode_blob)
        vkdev->reclaim_blob_allocator(decode_blob);
    if (decode_staging)
        vkdev->reclaim_staging_allocator(decode_staging);
#endif
    return encode_ok && decode_ok && repeat_ok ? 0 : 1;
}
