#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "awa/awa_layers.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"

namespace
{

constexpr char kGoldenMagic[8] = {'S', 'V', 'R', '2', 'F', '3', '2', '\0'};
constexpr std::uint32_t kGoldenVersion = 1;
constexpr float kAbsoluteTolerance = 5e-3f;
constexpr float kRelativeTolerance = 5e-3f;
constexpr float kEndToEndRelativeTolerance = 2e-2f;
constexpr double kEndToEndMeanAbsoluteErrorLimit = 1.5e-3;
constexpr double kEndToEndRmseLimit = 2e-3;

struct GoldenRecord
{
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

bool read_bytes(std::istream& input, void* destination, std::size_t size)
{
    return size == 0 || static_cast<bool>(input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size)));
}

std::uint16_t read_u16(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t read_u32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t read_u64(const std::uint8_t* bytes)
{
    std::uint64_t value = 0;
    for (int index = 0; index < 8; index++)
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    return value;
}

bool load_golden(const std::string& path, std::unordered_map<std::string, GoldenRecord>& records)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        std::fprintf(stderr, "failed to open golden pack: %s\n", path.c_str());
        return false;
    }

    std::uint8_t file_header[16];
    if (!read_bytes(input, file_header, sizeof(file_header)) ||
        std::memcmp(file_header, kGoldenMagic, sizeof(kGoldenMagic)) != 0 ||
        read_u32(file_header + 8) != kGoldenVersion)
    {
        std::fprintf(stderr, "invalid golden pack header: %s\n", path.c_str());
        return false;
    }

    const std::uint32_t record_count = read_u32(file_header + 12);
    if (record_count == 0 || record_count > 1024)
    {
        std::fprintf(stderr, "invalid golden record count: %u\n", record_count);
        return false;
    }

    for (std::uint32_t record_index = 0; record_index < record_count; record_index++)
    {
        std::uint8_t record_header[44];
        if (!read_bytes(input, record_header, sizeof(record_header)))
            return false;
        const std::uint16_t name_length = read_u16(record_header);
        const std::uint8_t rank = record_header[2];
        const std::uint64_t count = read_u64(record_header + 4);
        if (name_length == 0 || rank > 4 || count > (1ull << 31))
            return false;

        std::string name(name_length, '\0');
        if (!read_bytes(input, name.data(), name.size()) || records.count(name) != 0)
            return false;

        std::vector<std::uint64_t> shape(rank);
        std::uint64_t expected_count = 1;
        for (std::uint8_t dimension = 0; dimension < rank; dimension++)
        {
            shape[dimension] = read_u64(record_header + 12 + dimension * 8);
            if (shape[dimension] == 0 || expected_count > std::numeric_limits<std::uint64_t>::max() / shape[dimension])
                return false;
            expected_count *= shape[dimension];
        }
        if (expected_count != count || count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;

        std::vector<std::uint8_t> raw(static_cast<std::size_t>(count) * sizeof(float));
        if (!read_bytes(input, raw.data(), raw.size()))
            return false;
        GoldenRecord record;
        record.shape = std::move(shape);
        record.values.resize(static_cast<std::size_t>(count));
        for (std::size_t value_index = 0; value_index < record.values.size(); value_index++)
        {
            const std::uint32_t bits = read_u32(raw.data() + value_index * sizeof(float));
            std::memcpy(&record.values[value_index], &bits, sizeof(float));
        }
        records.emplace(std::move(name), std::move(record));
    }
    return true;
}

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
    return true;
}

bool load_graph(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
                ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_awa_layers(net);
    const std::string param = stem + ".ncnn.param";
    const std::string model = stem + ".ncnn.bin";
    if (net.load_param(param.c_str()) != 0 || net.load_model(model.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load graph: %s\n", stem.c_str());
        return false;
    }
    return true;
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

bool make_input(const GoldenRecord& record, ncnn::Mat& output)
{
    if (record.shape.size() == 1)
    {
        if (record.shape[0] > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            return false;
        output.create(static_cast<int>(record.shape[0]));
    }
    else if (record.shape.size() == 2 && record.shape[0] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
             record.shape[1] <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        output.create(static_cast<int>(record.shape[1]), static_cast<int>(record.shape[0]));
    }
    else
    {
        return false;
    }
    if (output.total() < record.values.size())
    {
        std::fprintf(stderr, "golden Mat size mismatch: dims=%d w=%d h=%d total=%zu expected=%zu\n", output.dims,
                     output.w, output.h, output.total(), record.values.size());
        return false;
    }
    output.fill(0.f);
    std::memcpy(output.data, record.values.data(), record.values.size() * sizeof(float));
    return true;
}

std::size_t logical_per_batch(const ncnn::Mat& value)
{
    if (value.elempack == 0)
        return 0;
    return static_cast<std::size_t>(value.w) * value.h * value.d * value.c * value.elempack;
}

bool finite(const ncnn::Mat& value)
{
    if (value.empty() || value.elempack != 1)
        return false;
    const std::size_t count = logical_per_batch(value);
#if NCNN_BATCH
    for (int batch = 0; batch < value.n; batch++)
    {
        const float* data = static_cast<const float*>(value.batch(batch).data);
        for (std::size_t index = 0; index < count; index++)
            if (!std::isfinite(data[index]))
                return false;
    }
#else
    const float* data = static_cast<const float*>(value.data);
    for (std::size_t index = 0; index < count; index++)
        if (!std::isfinite(data[index]))
            return false;
#endif
    return true;
}

bool compare_output(const ncnn::Mat& actual, const GoldenRecord& expected, const char* name)
{
    const std::size_t actual_count = logical_per_batch(actual) *
#if NCNN_BATCH
                                     static_cast<std::size_t>(actual.n);
#else
                                     1u;
#endif
    if (!finite(actual) || actual_count != expected.values.size())
    {
        std::fprintf(stderr, "unexpected output for %s: logical_total=%zu expected=%zu\n", name, actual_count,
                     expected.values.size());
        return false;
    }
    float max_error = 0.f;
    std::size_t max_index = 0;
    float max_actual = 0.f;
    float max_expected = 0.f;
    double total_absolute_error = 0.0;
    double total_squared_error = 0.0;
    std::size_t index = 0;
#if NCNN_BATCH
    const int batches = actual.n;
#else
    const int batches = 1;
#endif
    for (int batch = 0; batch < batches; batch++)
    {
        const float* actual_data = static_cast<const float*>(actual.batch(batch).data);
        for (std::size_t offset = 0; offset < logical_per_batch(actual); offset++, index++)
        {
            const float difference = std::fabs(actual_data[offset] - expected.values[index]);
            total_absolute_error += difference;
            total_squared_error += static_cast<double>(difference) * difference;
            if (difference > max_error)
            {
                max_error = difference;
                max_index = index;
                max_actual = actual_data[offset];
                max_expected = expected.values[index];
            }
        }
    }
    const bool end_to_end = std::strcmp(name, "output_video") == 0;
    const float relative_tolerance = end_to_end ? kEndToEndRelativeTolerance : kRelativeTolerance;
    const double mean_absolute_error = total_absolute_error / expected.values.size();
    const double rmse = std::sqrt(total_squared_error / expected.values.size());
    std::fprintf(stderr, "golden=%s values=%zu max_abs_error=%g mean_abs_error=%g rmse=%g\n", name,
                 expected.values.size(), max_error, mean_absolute_error, rmse);
    if (max_error > kAbsoluteTolerance + relative_tolerance * std::max(1.f, std::fabs(max_expected)) ||
        (end_to_end && (mean_absolute_error > kEndToEndMeanAbsoluteErrorLimit || rmse > kEndToEndRmseLimit)))
    {
        std::fprintf(stderr, "golden mismatch for %s at %zu: actual=%g expected=%g error=%g\n", name,
                     max_index, max_actual, max_expected, max_error);
        return false;
    }
    return true;
}

bool download_and_compare(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, const ncnn::VkMat& value,
                          const GoldenRecord& expected, const char* name)
{
    ncnn::Mat downloaded;
    ncnn::VkCompute compute(vkdev);
    compute.record_download(value, downloaded, opt);
    return compute.submit_and_wait() == 0 && compare_output(downloaded, expected, name);
}

bool has_records(const std::unordered_map<std::string, GoldenRecord>& records)
{
    return records.count("input_video_patches") != 0 && records.count("input_text") != 0 &&
           records.count("timestep") != 0 && records.count("input_video_hidden") != 0 &&
           records.count("input_text_hidden") != 0 && records.count("embedding") != 0 &&
           records.count("output_video") != 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::fprintf(stderr, "usage: test_dit_stack_vulkan <export-dir> [--trace]\n");
        return 2;
    }
    const bool trace = argc == 3 && std::string(argv[2]) == "--trace";
    if (argc == 3 && !trace)
        return 2;

    const std::string export_dir = argv[1];
    std::unordered_map<std::string, GoldenRecord> records;
    if (!load_golden(export_dir + "/dit_stack_golden.f32", records) || !has_records(records))
    {
        std::fprintf(stderr, "golden load failed: records=%zu\n", records.size());
        return 1;
    }
    ncnn::Mat video_input;
    ncnn::Mat text_input;
    ncnn::Mat timestep;
    const bool video_input_ok = make_input(records.at("input_video_patches"), video_input);
    const bool text_input_ok = make_input(records.at("input_text"), text_input);
    const bool timestep_ok = make_input(records.at("timestep"), timestep);
    if (!video_input_ok || !text_input_ok || !timestep_ok)
    {
        std::fprintf(stderr, "golden input materialization failed: video=%d text=%d timestep=%d\n", video_input_ok,
                     text_input_ok, timestep_ok);
        return 1;
    }
    if (text_input.dims != 2 || text_input.w != 5120 || (text_input.h != 58 && text_input.h != 64))
    {
        std::fprintf(stderr, "unsupported text input shape: dims=%d w=%d h=%d\n", text_input.dims,
                     text_input.w, text_input.h);
        return 1;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        std::fprintf(stderr, "no Vulkan device available\n");
        return 1;
    }
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    int result = 1;

    do
    {
        ncnn::Net video_unpack_net;
        ncnn::Net text_unpack_net;
        if (!load_packing_graph(video_unpack_net, vkdev, blob_allocator, staging_allocator) ||
            !load_packing_graph(text_unpack_net, vkdev, blob_allocator, staging_allocator))
            break;
        ncnn::VkMat video_gpu;
        ncnn::VkMat text_gpu;
        ncnn::VkMat embedding_gpu;

        {
            ncnn::Net net;
            if (!load_graph(net, export_dir + "/dit_input", vkdev, blob_allocator, staging_allocator))
                break;
            ncnn::VkMat video_input_gpu;
            ncnn::VkMat text_input_gpu;
            ncnn::VkCompute upload(vkdev);
            upload.record_upload(video_input, video_input_gpu, net.opt);
            upload.record_upload(text_input, text_input_gpu, net.opt);
            if (upload.submit_and_wait() != 0)
                break;
            ncnn::Extractor extractor = net.create_extractor();
            extractor.set_light_mode(false);
            ncnn::VkCompute compute(vkdev);
            if (extractor.input("in0", video_input_gpu) != 0 || extractor.input("in1", text_input_gpu) != 0 ||
                extractor.extract("out0", video_gpu, compute) != 0 || extractor.extract("out1", text_gpu, compute) != 0 ||
                compute.submit_and_wait() != 0)
                break;
            ncnn::VkMat video_unpacked;
            ncnn::VkMat text_unpacked;
            if (!unpack_to_pack1(video_unpack_net, video_gpu, vkdev, video_unpacked) ||
                !unpack_to_pack1(text_unpack_net, text_gpu, vkdev, text_unpacked))
            {
                std::fprintf(stderr, "input graph pack1 conversion failed\n");
                break;
            }
            video_gpu = video_unpacked;
            text_gpu = text_unpacked;
            ncnn::VkMat video_batch;
            ncnn::VkMat text_batch;
            if (!matrix_to_batch_gpu(video_gpu, video_input.h, vkdev, net.opt, blob_allocator, video_batch) ||
                !matrix_to_batch_gpu(text_gpu, text_input.h, vkdev, net.opt, blob_allocator, text_batch))
            {
                std::fprintf(stderr, "input matrix-to-batch conversion failed\n");
                break;
            }
            video_gpu = video_batch;
            text_gpu = text_batch;

            if (trace &&
                (!download_and_compare(vkdev, net.opt, video_gpu, records.at("input_video_hidden"),
                                       "input_video_hidden") ||
                 !download_and_compare(vkdev, net.opt, text_gpu, records.at("input_text_hidden"),
                                       "input_text_hidden")))
                std::fprintf(stderr, "trace: input mismatch; continuing\n");
        }

        {
            ncnn::Net net;
            if (!load_graph(net, export_dir + "/dit_embedding", vkdev, blob_allocator, staging_allocator))
                break;
            ncnn::VkMat timestep_gpu;
            ncnn::VkCompute upload(vkdev);
            upload.record_upload(timestep, timestep_gpu, net.opt);
            if (upload.submit_and_wait() != 0)
                break;
            ncnn::Extractor extractor = net.create_extractor();
            extractor.set_light_mode(false);
            ncnn::VkCompute compute(vkdev);
            if (extractor.input("in0", timestep_gpu) != 0 || extractor.extract("out0", embedding_gpu, compute) != 0 ||
                compute.submit_and_wait() != 0)
                break;
            ncnn::VkMat embedding_unpacked;
            if (!unpack_to_pack1(video_unpack_net, embedding_gpu, vkdev, embedding_unpacked))
            {
                std::fprintf(stderr, "embedding pack1 conversion failed\n");
                break;
            }
            embedding_gpu = embedding_unpacked;
            if (trace && !download_and_compare(vkdev, net.opt, embedding_gpu, records.at("embedding"), "embedding"))
                std::fprintf(stderr, "trace: embedding mismatch; continuing\n");
        }

        for (int block_index = 0; block_index < 32; block_index++)
        {
            ncnn::Net net;
            char block_name[32];
            std::snprintf(block_name, sizeof(block_name), "/dit_block_%02d", block_index);
            if (!load_graph(net, export_dir + block_name, vkdev, blob_allocator, staging_allocator))
                break;
            ncnn::Extractor extractor = net.create_extractor();
            extractor.set_light_mode(false);
            ncnn::VkMat next_video_gpu;
            ncnn::VkMat next_text_gpu;
            ncnn::VkCompute compute(vkdev);
            const int input_video_ret = extractor.input("in0", video_gpu);
            const int input_text_ret = extractor.input("in1", text_gpu);
            const int input_embedding_ret = extractor.input("in2", embedding_gpu);
            const int output_video_ret = input_video_ret == 0 && input_text_ret == 0 && input_embedding_ret == 0 &&
                                             extractor.extract("out0", next_video_gpu, compute);
            const int output_text_ret = output_video_ret == 0 ? extractor.extract("out1", next_text_gpu, compute) : -1;
            if (input_video_ret != 0 || input_text_ret != 0 || input_embedding_ret != 0 || output_video_ret != 0 ||
                output_text_ret != 0 || compute.submit_and_wait() != 0)
            {
                std::fprintf(stderr, "block %02d failed: in=(%d,%d,%d) out=(%d,%d)\n", block_index,
                             input_video_ret, input_text_ret, input_embedding_ret, output_video_ret, output_text_ret);
                break;
            }
            video_gpu = next_video_gpu;
            text_gpu = next_text_gpu;
            if (trace)
            {
                char video_name[64];
                char text_name[64];
                std::snprintf(video_name, sizeof(video_name), "block_%02d_video", block_index);
                std::snprintf(text_name, sizeof(text_name), "block_%02d_text", block_index);
                if (!download_and_compare(vkdev, net.opt, video_gpu, records.at(video_name), video_name) ||
                    !download_and_compare(vkdev, net.opt, text_gpu, records.at(text_name), text_name))
                    std::fprintf(stderr, "trace: block %02d mismatch; continuing\n", block_index);
            }
            if (block_index == 31)
            {
                ncnn::Net output_net;
                if (!load_graph(output_net, export_dir + "/dit_output", vkdev, blob_allocator, staging_allocator))
                    break;
                ncnn::VkMat video_unpacked;
                ncnn::VkMat video_matrix_gpu;
                if (!unpack_to_pack1(video_unpack_net, video_gpu, vkdev, video_unpacked) ||
                    !batch_to_matrix_gpu(video_unpacked, vkdev, output_net.opt, blob_allocator, video_matrix_gpu))
                {
                    std::fprintf(stderr, "output batch-to-matrix conversion failed\n");
                    break;
                }
                ncnn::Extractor output_extractor = output_net.create_extractor();
                output_extractor.set_light_mode(false);
                ncnn::VkMat output_gpu;
                ncnn::VkCompute compute_output(vkdev);
                if (output_extractor.input("in0", video_matrix_gpu) != 0 || output_extractor.input("in1", embedding_gpu) != 0 ||
                    output_extractor.extract("out0", output_gpu, compute_output) != 0 ||
                    compute_output.submit_and_wait() != 0 ||
                    !download_and_compare(vkdev, output_net.opt, output_gpu, records.at("output_video"), "output_video"))
                    break;
                result = 0;
            }
        }
    } while (false);

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);
    if (result == 0)
        std::puts("seedvr2-dit-stack-vulkan: ok");
    return result;
}
