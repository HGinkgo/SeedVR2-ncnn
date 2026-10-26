#include "dit/dit_stack.h"
#include "dit/dit_stack_internal.h"
#include "dit/dit_precision.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "awa/awa_layers.h"
#include "datareader.h"
#include "layer_type.h"
#include "net.h"
#include "pipelinecache.h"
#include "inference/performance_profile.h"
#include "inference/vulkan_benchmark.h"

namespace seedvr2
{
namespace
{

constexpr int kLatentChannels = 16;
constexpr int kPatchSize = 2;
constexpr int kVideoPatchWidth = 132;
constexpr int kVideoHiddenWidth = 2560;
constexpr int kTextHiddenWidth = 2560;
constexpr int kTextInputWidth = 5120;
constexpr int kOutputPatchWidth = 64;

// Recent ncnn versions no longer expose the internal model-load observer.
// Keep the public profile useful by measuring the supported load calls directly.
struct DitLoadProfile final
{
    explicit DitLoadProfile(const PerformanceProfile* profile_in) : profile(profile_in) {}

    int load_param(ncnn::Net& net, const char* path)
    {
        const auto start = PerformanceProfile::Clock::now();
        const int result = net.load_param(path);
        param_ms += profile->elapsed_ms(start);
        return result;
    }

    int load_param_mem(ncnn::Net& net, const char* param)
    {
        const auto start = PerformanceProfile::Clock::now();
        const int result = net.load_param_mem(param);
        param_ms += profile->elapsed_ms(start);
        return result;
    }

    int load_model(ncnn::Net& net, const char* path)
    {
        const auto start = PerformanceProfile::Clock::now();
        const int result = net.load_model(path);
        bin_ms += profile->elapsed_ms(start);
        return result;
    }

    void report() const
    {
        std::fprintf(stderr, "%s\n", format_profile_dit_load_line("param", param_ms).c_str());
        std::fprintf(stderr, "%s\n", format_profile_dit_load_line("bin", bin_ms).c_str());
        // ncnn's public load_model call covers layer loading and Vulkan pipeline setup.
        std::fprintf(stderr, "%s\n", format_profile_dit_stage_line("ncnn-model-load", bin_ms).c_str());
    }

    const PerformanceProfile* profile = nullptr;
    double param_ms = 0.0;
    double bin_ms = 0.0;
};

} // namespace

void configure_dit_vulkan_net_impl(ncnn::Net& net,
                                   ncnn::VulkanDevice* vkdev,
                                   ncnn::VkAllocator* blob_allocator,
                                   ncnn::VkAllocator* staging_allocator,
                                   ncnn::PipelineCache* pipeline_cache,
                                   bool use_bf16)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_bf16_packed = use_bf16;
    net.opt.use_bf16_storage = use_bf16;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.blob_vkallocator = blob_allocator;
    net.opt.workspace_vkallocator = blob_allocator;
    net.opt.staging_vkallocator = staging_allocator;
    net.opt.pipeline_cache = pipeline_cache;
    net.set_vulkan_device(vkdev);
}

void configure_dit_vulkan_net(ncnn::Net& net,
                              ncnn::VulkanDevice* vkdev,
                              ncnn::VkAllocator* blob_allocator,
                              ncnn::VkAllocator* staging_allocator,
                              ncnn::PipelineCache* pipeline_cache)
{
    configure_dit_vulkan_net_impl(net, vkdev, blob_allocator, staging_allocator, pipeline_cache, true);
}

void configure_dit_vulkan_net_fp32(ncnn::Net& net,
                                   ncnn::VulkanDevice* vkdev,
                                   ncnn::VkAllocator* blob_allocator,
                                   ncnn::VkAllocator* staging_allocator,
                                   ncnn::PipelineCache* pipeline_cache)
{
    configure_dit_vulkan_net_impl(net, vkdev, blob_allocator, staging_allocator, pipeline_cache, false);
}

namespace
{

template<int Type>
ncnn::Layer* create_timestep_fp32_cpu(void*)
{
    ncnn::Layer* layer = ncnn::create_layer_cpu(Type);
    if (layer)
    {
        layer->support_bf16_storage = false;
        layer->support_fp16_storage = false;
    }
    return layer;
}

} // namespace

void register_dit_timestep_layers(ncnn::Net& net)
{
    // TimeEmbedding computes frequencies, t*frequency and sin/cos in FP32,
    // then casts once before proj_in. BF16 angles can change sin's sign at
    // t=1000. Keep this small prefix on the CPU; ncnn uploads/casts its output
    // at the first Vulkan InnerProduct, whose weights/activations stay BF16.
    // The FP32 path also needs CPU trig to avoid device sin/cos range error.
    net.register_custom_layer("Reshape", create_timestep_fp32_cpu<ncnn::LayerType::Reshape>);
    net.register_custom_layer("MemoryData", create_timestep_fp32_cpu<ncnn::LayerType::MemoryData>);
    net.register_custom_layer("BinaryOp", create_timestep_fp32_cpu<ncnn::LayerType::BinaryOp>);
    net.register_custom_layer("Split", create_timestep_fp32_cpu<ncnn::LayerType::Split>);
    net.register_custom_layer("UnaryOp", create_timestep_fp32_cpu<ncnn::LayerType::UnaryOp>);
    net.register_custom_layer("Concat", create_timestep_fp32_cpu<ncnn::LayerType::Concat>);
}

namespace
{

bool has_named_parameter(const std::string& param_path, const char* expected)
{
    std::ifstream stream(param_path);
    std::string line, type, name;
    while (std::getline(stream, line))
    {
        std::istringstream fields(line);
        if (fields >> type >> name && type == "MemoryData" && name == expected)
            return true;
    }
    return false;
}

bool load_graph(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
                ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator,
                ncnn::PipelineCache* pipeline_cache, const AwaRuntimeSpec* runtime_spec,
                DitLoadProfile* load_profile, bool use_fp32)
{
    const std::string graph_name = stem.substr(stem.find_last_of('/') + 1);
    const auto graph_start = load_profile && load_profile->profile && load_profile->profile->enabled()
                                 ? PerformanceProfile::Clock::now()
                                 : PerformanceProfile::Clock::time_point{};
    if (use_fp32)
        configure_dit_vulkan_net_fp32(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    else
        configure_dit_vulkan_net(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    register_seedvr2_awa_layers(net, runtime_spec);
    const std::string param_path = stem + ".ncnn.param";
    const bool mixed_block = !use_fp32 && graph_name.find("dit_block_") == 0 &&
                            has_named_parameter(param_path, "constant_p_ada_vid_attn_scale");
    const bool mixed_output = !use_fp32 && graph_name == "dit_output" &&
                             has_named_parameter(param_path, "constant_p_norm");
    if (mixed_block)
        register_dit_block_precision_layers(net);
    if (mixed_output)
        register_dit_output_precision_layers(net);
    if (graph_name == "dit_embedding")
        register_dit_timestep_layers(net);
    const std::string model_path = stem + ".ncnn.bin";
    const int param_status = load_profile ? load_profile->load_param(net, param_path.c_str())
                                          : net.load_param(param_path.c_str());
    if (param_status != 0)
    {
        std::fprintf(stderr, "load_graph: failed to load %s\n", param_path.c_str());
        return false;
    }
    if (mixed_block && !configure_dit_block_precision_layers(net))
    {
        std::fprintf(stderr, "load_graph: unrecognized BF16 Ada graph %s\n", param_path.c_str());
        return false;
    }
    if (mixed_output && !configure_dit_output_precision_layers(net))
    {
        std::fprintf(stderr, "load_graph: unrecognized BF16 output graph %s\n", param_path.c_str());
        return false;
    }
    const int model_status = load_profile ? load_profile->load_model(net, model_path.c_str())
                                          : net.load_model(model_path.c_str());
    if (model_status != 0)
    {
        std::fprintf(stderr, "load_graph: failed to load %s\n", model_path.c_str());
        return false;
    }
    if (load_profile && load_profile->profile)
        load_profile->profile->record_runtime_graph_load(false, load_profile->profile->elapsed_ms(graph_start));
    return true;
}

bool load_packing_graph(ncnn::Net& net, ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
                        ncnn::VkAllocator* staging_allocator, ncnn::PipelineCache* pipeline_cache,
                        DitLoadProfile* load_profile, bool use_fp32)
{
    const auto graph_start = load_profile && load_profile->profile && load_profile->profile->enabled()
                                 ? PerformanceProfile::Clock::now()
                                 : PerformanceProfile::Clock::time_point{};
    static const char kPackingParam[] =
        "7767517\n"
        "2 2\n"
        "Input in0 0 1 in0\n"
        "Packing unpack 1 1 in0 out0 0=1\n";
    if (use_fp32)
        configure_dit_vulkan_net_fp32(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    else
        configure_dit_vulkan_net(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    const int param_status = load_profile ? load_profile->load_param_mem(net, kPackingParam)
                                          : net.load_param_mem(kPackingParam);
    if (param_status != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    const bool loaded = net.load_model(model_reader) == 0;
    if (loaded && load_profile && load_profile->profile)
        load_profile->profile->record_runtime_graph_load(false, load_profile->profile->elapsed_ms(graph_start));
    return loaded;
}

bool load_graph_from_param(ncnn::Net& net, const char* param, ncnn::VulkanDevice* vkdev,
                           ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator,
                           ncnn::PipelineCache* pipeline_cache,
                           const PerformanceProfile* profile, bool use_fp32)
{
    const auto graph_start = profile && profile->enabled() ? PerformanceProfile::Clock::now()
                                                             : PerformanceProfile::Clock::time_point{};
    if (use_fp32)
        configure_dit_vulkan_net_fp32(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    else
        configure_dit_vulkan_net(net, vkdev, blob_allocator, staging_allocator, pipeline_cache);
    if (net.load_param_mem(param) != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    const bool loaded = net.load_model(model_reader) == 0;
    if (loaded && profile)
        profile->record_runtime_graph_load(true, profile->elapsed_ms(graph_start));
    return loaded;
}

// The two supported DiT package generations use the same [tokens, 2560]
// output shape after dit_input, but their block graphs consume different
// layouts.  The legacy TorchScript export uses InnerProduct and expects one
// token per ncnn batch item; full-matrix TorchScript and PT2 exports use Gemm
// plus an explicit rank-3 reshape and consume the complete token matrix.
// Infer this contract from
// the block graph itself so a package cannot silently take the wrong path.
bool detect_direct_matrix_contract(const std::string& stack_dir)
{
    std::ifstream param(stack_dir + "/dit_block_00.ncnn.param");
    if (!param)
        return false;

    bool has_gemm_projection = false;
    bool has_explicit_rank3_reshape = false;
    std::string line;
    while (std::getline(param, line))
    {
        if (line.rfind("Gemm gemm_0 ", 0) == 0)
            has_gemm_projection = true;
        if (line.rfind("Reshape ", 0) == 0 && line.find("11=3") != std::string::npos)
            has_explicit_rank3_reshape = true;
    }
    return has_gemm_projection && has_explicit_rank3_reshape;
}

bool is_plan_latent(const ncnn::VkMat& value, const ResolutionPlan& plan, bool use_fp32)
{
    return !value.empty() && (value.dims == 3 || value.dims == 4) && value.w == plan.latent_width &&
           value.h == plan.latent_height && value.d == 1 && value.c == kLatentChannels && value.elempack == 1 &&
           (use_fp32 ? value.elemsize == 4u : value.elemsize == 2u);
}

void collapse_single_frame(ncnn::VkMat& value)
{
    if (value.dims == 4 && value.d == 1)
    {
        value.dims = 3;
        value.d = 1;
    }
}

bool unpack_gpu_to_pack1(const ncnn::VkMat& input, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt,
                         ncnn::VkMat& output, const PerformanceProfile* profile)
{
    if (input.empty() || !vkdev)
        return false;
    if (input.elempack == 1)
    {
        output = input;
        return true;
    }

    ncnn::VkCompute compute(vkdev);
    vkdev->convert_packing(input, output, 1, compute, opt);
    return !output.empty() && submit_and_wait_profiled(compute, profile) == 0;
}

bool convert_precision_gpu(const ncnn::VkMat& input, ncnn::VulkanDevice* vkdev,
                           const ncnn::Option& destination_opt, bool to_bf16,
                           ncnn::VkMat& output, const PerformanceProfile* profile)
{
    if (input.empty() || !vkdev || input.elembits() != (to_bf16 ? 32 : 16))
    {
        std::fprintf(stderr, "convert_precision_gpu: invalid input empty=%d elemsize=%zu pack=%d to_bf16=%d\n",
                     input.empty(), input.elemsize, input.elempack, to_bf16 ? 1 : 0);
        return false;
    }

    ncnn::Option convert_opt = destination_opt;
    // convert_packing infers the source 16-bit format from opt, so describe
    // the input representation here while the explicit cast type selects
    // the destination representation.
    convert_opt.use_bf16_packed = !to_bf16;
    convert_opt.use_bf16_storage = !to_bf16;
    convert_opt.use_fp16_packed = false;
    convert_opt.use_fp16_storage = false;
    ncnn::VkCompute compute(vkdev);
    // VulkanDevice uses cast type 1 for FP32 and 5 for BF16. Keeping the cast
    // on-device avoids materializing the large token matrices on the CPU.
    vkdev->convert_packing(input, output, 1, to_bf16 ? 5 : 1, compute, convert_opt);
    const bool converted = !output.empty() && submit_and_wait_profiled(compute, profile) == 0;
    if (!converted)
        std::fprintf(stderr, "convert_precision_gpu: cast failed dims=%d w=%d h=%d n=%d size=%zu pack=%d to_bf16=%d\n",
                     input.dims, input.w, input.h, input.n, input.elemsize, input.elempack, to_bf16 ? 1 : 0);
    return converted;
}

bool unpack_to_pack1(ncnn::Net& net, const ncnn::VkMat& packed, ncnn::VulkanDevice* vkdev,
                     ncnn::VkMat& unpacked, const PerformanceProfile* profile)
{
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    return prepare_ncnn_layer_benchmark(compute, net) && extractor.input("in0", packed) == 0 &&
           extractor.extract("out0", unpacked, compute) == 0 &&
           submit_and_wait_profiled(compute, profile) == 0;
}

bool matrix_to_batch_gpu(const ncnn::VkMat& matrix, int rows, ncnn::VulkanDevice* vkdev,
                         const ncnn::Option& opt, ncnn::VkAllocator* allocator, ncnn::VkMat& batch,
                         const PerformanceProfile* profile)
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
    return submit_and_wait_profiled(compute, profile) == 0;
}

bool batch_to_matrix_gpu(const ncnn::VkMat& batch, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt,
                         ncnn::VkAllocator* allocator, ncnn::VkMat& matrix,
                         const PerformanceProfile* profile)
{
    if (batch.empty() || batch.elempack != 1 || batch.w <= 0)
    {
        std::fprintf(stderr,
                     "batch_to_matrix_gpu: invalid batch empty=%d dims=%d w=%d h=%d d=%d c=%d n=%d pack=%d elem=%zu\n",
                     batch.empty(), batch.dims, batch.w, batch.h, batch.d, batch.c, batch.n, batch.elempack,
                     batch.elemsize);
        return false;
    }

    if (batch.dims != 1 || batch.n <= 1)
    {
        std::fprintf(stderr,
                     "batch_to_matrix_gpu: unsupported batch layout dims=%d w=%d h=%d d=%d c=%d n=%d pack=%d elem=%zu\n",
                     batch.dims, batch.w, batch.h, batch.d, batch.c, batch.n, batch.elempack, batch.elemsize);
        return false;
    }
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
    return submit_and_wait_profiled(compute, profile) == 0;
}

} // namespace

struct DitStackSession::Impl
{
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    bool use_fp32 = false;
    int fp32_prefix_blocks = 0;
    const PerformanceProfile* profile = nullptr;
    std::unique_ptr<ncnn::PipelineCache> owned_pipeline_cache;
    ncnn::PipelineCache* pipeline_cache = nullptr;
    bool direct_matrix_contract = false;
    ncnn::Net dit_input;
    ncnn::Net dit_embedding;
    ncnn::Net packing;
    ncnn::Net fp32_packing;
    std::vector<std::unique_ptr<ncnn::Net>> blocks;
    ncnn::Net dit_output;

    void clear()
    {
        dit_output.clear();
        for (auto& block : blocks)
            if (block)
                block->clear();
        blocks.clear();
        packing.clear();
        fp32_packing.clear();
        dit_embedding.clear();
        dit_input.clear();
        owned_pipeline_cache.reset();
        pipeline_cache = nullptr;
    }
};

DitStackSession::DitStackSession() = default;

DitStackSession::~DitStackSession()
{
    clear();
}

void DitStackSession::clear()
{
    if (impl_)
    {
        impl_->clear();
        impl_.reset();
    }
}

DitStackSession::DitStackSession(DitStackSession&&) noexcept = default;

DitStackSession& DitStackSession::operator=(DitStackSession&&) noexcept = default;

bool DitStackSession::open(const std::string& stack_dir,
                           const ResolutionPlan& plan,
                           ncnn::VulkanDevice* vkdev,
                           ncnn::VkAllocator* blob_allocator,
                           ncnn::VkAllocator* staging_allocator,
                           DitStackSession& session,
                           const PerformanceProfile* profile,
                           ncnn::PipelineCache* pipeline_cache,
                           bool use_fp32)
{
    if (!vkdev || !blob_allocator || !staging_allocator || plan.video_tokens <= 0)
        return false;
    const AwaRuntimeSpec runtime_spec = {plan.source_t, plan.source_height, plan.source_width, plan.video_tokens};
    if (!runtime_spec.valid())
        return false;

    std::unique_ptr<Impl> candidate(new Impl);
    candidate->vkdev = vkdev;
    candidate->blob_allocator = blob_allocator;
    candidate->staging_allocator = staging_allocator;
    candidate->profile = profile;
    candidate->use_fp32 = use_fp32;
    candidate->fp32_prefix_blocks = use_fp32 ? 0 : 9;
    candidate->direct_matrix_contract = detect_direct_matrix_contract(stack_dir);
    if (pipeline_cache)
        candidate->pipeline_cache = pipeline_cache;
    else
    {
        candidate->owned_pipeline_cache.reset(new ncnn::PipelineCache(vkdev));
        candidate->pipeline_cache = candidate->owned_pipeline_cache.get();
    }
    DitLoadProfile load_profile(profile && profile->enabled() ? profile : nullptr);
    DitLoadProfile* load_profile_ptr = load_profile.profile ? &load_profile : nullptr;
    const bool use_fp32_prefix = candidate->fp32_prefix_blocks > 0;
    const bool use_fp32_input = use_fp32 || use_fp32_prefix;
    if (!load_graph(candidate->dit_input, stack_dir + "/dit_input", vkdev, blob_allocator, staging_allocator,
                    candidate->pipeline_cache, &runtime_spec, load_profile_ptr, use_fp32_input) ||
        !load_graph(candidate->dit_embedding, stack_dir + "/dit_embedding", vkdev, blob_allocator,
                    staging_allocator, candidate->pipeline_cache, &runtime_spec, load_profile_ptr, use_fp32_input) ||
        !load_packing_graph(candidate->packing, vkdev, blob_allocator, staging_allocator,
                            candidate->pipeline_cache, load_profile_ptr, use_fp32))
        return false;
    if (use_fp32_prefix &&
        !load_packing_graph(candidate->fp32_packing, vkdev, blob_allocator, staging_allocator,
                            candidate->pipeline_cache, load_profile_ptr, true))
        return false;

    candidate->blocks.reserve(32);
    for (int block_index = 0; block_index < 32; block_index++)
    {
        std::unique_ptr<ncnn::Net> block(new ncnn::Net);
        const std::string block_name = stack_dir + "/dit_block_" +
                                       (block_index < 10 ? "0" : "") + std::to_string(block_index);
        const bool block_use_fp32 = use_fp32 || block_index < candidate->fp32_prefix_blocks;
        if (!load_graph(*block, block_name, vkdev, blob_allocator, staging_allocator,
                        candidate->pipeline_cache, &runtime_spec, load_profile_ptr, block_use_fp32))
            return false;
        candidate->blocks.push_back(std::move(block));
    }
    if (!load_graph(candidate->dit_output, stack_dir + "/dit_output", vkdev, blob_allocator, staging_allocator,
                    candidate->pipeline_cache, &runtime_spec, load_profile_ptr, use_fp32))
        return false;

    if (load_profile_ptr)
        load_profile_ptr->report();

    session.impl_ = std::move(candidate);
    return true;
}

bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                const ResolutionPlan& plan,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches,
                                ncnn::PipelineCache* pipeline_cache,
                                const PerformanceProfile* profile,
                                bool use_fp32)
{
    if (!vkdev || !blob_allocator || !staging_allocator || noise.empty() || condition.empty())
    {
        std::fprintf(stderr,
                     "make_dit_input_patches_gpu: input mismatch noise=(empty=%d,dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d,es=%zu) "
                     "condition=(empty=%d,dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d,es=%zu)\n",
                     noise.empty(), noise.dims, noise.w, noise.h, noise.d, noise.c, noise.elempack, noise.elemsize,
                     condition.empty(), condition.dims, condition.w, condition.h, condition.d, condition.c,
                     condition.elempack, condition.elemsize);
        return false;
    }

    ncnn::Net net;
    std::ostringstream param;
    param << "7767517\n7 7\n"
          << "Input noise 0 1 noise\n"
          << "Input condition 0 1 condition\n"
          << "Input mask 0 1 mask\n"
          << "Concat concat 3 1 noise condition mask video 0=0\n"
          << "Reorg patchify 1 1 video patch_grid 0=2 1=1\n"
          << "Permute token_major 1 1 patch_grid token_grid 0=3\n"
          << "Reshape flatten 1 1 token_grid patches 0=132 1=" << plan.video_tokens << "\n";
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator, pipeline_cache,
                               profile, use_fp32))
    {
        std::fprintf(stderr, "make_dit_input_patches_gpu: graph load failed\n");
        return false;
    }

    ncnn::VkMat noise_pack1;
    ncnn::VkMat condition_pack1;
    if (!unpack_gpu_to_pack1(noise, vkdev, net.opt, noise_pack1, profile) ||
        !unpack_gpu_to_pack1(condition, vkdev, net.opt, condition_pack1, profile))
    {
        std::fprintf(stderr,
                     "make_dit_input_patches_gpu: invalid latent noise=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d) "
                     "condition=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d)\n",
                     noise_pack1.dims, noise_pack1.w, noise_pack1.h, noise_pack1.d, noise_pack1.c,
                     noise_pack1.elempack, condition_pack1.dims, condition_pack1.w, condition_pack1.h,
                     condition_pack1.d, condition_pack1.c, condition_pack1.elempack);
        return false;
    }
    collapse_single_frame(noise_pack1);
    collapse_single_frame(condition_pack1);
    if (!is_plan_latent(noise_pack1, plan, use_fp32) || !is_plan_latent(condition_pack1, plan, use_fp32) ||
        noise_pack1.dims != condition_pack1.dims ||
        noise_pack1.w != condition_pack1.w || noise_pack1.h != condition_pack1.h || noise_pack1.c != condition_pack1.c)
    {
        std::fprintf(stderr,
                     "make_dit_input_patches_gpu: normalized latent mismatch noise=(dims=%d,w=%d,h=%d,d=%d,c=%d) "
                     "condition=(dims=%d,w=%d,h=%d,d=%d,c=%d)\n",
                     noise_pack1.dims, noise_pack1.w, noise_pack1.h, noise_pack1.d, noise_pack1.c,
                     condition_pack1.dims, condition_pack1.w, condition_pack1.h, condition_pack1.d,
                     condition_pack1.c);
        return false;
    }

    ncnn::Mat mask;
    if (noise_pack1.dims == 3)
        mask.create(plan.latent_width, plan.latent_height, 1);
    else
        mask.create(plan.latent_width, plan.latent_height, 1, 1);
    if (mask.empty())
    {
        std::fprintf(stderr, "make_dit_input_patches_gpu: mask allocation failed\n");
        return false;
    }
    mask.fill(1.f);
    ncnn::VkMat mask_gpu;
    {
        ncnn::VkCompute upload(vkdev);
        if (profile)
            profile->record_runtime_upload();
        upload.record_upload(mask, mask_gpu, net.opt);
        if (submit_and_wait_profiled(upload, profile) != 0)
        {
            std::fprintf(stderr, "make_dit_input_patches_gpu: mask upload failed\n");
            return false;
        }
    }
    ncnn::VkMat mask_pack1;
    if (!unpack_gpu_to_pack1(mask_gpu, vkdev, net.opt, mask_pack1, profile) || mask_pack1.elempack != 1)
    {
        std::fprintf(stderr, "make_dit_input_patches_gpu: mask packing failed\n");
        return false;
    }

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    if (!prepare_ncnn_layer_benchmark(compute, net))
        return false;
    const int noise_ret = extractor.input("noise", noise_pack1);
    const int condition_ret = extractor.input("condition", condition_pack1);
    const int mask_ret = extractor.input("mask", mask_pack1);
    ncnn::VkMat packed_patches;
    const int extract_ret = noise_ret == 0 && condition_ret == 0 && mask_ret == 0
                                ? extractor.extract("patches", packed_patches, compute)
                                : -1;
    const int submit_ret = extract_ret == 0 ? submit_and_wait_profiled(compute, profile) : -1;
    const bool unpack_ret = submit_ret == 0 && unpack_gpu_to_pack1(packed_patches, vkdev, net.opt, patches, profile);
    const bool valid_output = unpack_ret && !patches.empty() && patches.dims == 2 && patches.w == kVideoPatchWidth &&
                              patches.h == plan.video_tokens && patches.elempack == 1;
    if (noise_ret != 0 || condition_ret != 0 || mask_ret != 0 || extract_ret != 0 || submit_ret != 0 || !valid_output)
    {
        std::fprintf(stderr,
                     "make_dit_input_patches_gpu: input=%d/%d/%d extract=%d submit=%d "
                     "mask=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d) output=(dims=%d,w=%d,h=%d,d=%d,c=%d,pack=%d)\n",
                     noise_ret, condition_ret, mask_ret, extract_ret, submit_ret, mask_pack1.dims, mask_pack1.w,
                     mask_pack1.h, mask_pack1.d, mask_pack1.c, mask_pack1.elempack, patches.dims, patches.w,
                     patches.h, patches.d, patches.c, patches.elempack);
    }
    return noise_ret == 0 && condition_ret == 0 && mask_ret == 0 && extract_ret == 0 && submit_ret == 0 &&
           valid_output;
}

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     const ResolutionPlan& plan,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches,
                                     ncnn::PipelineCache* pipeline_cache,
                                     const PerformanceProfile* profile,
                                     bool use_fp32)
{
    if (!vkdev || !blob_allocator || !staging_allocator)
        return false;

    ncnn::Net net;
    std::ostringstream param;
    param << "7767517\n4 4\n"
          << "Input latent 0 1 latent\n"
          << "Reorg patchify 1 1 latent patch_grid 0=2 1=1\n"
          << "Permute token_major 1 1 patch_grid token_grid 0=3\n"
          << "Reshape flatten 1 1 token_grid patches 0=64 1=" << plan.video_tokens << "\n";
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator, pipeline_cache,
                               profile, use_fp32))
        return false;

    ncnn::VkMat latent_pack1;
    if (!unpack_gpu_to_pack1(latent, vkdev, net.opt, latent_pack1, profile) ||
        !is_plan_latent(latent_pack1, plan, use_fp32))
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat packed_patches;
    if (!prepare_ncnn_layer_benchmark(compute, net) || extractor.input("latent", latent_pack1) != 0 ||
        extractor.extract("patches", packed_patches, compute) != 0 ||
        submit_and_wait_profiled(compute, profile) != 0 ||
        !unpack_gpu_to_pack1(packed_patches, vkdev, net.opt, patches, profile))
        return false;
    return !patches.empty() && patches.dims == 2 && patches.w == kOutputPatchWidth &&
           patches.h == plan.video_tokens && patches.elempack == 1;
}

bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            const ResolutionPlan& plan,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent,
                            ncnn::PipelineCache* pipeline_cache,
                            const PerformanceProfile* profile,
                            bool use_fp32)
{
    if (!vkdev || !blob_allocator || !staging_allocator)
        return false;

    ncnn::Net net;
    std::ostringstream param;
    param << "7767517\n5 5\n"
          << "Input patches 0 1 patches\n"
          << "Permute transpose 1 1 patches transposed 0=1\n"
          << "Reshape patch_grid 1 1 transposed patch_grid 0=" << plan.source_width << " 1="
          << plan.source_height << " 2=64\n"
          << "ShuffleChannel channel_major 1 1 patch_grid shuffled 0=4\n"
          << "PixelShuffle unpatch 1 1 shuffled latent 0=2 1=0\n";
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator, pipeline_cache,
                               profile, use_fp32))
        return false;

    ncnn::VkMat patches_pack1;
    if (!unpack_gpu_to_pack1(patches, vkdev, net.opt, patches_pack1, profile) || patches_pack1.dims != 2 ||
        patches_pack1.w != kOutputPatchWidth || patches_pack1.h != plan.video_tokens)
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat packed_latent;
    if (!prepare_ncnn_layer_benchmark(compute, net) || extractor.input("patches", patches_pack1) != 0 ||
        extractor.extract("latent", packed_latent, compute) != 0 ||
        submit_and_wait_profiled(compute, profile) != 0 ||
        !unpack_gpu_to_pack1(packed_latent, vkdev, net.opt, latent, profile))
        return false;
    return is_plan_latent(latent, plan, use_fp32);
}

bool DitStackSession::run(const ncnn::VkMat& input_patches,
                          const ncnn::Mat& text,
                          float timestep_value,
                          const ResolutionPlan& plan,
                          ncnn::VkMat& output_matrix_gpu,
                          const DitStackObserver* observer) const
{
    if (!impl_ || !impl_->vkdev || !impl_->blob_allocator || !impl_->staging_allocator)
        return false;

    ncnn::VulkanDevice* vkdev = impl_->vkdev;
    ncnn::VkAllocator* blob_allocator = impl_->blob_allocator;
    ncnn::VkAllocator* staging_allocator = impl_->staging_allocator;

    ncnn::Option input_opt;
    input_opt.use_vulkan_compute = true;
    input_opt.use_packing_layout = false;
    input_opt.use_bf16_packed = !impl_->use_fp32;
    input_opt.use_bf16_storage = !impl_->use_fp32;
    input_opt.use_fp16_packed = false;
    input_opt.use_fp16_storage = false;
    input_opt.use_fp16_arithmetic = false;
    input_opt.blob_vkallocator = blob_allocator;
    input_opt.workspace_vkallocator = blob_allocator;
    input_opt.staging_vkallocator = staging_allocator;
    ncnn::VkMat input_patches_pack1;
    if (!unpack_gpu_to_pack1(input_patches, vkdev, input_opt, input_patches_pack1, impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: input patch unpacking failed\n");
        return false;
    }
    if (impl_->fp32_prefix_blocks > 0)
    {
        ncnn::VkMat input_patches_fp32;
        if (!convert_precision_gpu(input_patches_pack1, vkdev, impl_->dit_input.opt, false,
                                   input_patches_fp32, impl_->profile))
        {
            std::fprintf(stderr, "DitStackSession::run: FP32 prefix patch conversion failed\n");
            return false;
        }
        input_patches_pack1 = input_patches_fp32;
    }

    if (input_patches_pack1.empty() || input_patches_pack1.dims != 2 || input_patches_pack1.w != kVideoPatchWidth ||
        input_patches_pack1.h != plan.video_tokens || input_patches_pack1.elempack != 1 || text.dims != 2 ||
        text.w != kTextInputWidth || (text.h != 58 && text.h != 64) || text.elemsize != 4u)
    {
        std::fprintf(stderr,
                     "DitStackSession::run: invalid dynamic stack input patches=%d:%dx%dx%dx%d pack=%d elem=%zu "
                     "text=%d:%dx%d elem=%zu tokens=%d\n",
                     input_patches_pack1.dims, input_patches_pack1.w, input_patches_pack1.h, input_patches_pack1.d,
                     input_patches_pack1.c, input_patches_pack1.elempack, input_patches_pack1.elemsize, text.dims,
                     text.w, text.h, text.elemsize,
                     plan.video_tokens);
        return false;
    }

    ncnn::Mat timestep(1);
    timestep[0] = timestep_value;
    ncnn::Net& dit_input = impl_->dit_input;
    ncnn::Net& dit_embedding = impl_->dit_embedding;
    ncnn::Net& packing = impl_->packing;
    ncnn::Net& input_packing = impl_->fp32_prefix_blocks > 0 ? impl_->fp32_packing : packing;

    ncnn::VkMat video_packed;
    ncnn::VkMat text_packed;
    {
        ncnn::Extractor extractor = dit_input.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (!prepare_ncnn_layer_benchmark(compute, dit_input) ||
            extractor.input("in0", input_patches_pack1) != 0 || extractor.input("in1", text) != 0 ||
            extractor.extract("out0", video_packed, compute) != 0 ||
            extractor.extract("out1", text_packed, compute) != 0 ||
            submit_and_wait_profiled(compute, impl_->profile) != 0)
        {
            std::fprintf(stderr, "DitStackSession::run: dit_input execution failed\n");
            return false;
        }
    }
    ncnn::VkMat video_matrix;
    ncnn::VkMat text_matrix;
    if (!unpack_to_pack1(input_packing, video_packed, vkdev, video_matrix, impl_->profile) ||
        !unpack_to_pack1(input_packing, text_packed, vkdev, text_matrix, impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: input unpacking failed\n");
        return false;
    }

    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    // The PT2 stack graphs are exported from the complete token matrices and
    // therefore accept [tokens, features] directly.  Keep the historical
    // per-token batch path for packages that were exported with that contract,
    // while selecting the matrix path when both input graphs preserve a
    // single 2-D matrix.
    const bool direct_matrix_contract = impl_->direct_matrix_contract && video_matrix.dims == 2 && video_matrix.n == 1 &&
                                        video_matrix.w == kVideoHiddenWidth &&
                                        video_matrix.h == plan.video_tokens && video_matrix.elempack == 1 &&
                                        text_matrix.dims == 2 && text_matrix.n == 1 &&
                                        text_matrix.w == kTextHiddenWidth && text_matrix.h == text.h &&
                                        text_matrix.elempack == 1;
    if (impl_->profile && impl_->profile->enabled())
    {
        std::fprintf(stderr,
                     "dit-input-layout video=%d:%dx%dx%dx%d n=%d text=%d:%dx%dx%dx%d n=%d direct=%d tokens=%d\n",
                     video_matrix.dims, video_matrix.w, video_matrix.h, video_matrix.d, video_matrix.c,
                     video_matrix.n, text_matrix.dims, text_matrix.w, text_matrix.h, text_matrix.d, text_matrix.c,
                     text_matrix.n, direct_matrix_contract ? 1 : 0, plan.video_tokens);
    }
    if (direct_matrix_contract)
    {
        video_gpu = video_matrix;
        text_gpu = text_matrix;
    }
    else if (!matrix_to_batch_gpu(video_matrix, plan.video_tokens, vkdev, dit_input.opt, blob_allocator, video_gpu,
                                  impl_->profile) ||
             !matrix_to_batch_gpu(text_matrix, text.h, vkdev, dit_input.opt, blob_allocator, text_gpu,
                                  impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: input batch conversion failed\n");
        return false;
    }
    if (observer && (!(*observer)("input_video", video_gpu) || !(*observer)("input_text", text_gpu)))
        return false;

    ncnn::VkMat embedding_packed;
    {
        ncnn::Extractor extractor = dit_embedding.create_extractor();
        // Keep the scalar in FP32 until after the sinusoidal prefix. A BF16
        // upload here would round non-integral sampling timesteps too early.
        if (extractor.input("in0", timestep) != 0)
        {
            std::fprintf(stderr, "DitStackSession::run: timestep input failed\n");
            return false;
        }
        ncnn::VkCompute compute(vkdev);
        if (!prepare_ncnn_layer_benchmark(compute, dit_embedding) ||
            extractor.extract("out0", embedding_packed, compute) != 0 ||
            submit_and_wait_profiled(compute, impl_->profile) != 0)
        {
            std::fprintf(stderr, "DitStackSession::run: timestep embedding failed\n");
            return false;
        }
    }
    ncnn::VkMat embedding_gpu;
    if (!unpack_to_pack1(input_packing, embedding_packed, vkdev, embedding_gpu, impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: embedding unpacking failed\n");
        return false;
    }
    if (observer && !(*observer)("embedding", embedding_gpu))
        return false;

    for (int block_index = 0; block_index < 32; block_index++)
    {
        ncnn::Net& block = *impl_->blocks[block_index];
        if (impl_->fp32_prefix_blocks > 0 && block_index == impl_->fp32_prefix_blocks)
        {
            ncnn::VkMat video_bf16;
            ncnn::VkMat text_bf16;
            ncnn::VkMat embedding_bf16;
            if (!convert_precision_gpu(video_gpu, vkdev, block.opt, true, video_bf16, impl_->profile) ||
                !convert_precision_gpu(text_gpu, vkdev, block.opt, true, text_bf16, impl_->profile) ||
                !convert_precision_gpu(embedding_gpu, vkdev, block.opt, true, embedding_bf16, impl_->profile))
            {
                std::fprintf(stderr, "DitStackSession::run: BF16 suffix input conversion failed at block %d\n",
                             block_index);
                return false;
            }
            video_gpu = video_bf16;
            text_gpu = text_bf16;
            embedding_gpu = embedding_bf16;
        }
        ncnn::Extractor extractor = block.create_extractor();
        extractor.set_light_mode(plan.image_width == 256 && plan.image_height == 256);
        ncnn::VkMat next_video;
        ncnn::VkMat next_text;
        ncnn::VkCompute compute(vkdev);
        if (!prepare_ncnn_layer_benchmark(compute, block))
            return false;
        const int video_input_status = extractor.input("in0", video_gpu);
        const int text_input_status = extractor.input("in1", text_gpu);
        const int embedding_input_status = extractor.input("in2", embedding_gpu);
        const int video_output_status = extractor.extract("out0", next_video, compute);
        const int text_output_status = extractor.extract("out1", next_text, compute);
        const int submit_status = submit_and_wait_profiled(compute, impl_->profile);
        if (video_input_status != 0 || text_input_status != 0 || embedding_input_status != 0 ||
            video_output_status != 0 || text_output_status != 0 || submit_status != 0)
        {
            std::fprintf(stderr,
                         "DitStackSession::run: block %d execution failed in=%d/%d/%d out=%d/%d submit=%d\n",
                         block_index, video_input_status, text_input_status, embedding_input_status,
                         video_output_status, text_output_status, submit_status);
            return false;
        }
        video_gpu = next_video;
        text_gpu = next_text;
        if (observer)
        {
            const std::string prefix = "block_" + std::to_string(block_index);
            if (!(*observer)((prefix + "_video").c_str(), video_gpu) ||
                !(*observer)((prefix + "_text").c_str(), text_gpu))
                return false;
        }
    }

    ncnn::Net& dit_output = impl_->dit_output;
    ncnn::VkMat video_unpacked;
    if (!unpack_to_pack1(packing, video_gpu, vkdev, video_unpacked, impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: final video unpacking failed\n");
        return false;
    }
    ncnn::VkMat video_matrix_final;
    if (direct_matrix_contract)
    {
        if (video_unpacked.dims != 2 || video_unpacked.n != 1 || video_unpacked.w != kVideoHiddenWidth ||
            video_unpacked.h != plan.video_tokens || video_unpacked.elempack != 1)
        {
            std::fprintf(stderr,
                         "DitStackSession::run: PT2 matrix output layout mismatch dims=%d w=%d h=%d d=%d c=%d n=%d pack=%d\n",
                         video_unpacked.dims, video_unpacked.w, video_unpacked.h, video_unpacked.d,
                         video_unpacked.c, video_unpacked.n, video_unpacked.elempack);
            return false;
        }
        video_matrix_final = video_unpacked;
    }
    else if (!batch_to_matrix_gpu(video_unpacked, vkdev, dit_output.opt, blob_allocator, video_matrix_final,
                                  impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: final batch conversion failed\n");
        return false;
    }

    ncnn::VkMat output_packed;
    {
        ncnn::Extractor extractor = dit_output.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (!prepare_ncnn_layer_benchmark(compute, dit_output) ||
            extractor.input("in0", video_matrix_final) != 0 || extractor.input("in1", embedding_gpu) != 0 ||
            extractor.extract("out0", output_packed, compute) != 0 ||
            submit_and_wait_profiled(compute, impl_->profile) != 0)
        {
            std::fprintf(stderr, "DitStackSession::run: output graph execution failed\n");
            return false;
        }
    }
    if (!unpack_to_pack1(packing, output_packed, vkdev, output_matrix_gpu, impl_->profile))
    {
        std::fprintf(stderr, "DitStackSession::run: output unpacking failed\n");
        return false;
    }
    return !observer || (*observer)("output", output_matrix_gpu);
}

} // namespace seedvr2
