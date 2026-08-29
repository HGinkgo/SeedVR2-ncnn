#include "dit/dit_stack.h"

#include <cstdio>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "awa/awa_layers.h"
#include "datareader.h"
#include "net.h"

namespace seedvr2
{
namespace
{

constexpr int kLatentChannels = 16;
constexpr int kPatchSize = 2;
constexpr int kVideoPatchWidth = 132;
constexpr int kTextInputWidth = 5120;
constexpr int kOutputPatchWidth = 64;

void configure(ncnn::Net& net, ncnn::VulkanDevice* vkdev, ncnn::VkAllocator* blob_allocator,
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
    net.set_vulkan_device(vkdev);
}

bool load_graph(ncnn::Net& net, const std::string& stem, ncnn::VulkanDevice* vkdev,
                ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    register_seedvr2_awa_layers(net);
    const std::string param_path = stem + ".ncnn.param";
    const std::string model_path = stem + ".ncnn.bin";
    if (net.load_param(param_path.c_str()) != 0)
    {
        std::fprintf(stderr, "load_graph: failed to load %s\n", param_path.c_str());
        return false;
    }
    if (net.load_model(model_path.c_str()) != 0)
    {
        std::fprintf(stderr, "load_graph: failed to load %s\n", model_path.c_str());
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

bool load_graph_from_param(ncnn::Net& net, const char* param, ncnn::VulkanDevice* vkdev,
                           ncnn::VkAllocator* blob_allocator, ncnn::VkAllocator* staging_allocator)
{
    configure(net, vkdev, blob_allocator, staging_allocator);
    if (net.load_param_mem(param) != 0)
        return false;
    const unsigned char* empty_model = nullptr;
    ncnn::DataReaderFromMemory model_reader(empty_model);
    return net.load_model(model_reader) == 0;
}

bool is_plan_latent(const ncnn::VkMat& value, const ResolutionPlan& plan)
{
    return !value.empty() && (value.dims == 3 || value.dims == 4) && value.w == plan.latent_width &&
           value.h == plan.latent_height && value.d == 1 && value.c == kLatentChannels && value.elempack == 1 &&
           value.elemsize == 4u;
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
                         ncnn::VkMat& output)
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
    return !output.empty() && compute.submit_and_wait() == 0;
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

} // namespace

struct DitStackSession::Impl
{
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    ncnn::Net dit_input;
    ncnn::Net dit_embedding;
    ncnn::Net packing;
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
        dit_embedding.clear();
        dit_input.clear();
    }
};

DitStackSession::DitStackSession() = default;

DitStackSession::~DitStackSession()
{
    if (impl_)
        impl_->clear();
}

DitStackSession::DitStackSession(DitStackSession&&) noexcept = default;

DitStackSession& DitStackSession::operator=(DitStackSession&&) noexcept = default;

bool DitStackSession::open(const std::string& stack_dir,
                           const ResolutionPlan& plan,
                           ncnn::VulkanDevice* vkdev,
                           ncnn::VkAllocator* blob_allocator,
                           ncnn::VkAllocator* staging_allocator,
                           DitStackSession& session)
{
    if (!vkdev || !blob_allocator || !staging_allocator || plan.video_tokens <= 0)
        return false;

    std::unique_ptr<Impl> candidate(new Impl);
    candidate->vkdev = vkdev;
    candidate->blob_allocator = blob_allocator;
    candidate->staging_allocator = staging_allocator;
    if (!load_graph(candidate->dit_input, stack_dir + "/dit_input", vkdev, blob_allocator, staging_allocator) ||
        !load_graph(candidate->dit_embedding, stack_dir + "/dit_embedding", vkdev, blob_allocator,
                    staging_allocator) ||
        !load_packing_graph(candidate->packing, vkdev, blob_allocator, staging_allocator))
        return false;

    candidate->blocks.reserve(32);
    for (int block_index = 0; block_index < 32; block_index++)
    {
        std::unique_ptr<ncnn::Net> block(new ncnn::Net);
        const std::string block_name = stack_dir + "/dit_block_" +
                                       (block_index < 10 ? "0" : "") + std::to_string(block_index);
        if (!load_graph(*block, block_name, vkdev, blob_allocator, staging_allocator))
            return false;
        candidate->blocks.push_back(std::move(block));
    }
    if (!load_graph(candidate->dit_output, stack_dir + "/dit_output", vkdev, blob_allocator, staging_allocator))
        return false;

    session.impl_ = std::move(candidate);
    return true;
}

bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                const ResolutionPlan& plan,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches)
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
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator))
    {
        std::fprintf(stderr, "make_dit_input_patches_gpu: graph load failed\n");
        return false;
    }

    ncnn::VkMat noise_pack1;
    ncnn::VkMat condition_pack1;
    if (!unpack_gpu_to_pack1(noise, vkdev, net.opt, noise_pack1) ||
        !unpack_gpu_to_pack1(condition, vkdev, net.opt, condition_pack1))
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
    if (!is_plan_latent(noise_pack1, plan) || !is_plan_latent(condition_pack1, plan) ||
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
        upload.record_upload(mask, mask_gpu, net.opt);
        if (upload.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "make_dit_input_patches_gpu: mask upload failed\n");
            return false;
        }
    }
    ncnn::VkMat mask_pack1;
    if (!unpack_gpu_to_pack1(mask_gpu, vkdev, net.opt, mask_pack1) || mask_pack1.elempack != 1)
    {
        std::fprintf(stderr, "make_dit_input_patches_gpu: mask packing failed\n");
        return false;
    }

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    const int noise_ret = extractor.input("noise", noise_pack1);
    const int condition_ret = extractor.input("condition", condition_pack1);
    const int mask_ret = extractor.input("mask", mask_pack1);
    ncnn::VkMat packed_patches;
    const int extract_ret = noise_ret == 0 && condition_ret == 0 && mask_ret == 0
                                ? extractor.extract("patches", packed_patches, compute)
                                : -1;
    const int submit_ret = extract_ret == 0 ? compute.submit_and_wait() : -1;
    const bool unpack_ret = submit_ret == 0 && unpack_gpu_to_pack1(packed_patches, vkdev, net.opt, patches);
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

bool make_dit_input_patches_gpu(const ncnn::VkMat& noise,
                                const ncnn::VkMat& condition,
                                ncnn::VulkanDevice* vkdev,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator,
                                ncnn::VkMat& patches)
{
    ResolutionPlan plan;
    return ResolutionPlan::from_explicit(128, 128, plan) &&
           make_dit_input_patches_gpu(noise, condition, plan, vkdev, blob_allocator, staging_allocator, patches);
}

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     const ResolutionPlan& plan,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches)
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
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator))
        return false;

    ncnn::VkMat latent_pack1;
    if (!unpack_gpu_to_pack1(latent, vkdev, net.opt, latent_pack1) || !is_plan_latent(latent_pack1, plan))
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat packed_patches;
    if (extractor.input("latent", latent_pack1) != 0 || extractor.extract("patches", packed_patches, compute) != 0 ||
        compute.submit_and_wait() != 0 || !unpack_gpu_to_pack1(packed_patches, vkdev, net.opt, patches))
        return false;
    return !patches.empty() && patches.dims == 2 && patches.w == kOutputPatchWidth &&
           patches.h == plan.video_tokens && patches.elempack == 1;
}

bool patch_latent_for_dit_output_gpu(const ncnn::VkMat& latent,
                                     ncnn::VulkanDevice* vkdev,
                                     ncnn::VkAllocator* blob_allocator,
                                     ncnn::VkAllocator* staging_allocator,
                                     ncnn::VkMat& patches)
{
    ResolutionPlan plan;
    return ResolutionPlan::from_explicit(128, 128, plan) &&
           patch_latent_for_dit_output_gpu(latent, plan, vkdev, blob_allocator, staging_allocator, patches);
}

bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            const ResolutionPlan& plan,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent)
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
    if (!load_graph_from_param(net, param.str().c_str(), vkdev, blob_allocator, staging_allocator))
        return false;

    ncnn::VkMat patches_pack1;
    if (!unpack_gpu_to_pack1(patches, vkdev, net.opt, patches_pack1) || patches_pack1.dims != 2 ||
        patches_pack1.w != kOutputPatchWidth || patches_pack1.h != plan.video_tokens)
        return false;

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(false);
    ncnn::VkCompute compute(vkdev);
    ncnn::VkMat packed_latent;
    if (extractor.input("patches", patches_pack1) != 0 || extractor.extract("latent", packed_latent, compute) != 0 ||
        compute.submit_and_wait() != 0 || !unpack_gpu_to_pack1(packed_latent, vkdev, net.opt, latent))
        return false;
    return is_plan_latent(latent, plan);
}

bool unpatch_dit_output_gpu(const ncnn::VkMat& patches,
                            ncnn::VulkanDevice* vkdev,
                            ncnn::VkAllocator* blob_allocator,
                            ncnn::VkAllocator* staging_allocator,
                            ncnn::VkMat& latent)
{
    ResolutionPlan plan;
    return ResolutionPlan::from_explicit(128, 128, plan) &&
           unpatch_dit_output_gpu(patches, plan, vkdev, blob_allocator, staging_allocator, latent);
}

bool run_dit_stack_gpu(const ncnn::Mat& latent_input,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       const ResolutionPlan& plan,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu)
{
    if (latent_input.empty() || (latent_input.dims != 3 && latent_input.dims != 4) || latent_input.d != 1 ||
        latent_input.c != kLatentChannels || latent_input.w != plan.latent_width || latent_input.h != plan.latent_height ||
        latent_input.elemsize != 4u || !vkdev || !blob_allocator || !staging_allocator)
        return false;

    ncnn::Mat patches(kVideoPatchWidth, plan.video_tokens);
    if (patches.empty())
        return false;
    patches.fill(0.f);
    for (int patch_y = 0; patch_y < plan.latent_height / kPatchSize; patch_y++)
        for (int patch_x = 0; patch_x < plan.latent_width / kPatchSize; patch_x++)
        {
            const int token = patch_y * (plan.latent_width / kPatchSize) + patch_x;
            float* output = patches.row(token);
            for (int dy = 0; dy < kPatchSize; dy++)
                for (int dx = 0; dx < kPatchSize; dx++)
                    for (int channel = 0; channel < kLatentChannels; channel++)
                    {
                        const int offset = (dy * kPatchSize + dx) * kLatentChannels + channel;
                        output[offset] = latent_input.channel(channel).row(patch_y * kPatchSize + dy)[patch_x * kPatchSize + dx];
                    }
            for (int offset = kLatentChannels; offset < kVideoPatchWidth; offset++)
                output[offset] = offset == kVideoPatchWidth - 1 ? 1.f : 0.f;
        }

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    ncnn::VkMat patches_gpu;
    {
        ncnn::VkCompute compute(vkdev);
        compute.record_upload(patches, patches_gpu, opt);
        if (compute.submit_and_wait() != 0)
            return false;
    }
    return run_dit_stack_gpu(patches_gpu, text, timestep_value, stack_dir, plan, vkdev, blob_allocator,
                             staging_allocator, output_matrix_gpu);
}

bool run_dit_stack_gpu(const ncnn::Mat& latent_input,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu)
{
    ResolutionPlan plan;
    return ResolutionPlan::from_explicit(128, 128, plan) &&
           run_dit_stack_gpu(latent_input, text, timestep_value, stack_dir, plan, vkdev, blob_allocator,
                             staging_allocator, output_matrix_gpu);
}

bool DitStackSession::run(const ncnn::VkMat& input_patches,
                          const ncnn::Mat& text,
                          float timestep_value,
                          const ResolutionPlan& plan,
                          ncnn::VkMat& output_matrix_gpu) const
{
    if (!impl_ || !impl_->vkdev || !impl_->blob_allocator || !impl_->staging_allocator)
        return false;

    ncnn::VulkanDevice* vkdev = impl_->vkdev;
    ncnn::VkAllocator* blob_allocator = impl_->blob_allocator;
    ncnn::VkAllocator* staging_allocator = impl_->staging_allocator;

    ncnn::Option input_opt;
    input_opt.use_vulkan_compute = true;
    input_opt.use_packing_layout = false;
    input_opt.use_fp16_packed = false;
    input_opt.use_fp16_storage = false;
    input_opt.use_fp16_arithmetic = false;
    input_opt.blob_vkallocator = blob_allocator;
    input_opt.workspace_vkallocator = blob_allocator;
    input_opt.staging_vkallocator = staging_allocator;
    ncnn::VkMat input_patches_pack1;
    if (!unpack_gpu_to_pack1(input_patches, vkdev, input_opt, input_patches_pack1))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: input patch unpacking failed\n");
        return false;
    }

    if (input_patches_pack1.empty() || input_patches_pack1.dims != 2 || input_patches_pack1.w != kVideoPatchWidth ||
        input_patches_pack1.h != plan.video_tokens || input_patches_pack1.elempack != 1 || text.dims != 2 ||
        text.w != kTextInputWidth || (text.h != 58 && text.h != 64) || text.elemsize != 4u)
    {
        std::fprintf(stderr,
                     "run_dit_stack_gpu: invalid dynamic stack input patches=%d:%dx%dx%dx%d pack=%d elem=%zu "
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

    ncnn::VkMat video_packed;
    ncnn::VkMat text_packed;
    {
        ncnn::Extractor extractor = dit_input.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", input_patches_pack1) != 0 || extractor.input("in1", text) != 0 ||
            extractor.extract("out0", video_packed, compute) != 0 ||
            extractor.extract("out1", text_packed, compute) != 0 || compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "run_dit_stack_gpu: dit_input execution failed\n");
            return false;
        }
    }
    ncnn::VkMat video_matrix;
    ncnn::VkMat text_matrix;
    if (!unpack_to_pack1(packing, video_packed, vkdev, video_matrix) ||
        !unpack_to_pack1(packing, text_packed, vkdev, text_matrix))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: input unpacking failed\n");
        return false;
    }

    ncnn::VkMat video_gpu;
    ncnn::VkMat text_gpu;
    if (!matrix_to_batch_gpu(video_matrix, plan.video_tokens, vkdev, dit_input.opt, blob_allocator, video_gpu) ||
        !matrix_to_batch_gpu(text_matrix, text.h, vkdev, dit_input.opt, blob_allocator, text_gpu))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: input batch conversion failed\n");
        return false;
    }

    ncnn::VkMat embedding_packed;
    {
        ncnn::Extractor extractor = dit_embedding.create_extractor();
        ncnn::VkCompute upload(vkdev);
        ncnn::VkMat timestep_gpu;
        upload.record_upload(timestep, timestep_gpu, dit_embedding.opt);
        if (upload.submit_and_wait() != 0 || extractor.input("in0", timestep_gpu) != 0)
        {
            std::fprintf(stderr, "run_dit_stack_gpu: timestep upload failed\n");
            return false;
        }
        ncnn::VkCompute compute(vkdev);
        if (extractor.extract("out0", embedding_packed, compute) != 0 || compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "run_dit_stack_gpu: timestep embedding failed\n");
            return false;
        }
    }
    ncnn::VkMat embedding_gpu;
    if (!unpack_to_pack1(packing, embedding_packed, vkdev, embedding_gpu))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: embedding unpacking failed\n");
        return false;
    }

    for (int block_index = 0; block_index < 32; block_index++)
    {
        ncnn::Net& block = *impl_->blocks[block_index];
        ncnn::Extractor extractor = block.create_extractor();
        extractor.set_light_mode(false);
        ncnn::VkMat next_video;
        ncnn::VkMat next_text;
        ncnn::VkCompute compute(vkdev);
        const int video_input_status = extractor.input("in0", video_gpu);
        const int text_input_status = extractor.input("in1", text_gpu);
        const int embedding_input_status = extractor.input("in2", embedding_gpu);
        const int video_output_status = extractor.extract("out0", next_video, compute);
        const int text_output_status = extractor.extract("out1", next_text, compute);
        const int submit_status = compute.submit_and_wait();
        if (video_input_status != 0 || text_input_status != 0 || embedding_input_status != 0 ||
            video_output_status != 0 || text_output_status != 0 || submit_status != 0)
        {
            std::fprintf(stderr,
                         "run_dit_stack_gpu: block %d execution failed in=%d/%d/%d out=%d/%d submit=%d\n",
                         block_index, video_input_status, text_input_status, embedding_input_status,
                         video_output_status, text_output_status, submit_status);
            return false;
        }
        video_gpu = next_video;
        text_gpu = next_text;
    }

    ncnn::Net& dit_output = impl_->dit_output;
    ncnn::VkMat video_unpacked;
    if (!unpack_to_pack1(packing, video_gpu, vkdev, video_unpacked))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: final video unpacking failed\n");
        return false;
    }
    ncnn::VkMat video_matrix_final;
    if (!batch_to_matrix_gpu(video_unpacked, vkdev, dit_output.opt, blob_allocator, video_matrix_final))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: final batch conversion failed\n");
        return false;
    }

    ncnn::VkMat output_packed;
    {
        ncnn::Extractor extractor = dit_output.create_extractor();
        ncnn::VkCompute compute(vkdev);
        if (extractor.input("in0", video_matrix_final) != 0 || extractor.input("in1", embedding_gpu) != 0 ||
            extractor.extract("out0", output_packed, compute) != 0 || compute.submit_and_wait() != 0)
        {
            std::fprintf(stderr, "run_dit_stack_gpu: output graph execution failed\n");
            return false;
        }
    }
    if (!unpack_to_pack1(packing, output_packed, vkdev, output_matrix_gpu))
    {
        std::fprintf(stderr, "run_dit_stack_gpu: output unpacking failed\n");
        return false;
    }
    return true;
}

bool run_dit_stack_gpu(const ncnn::VkMat& input_patches,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       const ResolutionPlan& plan,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu)
{
    DitStackSession session;
    return DitStackSession::open(stack_dir, plan, vkdev, blob_allocator, staging_allocator, session) &&
           session.run(input_patches, text, timestep_value, plan, output_matrix_gpu);
}

bool run_dit_stack_gpu(const ncnn::VkMat& input_patches,
                       const ncnn::Mat& text,
                       float timestep_value,
                       const std::string& stack_dir,
                       ncnn::VulkanDevice* vkdev,
                       ncnn::VkAllocator* blob_allocator,
                       ncnn::VkAllocator* staging_allocator,
                       ncnn::VkMat& output_matrix_gpu)
{
    ResolutionPlan plan;
    return ResolutionPlan::from_explicit(128, 128, plan) &&
           run_dit_stack_gpu(input_patches, text, timestep_value, stack_dir, plan, vkdev, blob_allocator,
                             staging_allocator, output_matrix_gpu);
}

} // namespace seedvr2
