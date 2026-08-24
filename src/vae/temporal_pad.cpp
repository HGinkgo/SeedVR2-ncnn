#include "temporal_pad.h"
#include "depth_to_space.h"

#include <cstring>
#include <vector>

#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "layer/vulkan/shader/temporal_pad.comp.hex.h"
#endif

SeedVR2TemporalPad::SeedVR2TemporalPad()
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
    support_vulkan = true;
    support_vulkan_packing = false;
}

int SeedVR2TemporalPad::load_param(const ncnn::ParamDict& pd)
{
    prepend_ = pd.get(0, 0);
    return prepend_ > 0 ? 0 : -1;
}

int SeedVR2TemporalPad::forward(const ncnn::Mat& bottom_blob,
                                 ncnn::Mat& top_blob,
                                 const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.empty() || prepend_ <= 0)
        return -1;

    top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d + prepend_, bottom_blob.c,
                    bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const size_t plane_bytes = static_cast<size_t>(bottom_blob.w) * bottom_blob.h * bottom_blob.elemsize;
    for (int channel = 0; channel < bottom_blob.c; channel++)
    {
        const void* first_frame = bottom_blob.channel(channel).depth(0).data;
        for (int depth = 0; depth < prepend_; depth++)
            std::memcpy(top_blob.channel(channel).depth(depth).data, first_frame, plane_bytes);
        for (int depth = 0; depth < bottom_blob.d; depth++)
            std::memcpy(top_blob.channel(channel).depth(depth + prepend_).data,
                        bottom_blob.channel(channel).depth(depth).data, plane_bytes);
    }
    return 0;
}

#if NCNN_VULKAN
int SeedVR2TemporalPad::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute || !vkdev)
        return 0;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(temporal_pad_comp_data, sizeof(temporal_pad_comp_data), opt, spirv) != 0)
        return -1;

    ncnn::Pipeline* candidate = new ncnn::Pipeline(vkdev);
    candidate->set_optimal_local_size_xyz(4, 4, 4);
    if (candidate->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        delete candidate;
        return -1;
    }
    pipeline_ = candidate;
    return 0;
}

int SeedVR2TemporalPad::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_;
    pipeline_ = 0;
    return 0;
}

int SeedVR2TemporalPad::forward(const ncnn::VkMat& bottom_blob,
                                 ncnn::VkMat& top_blob,
                                 ncnn::VkCompute& cmd,
                                 const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.empty() || prepend_ <= 0 || pipeline_ == 0)
        return -1;

    top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d + prepend_, bottom_blob.c,
                    bottom_blob.elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.d;
    constants[3].i = bottom_blob.c;
    constants[4].i = static_cast<int>(bottom_blob.cstep);
    constants[5].i = static_cast<int>(top_blob.cstep);
    constants[6].i = prepend_;
    cmd.record_pipeline(pipeline_, bindings, constants, top_blob);
    return 0;
}
#endif

DEFINE_LAYER_CREATOR(SeedVR2TemporalPad)

void register_seedvr2_vae_layers(ncnn::Net& net)
{
    net.register_custom_layer("SeedVR2TemporalPad", SeedVR2TemporalPad_layer_creator);
    net.register_custom_layer("SeedVR2DepthToSpace", SeedVR2DepthToSpace_layer_creator);
}
