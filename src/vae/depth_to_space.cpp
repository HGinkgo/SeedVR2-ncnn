#include "depth_to_space.h"

#include <cstring>
#include <vector>

#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "layer/vulkan/shader/depth_to_space.comp.hex.h"
#endif

SeedVR2DepthToSpace::SeedVR2DepthToSpace()
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
    support_vulkan = true;
    support_vulkan_packing = false;
}

int SeedVR2DepthToSpace::load_param(const ncnn::ParamDict& pd)
{
    x_ = pd.get(0, 0);
    y_ = pd.get(1, 0);
    z_ = pd.get(2, 0);
    return x_ > 0 && y_ > 0 && z_ > 0 ? 0 : -1;
}

int SeedVR2DepthToSpace::forward(const ncnn::Mat& bottom_blob,
                                 ncnn::Mat& top_blob,
                                 const ncnn::Option& opt) const
{
    const int block = x_ * y_ * z_;
    if (bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.elempack != 1 || bottom_blob.empty() ||
        block <= 0 || bottom_blob.c % block != 0)
        return -1;

    const int output_channels = bottom_blob.c / block;
    top_blob.create(bottom_blob.w * y_, bottom_blob.h * x_, bottom_blob.d * z_, output_channels,
                    bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const size_t element_bytes = bottom_blob.elemsize;
    for (int channel = 0; channel < output_channels; channel++)
        for (int depth = 0; depth < top_blob.d; depth++)
            for (int row = 0; row < top_blob.h; row++)
                for (int column = 0; column < top_blob.w; column++)
                {
                    const int source_depth = depth / z_;
                    const int source_z = depth % z_;
                    const int source_row = row / x_;
                    const int source_x = row % x_;
                    const int source_column = column / y_;
                    const int source_y = column % y_;
                    const int source_channel = (((source_x * y_) + source_y) * z_ + source_z) * output_channels + channel;
                    const unsigned char* source = reinterpret_cast<const unsigned char*>(bottom_blob.channel(source_channel).depth(source_depth).row(source_row)) +
                                                  static_cast<size_t>(source_column) * element_bytes;
                    unsigned char* destination = reinterpret_cast<unsigned char*>(top_blob.channel(channel).depth(depth).row(row)) +
                                                  static_cast<size_t>(column) * element_bytes;
                    std::memcpy(destination, source, element_bytes);
                }
    return 0;
}

#if NCNN_VULKAN
int SeedVR2DepthToSpace::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute || !vkdev)
        return 0;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(depth_to_space_comp_data, sizeof(depth_to_space_comp_data), opt, spirv) != 0)
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

int SeedVR2DepthToSpace::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_;
    pipeline_ = 0;
    return 0;
}

int SeedVR2DepthToSpace::forward(const ncnn::VkMat& bottom_blob,
                                 ncnn::VkMat& top_blob,
                                 ncnn::VkCompute& cmd,
                                 const ncnn::Option& opt) const
{
    const int block = x_ * y_ * z_;
    // ncnn may pass a packed VkMat only when packing is explicitly enabled; this
    // fixed VAE path intentionally accepts pack1 tensors.
    if (bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.elempack != 1 || bottom_blob.empty() ||
        block <= 0 || bottom_blob.c % block != 0 || pipeline_ == 0)
        return -1;

    const int output_channels = bottom_blob.c / block;
    top_blob.create(bottom_blob.w * y_, bottom_blob.h * x_, bottom_blob.d * z_, output_channels,
                    bottom_blob.elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    std::vector<ncnn::vk_constant_type> constants(11);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.d;
    constants[3].i = output_channels;
    constants[4].i = x_;
    constants[5].i = y_;
    constants[6].i = z_;
    constants[7].i = static_cast<int>(bottom_blob.cstep);
    constants[8].i = static_cast<int>(top_blob.cstep);
    constants[9].i = top_blob.w;
    constants[10].i = top_blob.h;
    cmd.record_pipeline(pipeline_, bindings, constants, top_blob);
    return 0;
}
#endif

DEFINE_LAYER_CREATOR(SeedVR2DepthToSpace)
