#include "pointwise_conv3d.h"

#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#include "layer/vulkan/shader/pointwise_conv3d.comp.hex.h"
#endif

SeedVR2PointwiseConv3D::SeedVR2PointwiseConv3D()
{
    support_vulkan = true;
    support_packing = false;
    support_bf16_storage = true;
    support_vulkan_packing = false;
}

int SeedVR2PointwiseConv3D::load_param(const ncnn::ParamDict& pd)
{
    const int result = ncnn::Convolution3D::load_param(pd);
    return result == 0 && kernel_w == 1 && kernel_h == 1 && kernel_d == 1 &&
                   stride_w == 1 && stride_h == 1 && stride_d == 1 &&
                   dilation_w == 1 && dilation_h == 1 && dilation_d == 1 &&
                   pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0 &&
                   pad_front == 0 && pad_behind == 0
               ? 0
               : -1;
}

int SeedVR2PointwiseConv3D::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                                     const ncnn::Option& opt) const
{
    if (!bias_term || bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.elempack != 1 ||
        (bottom_blob.elemsize != 2u && bottom_blob.elemsize != 4u) || bottom_blob.empty())
        return -1;

    top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, num_output,
                    bottom_blob.elemsize, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int input_size = bottom_blob.w * bottom_blob.h * bottom_blob.d;
    const float* weights = static_cast<const float*>(weight_data.data);
    const float* bias = static_cast<const float*>(bias_data.data);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int output_channel = 0; output_channel < num_output; output_channel++)
    {
        void* output_data = top_blob.channel(output_channel).data;
        const float* channel_weights = weights + static_cast<size_t>(output_channel) * bottom_blob.c;
        for (int index = 0; index < input_size; index++)
        {
            float sum = bias[output_channel];
            for (int input_channel = 0; input_channel < bottom_blob.c; input_channel++)
            {
                float value;
                const ncnn::Mat input = bottom_blob.channel(input_channel);
                if (bottom_blob.elemsize == 2u)
                    value = ncnn::bfloat16_to_float32(static_cast<const unsigned short*>(input.data)[index]);
                else
                    value = static_cast<const float*>(input.data)[index];
                sum += value * channel_weights[input_channel];
            }
            if (bottom_blob.elemsize == 2u)
                static_cast<unsigned short*>(output_data)[index] = ncnn::float32_to_bfloat16(sum);
            else
                static_cast<float*>(output_data)[index] = sum;
        }
    }
    return 0;
}

#if NCNN_VULKAN
int SeedVR2PointwiseConv3D::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (weight_data.empty())
        return -1;

    ncnn::Option fp32_opt = opt;
    fp32_opt.use_bf16_packed = false;
    fp32_opt.use_bf16_storage = false;
    fp32_opt.use_fp16_packed = false;
    fp32_opt.use_fp16_storage = false;
    cmd.record_upload(weight_data, weight_data_gpu, fp32_opt, false);
    if (bias_term)
        cmd.record_upload(bias_data, bias_data_gpu, fp32_opt, false);
    return weight_data_gpu.empty() || (bias_term && bias_data_gpu.empty()) ? -100 : 0;
}

int SeedVR2PointwiseConv3D::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute || !vkdev)
        return 0;

    ncnn::Option shader_opt = opt;
    shader_opt.use_bf16_storage = false;
    shader_opt.use_bf16_packed = true;

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(pointwise_conv3d_comp_data, sizeof(pointwise_conv3d_comp_data), shader_opt, spirv) != 0)
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

int SeedVR2PointwiseConv3D::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_;
    pipeline_ = nullptr;
    weight_data_gpu = ncnn::VkMat();
    bias_data_gpu = ncnn::VkMat();
    return 0;
}

int SeedVR2PointwiseConv3D::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob,
                                     ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 4 || bottom_blob.n != 1 || bottom_blob.elempack != 1 ||
        bottom_blob.elemsize != 2u || bottom_blob.empty() || pipeline_ == nullptr ||
        weight_data_gpu.empty() || (bias_term && bias_data_gpu.empty()))
        return -1;

    top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, num_output,
                    bottom_blob.elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;
    std::vector<ncnn::vk_constant_type> constants(8);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.d;
    constants[3].i = bottom_blob.c;
    constants[4].i = static_cast<int>(bottom_blob.cstep);
    constants[5].i = num_output;
    constants[6].i = bias_term;
    constants[7].i = static_cast<int>(top_blob.cstep);
    ncnn::VkMat dispatcher;
    dispatcher.w = bottom_blob.w;
    dispatcher.h = bottom_blob.h * bottom_blob.d;
    dispatcher.c = num_output;
    cmd.record_pipeline(pipeline_, bindings, constants, dispatcher);
    return 0;
}
#endif

DEFINE_LAYER_CREATOR(SeedVR2PointwiseConv3D)
