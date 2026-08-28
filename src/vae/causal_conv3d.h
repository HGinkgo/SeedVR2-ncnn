#pragma once

#include "layer/convolution3d.h"

class SeedVR2CausalConv3D final : public ncnn::Convolution3D
{
public:
    SeedVR2CausalConv3D();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const ncnn::VkMat& bottom_blob,
                ncnn::VkMat& top_blob,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

private:
    int prepend_ = 0;

#if NCNN_VULKAN
    ncnn::VkMat weight_data_gpu;
    ncnn::VkMat bias_data_gpu;
    ncnn::Pipeline* pipeline_ = 0;
#endif
};

ncnn::Layer* SeedVR2CausalConv3D_layer_creator(void* userdata);
