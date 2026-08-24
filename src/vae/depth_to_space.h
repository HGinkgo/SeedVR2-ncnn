#pragma once

#include "layer.h"

#if NCNN_VULKAN
#include "command.h"
#include "pipeline.h"
#endif

namespace ncnn
{
class Net;
}

class SeedVR2DepthToSpace final : public ncnn::Layer
{
public:
    SeedVR2DepthToSpace();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const ncnn::VkMat& bottom_blob,
                ncnn::VkMat& top_blob,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

private:
    int x_ = 0;
    int y_ = 0;
    int z_ = 0;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_ = 0;
#endif
};

ncnn::Layer* SeedVR2DepthToSpace_layer_creator(void* userdata);

void register_seedvr2_vae_layers(ncnn::Net& net);
