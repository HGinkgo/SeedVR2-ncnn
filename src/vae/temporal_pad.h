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

class SeedVR2TemporalPad final : public ncnn::Layer
{
public:
    SeedVR2TemporalPad();

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
    int prepend_ = 0;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_ = 0;
#endif
};

void register_seedvr2_vae_layers(ncnn::Net& net);
