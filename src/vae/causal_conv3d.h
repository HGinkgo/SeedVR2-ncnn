#pragma once

#include "layer/convolution3d.h"

class SeedVR2CausalConv3D final : public ncnn::Convolution3D
{
public:
    SeedVR2CausalConv3D();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

private:
    int prepend_ = 0;
};

ncnn::Layer* SeedVR2CausalConv3D_layer_creator(void* userdata);
