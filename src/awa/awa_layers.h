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

class SeedVR2AWAPack final : public ncnn::Layer
{
public:
    SeedVR2AWAPack();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

private:
    int source_t_ = 0;
    int source_h_ = 0;
    int source_w_ = 0;
    int windows_t_ = 0;
    int windows_h_ = 0;
    int windows_w_ = 0;
    int text_tokens_ = 0;
    bool shifted_ = false;
    int source_tokens_ = 0;
    std::vector<int> sequence_index_;
    std::vector<int> cu_seqlens_;

#if NCNN_VULKAN
    ncnn::Mat mapping_cpu_;
    ncnn::VkMat mapping_gpu_;
    ncnn::Pipeline* pipeline_ = 0;
#endif
};

class SeedVR2AWAUnpack final : public ncnn::Layer
{
public:
    SeedVR2AWAUnpack();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override;
#endif

private:
    int source_t_ = 0;
    int source_h_ = 0;
    int source_w_ = 0;
    int windows_t_ = 0;
    int windows_h_ = 0;
    int windows_w_ = 0;
    int text_tokens_ = 0;
    bool shifted_ = false;
    int source_tokens_ = 0;
    int sequence_tokens_ = 0;
    int window_count_ = 0;
    std::vector<int> video_positions_;
    std::vector<int> text_positions_;

#if NCNN_VULKAN
    ncnn::Mat mapping_cpu_;
    ncnn::VkMat mapping_gpu_;
    ncnn::Pipeline* pipeline_video_ = 0;
    ncnn::Pipeline* pipeline_text_ = 0;
#endif
};

void register_seedvr2_awa_layers(ncnn::Net& net);
