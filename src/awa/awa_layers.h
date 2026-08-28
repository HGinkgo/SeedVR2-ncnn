#pragma once

#include <vector>

#include "layer.h"

#if NCNN_VULKAN
#include "command.h"
#include "pipeline.h"
#endif

namespace ncnn
{
class Net;
}

namespace seedvr2
{

struct AwaWindow final
{
    int t0 = 0;
    int t1 = 0;
    int h0 = 0;
    int h1 = 0;
    int w0 = 0;
    int w1 = 0;
};

// Build clipped windows in the same order used by the custom AWA layers.
std::vector<AwaWindow> make_awa_windows(int source_t, int source_h, int source_w,
                                        int windows_t, int windows_h, int windows_w, bool shifted);

} // namespace seedvr2

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

class SeedVR2MMRoPE final : public ncnn::Layer
{
public:
    SeedVR2MMRoPE();

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
    int rope_dim_ = 0;
    int sequence_tokens_ = 0;
    ncnn::Mat rotations_cpu_;

#if NCNN_VULKAN
    ncnn::VkMat rotations_gpu_;
    ncnn::Pipeline* pipeline_ = 0;
#endif
};

class SeedVR2WindowAttention final : public ncnn::Layer
{
public:
    SeedVR2WindowAttention();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override;

#if NCNN_VULKAN
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
    int sequence_tokens_ = 0;
    std::vector<int> window_offsets_;

#if NCNN_VULKAN
    ncnn::Pipeline* score_pipeline_ = 0;
    ncnn::Pipeline* softmax_pipeline_ = 0;
    ncnn::Pipeline* value_pipeline_ = 0;
#endif
};

void register_seedvr2_awa_layers(ncnn::Net& net);
