#include "awa_layers.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "layer/vulkan/shader/awa_pack.comp.hex.h"
#include "layer/vulkan/shader/awa_unpack.comp.hex.h"
#endif

namespace
{

struct Window
{
    int t0;
    int t1;
    int h0;
    int h1;
    int w0;
    int w1;
};

int ceil_div(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

int python_round_positive(double value)
{
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5)
        return static_cast<int>(lower);
    if (fraction > 0.5)
        return static_cast<int>(lower + 1.0);
    const int lower_int = static_cast<int>(lower);
    return lower_int % 2 == 0 ? lower_int : lower_int + 1;
}

std::vector<Window> make_windows(int source_t, int source_h, int source_w,
                                 int num_t, int num_h, int num_w, bool shifted)
{
    const double scale = std::sqrt(3600.0 / (source_h * source_w));
    const int resized_h = python_round_positive(source_h * scale);
    const int resized_w = python_round_positive(source_w * scale);
    const int window_h = ceil_div(resized_h, num_h);
    const int window_w = ceil_div(resized_w, num_w);
    const int window_t = ceil_div(std::min(source_t, 30), num_t);
    const double shift_t = shifted && window_t < source_t ? 0.5 : 0.0;
    const double shift_h = shifted && window_h < source_h ? 0.5 : 0.0;
    const double shift_w = shifted && window_w < source_w ? 0.5 : 0.0;
    const int count_t = static_cast<int>(std::ceil((source_t - shift_t) / window_t)) + (shift_t > 0 ? 1 : 0);
    const int count_h = static_cast<int>(std::ceil((source_h - shift_h) / window_h)) + (shift_h > 0 ? 1 : 0);
    const int count_w = static_cast<int>(std::ceil((source_w - shift_w) / window_w)) + (shift_w > 0 ? 1 : 0);

    std::vector<Window> windows;
    for (int iw = 0; iw < count_w; iw++)
    {
        for (int ih = 0; ih < count_h; ih++)
        {
            for (int it = 0; it < count_t; it++)
            {
                const int t0 = std::max(static_cast<int>((it - shift_t) * window_t), 0);
                const int h0 = std::max(static_cast<int>((ih - shift_h) * window_h), 0);
                const int w0 = std::max(static_cast<int>((iw - shift_w) * window_w), 0);
                const int t1 = std::min(static_cast<int>((it - shift_t + 1.0) * window_t), source_t);
                const int h1 = std::min(static_cast<int>((ih - shift_h + 1.0) * window_h), source_h);
                const int w1 = std::min(static_cast<int>((iw - shift_w + 1.0) * window_w), source_w);
                if (t1 > t0 && h1 > h0 && w1 > w0)
                    windows.push_back({t0, t1, h0, h1, w0, w1});
            }
        }
    }
    return windows;
}

bool load_common(const ncnn::ParamDict& pd, int& source_t, int& source_h, int& source_w,
                 int& windows_t, int& windows_h, int& windows_w, int& text_tokens, bool& shifted)
{
    source_t = pd.get(0, 0);
    source_h = pd.get(1, 0);
    source_w = pd.get(2, 0);
    windows_t = pd.get(3, 0);
    windows_h = pd.get(4, 0);
    windows_w = pd.get(5, 0);
    text_tokens = pd.get(6, 0);
    shifted = pd.get(7, 0) != 0;
    return source_t > 0 && source_h > 0 && source_w > 0 && windows_t > 0 &&
           windows_h > 0 && windows_w > 0 && text_tokens > 0;
}

bool is_qkv_input(const ncnn::Mat& blob, int source_tokens, int& heads, int& head_dim)
{
    if (blob.elemsize != 4u || blob.elempack != 1 || blob.w <= 0 || blob.h <= 0)
        return false;
    head_dim = blob.w;
    if (blob.dims == 3)
    {
        if (blob.h % 3 != 0 || blob.c != source_tokens)
            return false;
        heads = blob.h / 3;
    }
    else if (blob.dims == 4)
    {
        if (blob.d != 3 || blob.c != source_tokens)
            return false;
        heads = blob.h;
    }
    else
    {
        return false;
    }
    return heads > 0;
}

#if NCNN_VULKAN
bool is_qkv_input(const ncnn::VkMat& blob, int source_tokens, int& heads, int& head_dim)
{
    if ((blob.elemsize != 2u && blob.elemsize != 4u) || blob.elempack != 1 || blob.w <= 0 || blob.h <= 0)
        return false;
    head_dim = blob.w;
    if (blob.dims == 3)
    {
        if (blob.h % 3 != 0 || blob.c != source_tokens)
            return false;
        heads = blob.h / 3;
    }
    else if (blob.dims == 4)
    {
        if (blob.d != 3 || blob.c != source_tokens)
            return false;
        heads = blob.h;
    }
    else
    {
        return false;
    }
    return heads > 0;
}

int create_awa_pipeline(const ncnn::VulkanDevice* vkdev, const char* shader_data, size_t shader_size,
                        const ncnn::Option& opt, int mode, ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader_data, static_cast<int>(shader_size), opt, spirv) != 0)
        return -1;

    std::vector<ncnn::vk_specialization_type> specializations;
    if (mode >= 0)
    {
        specializations.resize(1);
        specializations[0].i = mode;
    }

    ncnn::Pipeline* candidate = new ncnn::Pipeline(vkdev);
    candidate->set_optimal_local_size_xyz(4, 4, 4);
    if (candidate->create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
    {
        delete candidate;
        return -1;
    }
    pipeline = candidate;
    return 0;
}
#endif

} // namespace

SeedVR2AWAPack::SeedVR2AWAPack()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
    support_any_packing = false;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
}

int SeedVR2AWAPack::load_param(const ncnn::ParamDict& pd)
{
    if (!load_common(pd, source_t_, source_h_, source_w_, windows_t_, windows_h_, windows_w_, text_tokens_, shifted_))
        return -1;
    source_tokens_ = source_t_ * source_h_ * source_w_;
    sequence_index_.clear();
    cu_seqlens_.clear();
    cu_seqlens_.push_back(0);
    const std::vector<Window> windows = make_windows(source_t_, source_h_, source_w_, windows_t_, windows_h_, windows_w_, shifted_);
    int cursor = 0;
    for (const Window& window : windows)
    {
        for (int t = window.t0; t < window.t1; t++)
            for (int h = window.h0; h < window.h1; h++)
                for (int w = window.w0; w < window.w1; w++)
                {
                    const int source_index = (t * source_h_ + h) * source_w_ + w;
                    sequence_index_.push_back(source_index);
                    cursor++;
                }
        for (int token = 0; token < text_tokens_; token++)
            sequence_index_.push_back(source_tokens_ + token);
        cu_seqlens_.push_back(cursor + text_tokens_);
        cursor += text_tokens_;
    }
#if NCNN_VULKAN
    mapping_cpu_.create(static_cast<int>((sequence_index_.size() + cu_seqlens_.size()) * sizeof(int)), static_cast<size_t>(1u));
    if (mapping_cpu_.empty())
        return -100;
    std::memcpy(mapping_cpu_.data, sequence_index_.data(), sequence_index_.size() * sizeof(int));
    std::memcpy(static_cast<unsigned char*>(mapping_cpu_.data) + sequence_index_.size() * sizeof(int),
                cu_seqlens_.data(), cu_seqlens_.size() * sizeof(int));
#endif
    return sequence_index_.empty() ? -1 : 0;
}

int SeedVR2AWAPack::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                            std::vector<ncnn::Mat>& top_blobs,
                            const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || sequence_index_.empty())
        return -1;
    const ncnn::Mat& video = bottom_blobs[0];
    const ncnn::Mat& text = bottom_blobs[1];
    int video_heads = 0;
    int video_head_dim = 0;
    int text_heads = 0;
    int text_head_dim = 0;
    if (!is_qkv_input(video, source_tokens_, video_heads, video_head_dim) ||
        !is_qkv_input(text, text_tokens_, text_heads, text_head_dim) ||
        video_heads != text_heads || video_head_dim != text_head_dim)
        return -1;
    top_blobs.resize(2);
    top_blobs[0].create(video_head_dim, 3 * video_heads, static_cast<int>(sequence_index_.size()), 4u, opt.blob_allocator);
    top_blobs[1].create(static_cast<int>(cu_seqlens_.size()), 4u, opt.blob_allocator);
    if (top_blobs[0].empty() || top_blobs[1].empty())
        return -100;
    for (size_t position = 0; position < sequence_index_.size(); position++)
    {
        const int source_index = sequence_index_[position];
        const ncnn::Mat& source = source_index < source_tokens_ ? video : text;
        const int token = source_index < source_tokens_ ? source_index : source_index - source_tokens_;
        float* packed_data = static_cast<float*>(top_blobs[0].channel(static_cast<int>(position)).data);
        const float* source_data = static_cast<const float*>(source.data) +
                                   static_cast<size_t>(token) * source.cstep;
        std::memcpy(packed_data, source_data, 3 * video_heads * video_head_dim * sizeof(float));
    }
    for (size_t i = 0; i < cu_seqlens_.size(); i++)
        top_blobs[1][i] = static_cast<float>(cu_seqlens_[i]);
    return 0;
}

#if NCNN_VULKAN
int SeedVR2AWAPack::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (mapping_cpu_.empty())
        return -1;
    cmd.record_upload(mapping_cpu_, mapping_gpu_, opt);
    return mapping_gpu_.empty() ? -100 : 0;
}

int SeedVR2AWAPack::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute || !vkdev)
        return 0;
    return create_awa_pipeline(vkdev, awa_pack_comp_data, sizeof(awa_pack_comp_data), opt, -1, pipeline_);
}

int SeedVR2AWAPack::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_;
    pipeline_ = 0;
    mapping_gpu_ = ncnn::VkMat();
    return 0;
}

int SeedVR2AWAPack::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                            std::vector<ncnn::VkMat>& top_blobs,
                            ncnn::VkCompute& cmd,
                            const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || sequence_index_.empty() || pipeline_ == 0 || mapping_gpu_.empty())
        return -1;
    const ncnn::VkMat& video = bottom_blobs[0];
    const ncnn::VkMat& text = bottom_blobs[1];
    int video_heads = 0;
    int video_head_dim = 0;
    int text_heads = 0;
    int text_head_dim = 0;
    if (!is_qkv_input(video, source_tokens_, video_heads, video_head_dim) ||
        !is_qkv_input(text, text_tokens_, text_heads, text_head_dim) ||
        video_heads != text_heads || video_head_dim != text_head_dim)
        return -1;

    top_blobs.resize(2);
    top_blobs[0].create(video_head_dim, 3 * video_heads, static_cast<int>(sequence_index_.size()),
                        video.elemsize, 1, opt.blob_vkallocator);
    top_blobs[1].create(static_cast<int>(cu_seqlens_.size()), video.elemsize, 1, opt.blob_vkallocator);
    if (top_blobs[0].empty() || top_blobs[1].empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(5);
    bindings[0] = video;
    bindings[1] = text;
    bindings[2] = top_blobs[0];
    bindings[3] = top_blobs[1];
    bindings[4] = mapping_gpu_;

    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].i = static_cast<int>(video.cstep);
    constants[1].i = static_cast<int>(text.cstep);
    constants[2].i = video_heads;
    constants[3].i = video_head_dim;
    constants[4].i = source_tokens_;
    constants[5].i = static_cast<int>(sequence_index_.size());
    constants[6].i = static_cast<int>(top_blobs[0].cstep);
    constants[7].i = static_cast<int>(sequence_index_.size());
    constants[8].i = static_cast<int>(cu_seqlens_.size());

    cmd.record_pipeline(pipeline_, bindings, constants, top_blobs[0]);
    return 0;
}
#endif

SeedVR2AWAUnpack::SeedVR2AWAUnpack()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
    support_any_packing = false;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
}

int SeedVR2AWAUnpack::load_param(const ncnn::ParamDict& pd)
{
    if (!load_common(pd, source_t_, source_h_, source_w_, windows_t_, windows_h_, windows_w_, text_tokens_, shifted_))
        return -1;
    source_tokens_ = source_t_ * source_h_ * source_w_;
    video_positions_.clear();
    video_positions_.assign(source_tokens_, -1);
    text_positions_.clear();
    const std::vector<Window> windows = make_windows(source_t_, source_h_, source_w_, windows_t_, windows_h_, windows_w_, shifted_);
    std::vector<int> target;
    int cursor = 0;
    for (const Window& window : windows)
    {
        for (int t = window.t0; t < window.t1; t++)
            for (int h = window.h0; h < window.h1; h++)
                for (int w = window.w0; w < window.w1; w++)
                {
                    const int source_index = (t * source_h_ + h) * source_w_ + w;
                    target.push_back(source_index);
                    video_positions_[source_index] = cursor++;
                }
        for (int token = 0; token < text_tokens_; token++)
            text_positions_.push_back(cursor + token);
        cursor += text_tokens_;
    }
    window_count_ = static_cast<int>(windows.size());
    sequence_tokens_ = cursor;
#if NCNN_VULKAN
    mapping_cpu_.create(static_cast<int>((video_positions_.size() + text_positions_.size()) * sizeof(int)), static_cast<size_t>(1u));
    if (mapping_cpu_.empty())
        return -100;
    std::memcpy(mapping_cpu_.data, video_positions_.data(), video_positions_.size() * sizeof(int));
    std::memcpy(static_cast<unsigned char*>(mapping_cpu_.data) + video_positions_.size() * sizeof(int),
                text_positions_.data(), text_positions_.size() * sizeof(int));
#endif
    return window_count_ > 0 && target.size() == static_cast<size_t>(source_tokens_) ? 0 : -1;
}

int SeedVR2AWAUnpack::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                              std::vector<ncnn::Mat>& top_blobs,
                              const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1 || window_count_ <= 0)
        return -1;
    const ncnn::Mat& packed = bottom_blobs[0];
    if (packed.dims != 3 || packed.elempack != 1 || packed.elemsize != 4u || packed.w <= 0 ||
        packed.h <= 0 || packed.c != sequence_tokens_)
        return -1;
    const int heads = packed.h;
    const int head_dim = packed.w;
    top_blobs.resize(2);
#if NCNN_BATCH
    top_blobs[0].create(head_dim, heads, source_w_, source_h_, 4u, 1, source_t_, opt.blob_allocator);
#else
    top_blobs[0].create(head_dim, heads, source_tokens_, 4u, opt.blob_allocator);
#endif
    top_blobs[1].create(head_dim, heads, text_tokens_, 4u, opt.blob_allocator);
    if (top_blobs[0].empty() || top_blobs[1].empty())
        return -100;
    for (int source_index = 0; source_index < source_tokens_; source_index++)
    {
        const int packed_index = video_positions_[source_index];
        if (packed_index < 0 || packed_index >= packed.c)
            return -1;
        const float* packed_data = static_cast<const float*>(packed.channel(packed_index).data);
        const int t = source_index / (source_h_ * source_w_);
        const int spatial = source_index % (source_h_ * source_w_);
        const int h = spatial / source_w_;
        const int w = spatial % source_w_;
        float* video_data = 0;
#if NCNN_BATCH
        video_data = static_cast<float*>(top_blobs[0].batch(t).channel(h).depth(w).data);
#else
        video_data = static_cast<float*>(top_blobs[0].channel(source_index).data);
#endif
        std::memcpy(video_data, packed_data, heads * head_dim * sizeof(float));
    }
    top_blobs[1].fill(0.f);
    for (int window = 0; window < window_count_; window++)
    {
        for (int token = 0; token < text_tokens_; token++)
        {
            const float* packed_data = static_cast<const float*>(packed.channel(text_positions_[window * text_tokens_ + token]).data);
            float* text_data = static_cast<float*>(top_blobs[1].channel(token).data);
            for (int value = 0; value < heads * head_dim; value++)
                text_data[value] += packed_data[value] / window_count_;
        }
    }
    return 0;
}

#if NCNN_VULKAN
int SeedVR2AWAUnpack::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (mapping_cpu_.empty())
        return -1;
    cmd.record_upload(mapping_cpu_, mapping_gpu_, opt);
    return mapping_gpu_.empty() ? -100 : 0;
}

int SeedVR2AWAUnpack::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute || !vkdev)
        return 0;
    if (create_awa_pipeline(vkdev, awa_unpack_comp_data, sizeof(awa_unpack_comp_data), opt, 0, pipeline_video_) != 0)
        return -1;
    if (create_awa_pipeline(vkdev, awa_unpack_comp_data, sizeof(awa_unpack_comp_data), opt, 1, pipeline_text_) != 0)
    {
        delete pipeline_video_;
        pipeline_video_ = 0;
        return -1;
    }
    return 0;
}

int SeedVR2AWAUnpack::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_video_;
    delete pipeline_text_;
    pipeline_video_ = 0;
    pipeline_text_ = 0;
    mapping_gpu_ = ncnn::VkMat();
    return 0;
}

int SeedVR2AWAUnpack::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                              std::vector<ncnn::VkMat>& top_blobs,
                              ncnn::VkCompute& cmd,
                              const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1 || window_count_ <= 0 || pipeline_video_ == 0 || pipeline_text_ == 0 || mapping_gpu_.empty())
        return -1;
    const ncnn::VkMat& packed = bottom_blobs[0];
    if ((packed.elemsize != 2u && packed.elemsize != 4u) || packed.elempack != 1 || packed.dims != 3 ||
        packed.w <= 0 || packed.h <= 0 || packed.c != sequence_tokens_)
        return -1;

    const int heads = packed.h;
    if (heads <= 0)
        return -1;
    const int head_dim = packed.w;
    top_blobs.resize(2);
#if NCNN_BATCH
    top_blobs[0].create(head_dim, heads, source_w_, source_h_, packed.elemsize, 1, source_t_, opt.blob_vkallocator);
#else
    top_blobs[0].create(head_dim, heads, source_tokens_, packed.elemsize, 1, opt.blob_vkallocator);
#endif
    top_blobs[1].create(head_dim, heads, text_tokens_, packed.elemsize, 1, opt.blob_vkallocator);
    if (top_blobs[0].empty() || top_blobs[1].empty())
        return -100;

    std::vector<ncnn::vk_constant_type> constants(12);
    constants[0].i = static_cast<int>(packed.cstep);
    constants[2].i = heads;
    constants[3].i = head_dim;
    constants[4].i = source_t_;
    constants[5].i = source_h_;
    constants[6].i = source_w_;
    constants[7].i = source_tokens_;
    constants[8].i = text_tokens_;
    constants[9].i = window_count_;
    constants[11].i = source_tokens_;

    for (int t = 0; t < source_t_; t++)
    {
#if NCNN_BATCH
        ncnn::VkMat video = top_blobs[0].batch(t);
#else
        ncnn::VkMat video = top_blobs[0];
#endif
        constants[1].i = static_cast<int>(video.cstep);
        constants[10].i = t;
        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = packed;
        bindings[1] = video;
        bindings[2] = top_blobs[1];
        bindings[3] = mapping_gpu_;
        cmd.record_pipeline(pipeline_video_, bindings, constants, video);
    }

    constants[1].i = static_cast<int>(top_blobs[1].cstep);
    constants[10].i = 0;
    std::vector<ncnn::VkMat> text_bindings(4);
    text_bindings[0] = packed;
#if NCNN_BATCH
    text_bindings[1] = top_blobs[0].batch(0);
#else
    text_bindings[1] = top_blobs[0];
#endif
    text_bindings[2] = top_blobs[1];
    text_bindings[3] = mapping_gpu_;
    cmd.record_pipeline(pipeline_text_, text_bindings, constants, top_blobs[1]);
    return 0;
}
#endif

DEFINE_LAYER_CREATOR(SeedVR2AWAPack)
DEFINE_LAYER_CREATOR(SeedVR2AWAUnpack)

void register_seedvr2_awa_layers(ncnn::Net& net)
{
    net.register_custom_layer("SeedVR2AWAPack", SeedVR2AWAPack_layer_creator);
    net.register_custom_layer("SeedVR2AWAUnpack", SeedVR2AWAUnpack_layer_creator);
}
