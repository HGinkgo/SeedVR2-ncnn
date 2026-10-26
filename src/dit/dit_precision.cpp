// SPDX-License-Identifier: BSD-3-Clause
#include "dit_precision.h"
#include "net.h"
#include "layer/rmsnorm.h"
#include "layer/memorydata.h"
#include "layer/binaryop.h"
#include "layer/concat.h"
#include "layer/unaryop.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#include "layer/vulkan/shader/dit_rmsnorm.comp.hex.h"
#include "layer/vulkan/shader/dit_binaryop.comp.hex.h"
#include "layer/vulkan/shader/dit_qkv_concat.comp.hex.h"
#include "layer/vulkan/shader/dit_output_unary.comp.hex.h"

namespace
{
class DitRMSNorm final : public ncnn::RMSNorm
{
public:
    DitRMSNorm()
    {
        support_vulkan = true;
        support_inplace = false;
        support_vulkan_packing = false;
    }
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!opt.use_vulkan_compute) return 0;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(dit_rmsnorm_comp_data, sizeof(dit_rmsnorm_comp_data), opt, spirv) != 0)
            return -1;
        pipeline_ = new ncnn::Pipeline(vkdev);
        pipeline_->set_optimal_local_size_xyz(128, 1, 1);
        std::vector<ncnn::vk_specialization_type> specs(2);
        specs[0].i = affine;
        specs[1].f = eps;
        return pipeline_->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
    }
    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_;
        pipeline_ = nullptr;
        gamma_gpu_.release();
        return 0;
    }
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        ncnn::Option fp32 = opt;
        fp32.use_bf16_packed = fp32.use_bf16_storage = false;
        fp32.use_fp16_packed = fp32.use_fp16_storage = false;
        ncnn::Mat gamma = affine ? gamma_data : ncnn::Mat(1);
        if (!affine) gamma.fill(1.f);
        cmd.record_upload(gamma, gamma_gpu_, fp32);
        return gamma_gpu_.empty() ? -100 : 0;
    }
    int forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        // Fixed DiT graphs normalize the last dimension (matrix width).
        const size_t scalar_size = opt.use_bf16_storage || opt.use_bf16_packed ? 2u : 4u;
        if (!pipeline_ || input.elempack != 1 || input.w != affine_size || input.n != 1 ||
            (input.dims != 2 && input.dims != 3) || input.elemsize != scalar_size)
            return -1;
        ncnn::Mat shape = input.shape();
        shape.elemsize = affine ? sizeof(float) : input.elemsize;
        output.create_like(shape, opt.blob_vkallocator);
        if (output.empty()) return -100;
        std::vector<ncnn::VkMat> bindings{input, output, gamma_gpu_, output};
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = input.w;
        constants[1].i = input.h * input.d;
        constants[2].i = input.c;
        constants[3].i = input.cstep;
        constants[4].i = output.cstep;
        ncnn::VkMat dispatcher;
        dispatcher.w = 128;
        dispatcher.h = input.h * input.d;
        dispatcher.c = input.c;
        cmd.record_pipeline(pipeline_, bindings, constants, dispatcher);
        return 0;
    }
private:
    ncnn::Pipeline* pipeline_ = nullptr;
    ncnn::VkMat gamma_gpu_;
};
ncnn::Layer* create_rmsnorm(void*) { return new DitRMSNorm; }

// PyTorch promotes Q/K to FP32 through the FP32 RMSNorm weights. V is
// still BF16; stack promotes it without rounding Q/K before RoPE.
class DitQKVConcat final : public ncnn::Concat
{
public:
    DitQKVConcat() { support_vulkan = true; }
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (axis != 1) return -1;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(dit_qkv_concat_comp_data, sizeof(dit_qkv_concat_comp_data), opt, spirv) != 0)
            return -1;
        pipeline_ = new ncnn::Pipeline(vkdev);
        pipeline_->set_optimal_local_size_xyz(64, 1, 1);
        return pipeline_->create(spirv.data(), spirv.size() * sizeof(uint32_t), {});
    }
    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_;
        pipeline_ = nullptr;
        return 0;
    }
    int forward(const std::vector<ncnn::VkMat>& input, std::vector<ncnn::VkMat>& output,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        if (input.size() != 3 || !pipeline_) return -1;
        const auto& q = input[0];
        for (const auto& tensor : input)
            if (tensor.dims != 3 || tensor.n != 1 || tensor.elempack != 1 ||
                tensor.w != q.w || tensor.h != q.h || tensor.c != q.c) return -1;
        if (q.elemsize != 4u || input[1].elemsize != 4u ||
            (input[2].elemsize != 2u && input[2].elemsize != 4u)) return -1;
        output[0].create(q.w, q.h * 3, q.c, 4u, 1, opt.blob_vkallocator);
        if (output[0].empty()) return -100;
        std::vector<ncnn::VkMat> bindings{q, input[1], input[2], input[2], output[0]};
        std::vector<ncnn::vk_constant_type> constants(8);
        constants[0].i = q.w;
        constants[1].i = q.h;
        constants[2].i = q.c;
        constants[3].i = q.cstep;
        constants[4].i = input[1].cstep;
        constants[5].i = input[2].cstep;
        constants[6].i = output[0].cstep;
        constants[7].i = input[2].elemsize == 4u;
        cmd.record_pipeline(pipeline_, bindings, constants, output[0]);
        return 0;
    }
private:
    ncnn::Pipeline* pipeline_ = nullptr;
};

class DitMemoryData final : public ncnn::MemoryData
{
public:
    DitMemoryData() { support_vulkan = true; }
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        ncnn::Option fp32 = opt;
        fp32.use_bf16_packed = fp32.use_bf16_storage = false;
        fp32.use_fp16_packed = fp32.use_fp16_storage = false;
        cmd.record_upload(data, data_gpu_, fp32, false);
        return data_gpu_.empty() ? -100 : 0;
    }
    int forward(const std::vector<ncnn::VkMat>&, std::vector<ncnn::VkMat>& output,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        cmd.record_clone(data_gpu_, output[0], opt);
        return output[0].empty() ? -100 : 0;
    }
private:
    ncnn::VkMat data_gpu_;
};

class DitBinaryOp final : public ncnn::BinaryOp
{
public:
    DitBinaryOp() { support_vulkan = true; support_inplace = false; }
    bool fp32_output = false;
    int load_param(const ncnn::ParamDict& pd) override
    {
        const int status = ncnn::BinaryOp::load_param(pd);
        one_blob_only = false;
        support_inplace = false;
        return status;
    }
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (op_type != Operation_ADD && op_type != Operation_MUL) return -1;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(dit_binaryop_comp_data, sizeof(dit_binaryop_comp_data), opt, spirv) != 0)
            return -1;
        pipeline_ = new ncnn::Pipeline(vkdev);
        pipeline_->set_optimal_local_size_xyz(64, 1, 1);
        std::vector<ncnn::vk_specialization_type> specs(1);
        specs[0].i = op_type;
        return pipeline_->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
    }
    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_;
        pipeline_ = nullptr;
        return 0;
    }
    int forward(const std::vector<ncnn::VkMat>& input, std::vector<ncnn::VkMat>& output,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        if (input.size() != (with_scalar ? 1u : 2u) || !pipeline_) return -1;
        const auto& a = input[0];
        const auto& rhs = with_scalar ? a : input[1];
        if (a.dims < 1 || a.dims > 2 || rhs.dims < 1 || rhs.dims > 2 || a.n != 1 || rhs.n != 1 ||
            a.elempack != 1 || rhs.elempack != 1 ||
            (a.elemsize != 2u && a.elemsize != 4u) || (rhs.elemsize != 2u && rhs.elemsize != 4u) ||
            (a.w != rhs.w && a.w != 1 && rhs.w != 1) ||
            (a.h != rhs.h && a.h != 1 && rhs.h != 1)) return -1;
        const int width = std::max(a.w, rhs.w), rows = std::max(a.h, rhs.h);
        const size_t scalar_size = fp32_output || !(opt.use_bf16_packed || opt.use_bf16_storage) ? 4u : 2u;
        if (a.dims == 1 && rhs.dims == 1)
            output[0].create(width, scalar_size, 1, opt.blob_vkallocator);
        else
            output[0].create(width, rows, scalar_size, 1, opt.blob_vkallocator);
        if (output[0].empty()) return -100;
        std::vector<ncnn::VkMat> bindings{a, a, rhs, rhs, output[0], output[0]};
        std::vector<ncnn::vk_constant_type> constants(11);
        constants[0].i = width;
        constants[1].i = rows;
        constants[2].i = a.w;
        constants[3].i = a.h;
        constants[4].i = rhs.w;
        constants[5].i = rhs.h;
        constants[6].i = a.elemsize == 4u;
        constants[7].i = rhs.elemsize == 4u;
        constants[8].i = output[0].elemsize == 4u;
        constants[9].i = with_scalar;
        constants[10].f = b;
        cmd.record_pipeline(pipeline_, bindings, constants, output[0]);
        return 0;
    }
private:
    ncnn::Pipeline* pipeline_ = nullptr;
};

class DitOutputUnary final : public ncnn::UnaryOp
{
public:
    DitOutputUnary() { support_vulkan = true; support_inplace = false; }
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (op_type != Operation_SQUARE && op_type != Operation_RSQRT) return -1;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(dit_output_unary_comp_data, sizeof(dit_output_unary_comp_data), opt, spirv) != 0)
            return -1;
        pipeline_ = new ncnn::Pipeline(vkdev);
        pipeline_->set_optimal_local_size_xyz(64, 1, 1);
        std::vector<ncnn::vk_specialization_type> specs(1);
        specs[0].i = op_type;
        return pipeline_->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
    }
    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_;
        pipeline_ = nullptr;
        return 0;
    }
    int forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        if (!pipeline_ || input.dims != 2 || input.n != 1 || input.elempack != 1 ||
            (input.elemsize != 2u && input.elemsize != 4u))
        {
            std::fprintf(stderr, "DitOutputUnary: unsupported layout %d:%dx%dx%dx%d n=%d pack=%d\n",
                         input.dims, input.w, input.h, input.d, input.c, input.n, input.elempack);
            return -1;
        }
        output.create(input.w, input.h, 4u, 1, opt.blob_vkallocator);
        if (output.empty()) return -100;
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].i = input.w;
        constants[1].i = input.h;
        constants[2].i = input.elemsize == 4u;
        cmd.record_pipeline(pipeline_, {input, input, output}, constants, output);
        return 0;
    }
private:
    ncnn::Pipeline* pipeline_ = nullptr;
};

ncnn::Layer* create_memorydata(void*) { return new DitMemoryData; }
ncnn::Layer* create_binaryop(void*) { return new DitBinaryOp; }
ncnn::Layer* create_qkv_concat(void*) { return new DitQKVConcat; }
ncnn::Layer* create_output_unary(void*) { return new DitOutputUnary; }

}
void seedvr2::register_dit_block_precision_layers(ncnn::Net& net)
{
    net.register_custom_layer("RMSNorm", create_rmsnorm);
    net.register_custom_layer("MemoryData", create_memorydata);
    net.register_custom_layer("BinaryOp", create_binaryop);
    net.register_custom_layer("Concat", create_qkv_concat);
}

namespace
{
bool configure_dit_block_precision_impl(ncnn::Net& net, int& sums, int& gemms, int& swishes)
{
    // Follow parameter views in the loaded graph, independent of blob numbers.
    // Ada parameter + embedding stays FP32; in-place hidden mul/add and
    // residual/SwiGLU operations round to BF16 at each PyTorch boundary.
    std::set<int> parameters;
    sums = 0;
    gemms = 0;
    swishes = 0;
    for (auto* layer : net.mutable_layers())
    {
        if (layer->type == "Gemm")
        {
            gemms++;
        }
        else if (layer->type == "Swish")
        {
            swishes++;
        }
        if (layer->type == "SeedVR2AWAPack")
            layer->featmask |= 1 << 2; // promoted FP32 QKV
        if (layer->type == "MemoryData" && layer->name.find("constant_p_ada_") == 0)
            parameters.insert(layer->tops.begin(), layer->tops.end());
        else if ((layer->type == "Reshape" || layer->type == "Split") &&
                 !layer->bottoms.empty() && parameters.count(layer->bottoms[0]))
        {
            // These views carry FP32 parameters in a BF16 graph. Their
            // packing/reshape shaders must preserve that scalar type.
            layer->featmask |= 1 << 2;
            parameters.insert(layer->tops.begin(), layer->tops.end());
        }
        else if (auto* binary = dynamic_cast<DitBinaryOp*>(layer))
        {
            for (int bottom : layer->bottoms)
                if (parameters.count(bottom)) binary->fp32_output = true;
            if (binary->fp32_output) sums++;
        }
    }
    return sums == 6 || sums == 12;
}
} // namespace

bool seedvr2::configure_dit_block_precision_layers(ncnn::Net& net)
{
    int sums = 0, gemms = 0, swishes = 0;
    if (!configure_dit_block_precision_impl(net, sums, gemms, swishes)) return false;
    return (sums == 12 && gemms == 10 && swishes == 2) ||
           (sums == 6 && gemms == 7 && swishes == 1); // paired blocks, then video-only last block
}

void seedvr2::register_dit_output_precision_layers(ncnn::Net& net)
{
    net.register_custom_layer("MemoryData", create_memorydata);
    net.register_custom_layer("BinaryOp", create_binaryop);
    net.register_custom_layer("UnaryOp", create_output_unary);
}

bool seedvr2::configure_dit_output_precision_layers(ncnn::Net& net)
{
    std::set<int> fp32_blobs;
    int norms = 0, projections = 0, gemms = 0;
    for (auto* layer : net.mutable_layers())
    {
        if (layer->type == "Gemm")
        {
            gemms++;
        }
        if (layer->type == "MemoryData" || dynamic_cast<DitOutputUnary*>(layer))
            fp32_blobs.insert(layer->tops.begin(), layer->tops.end());
        else if (auto* binary = dynamic_cast<DitBinaryOp*>(layer))
        {
            binary->fp32_output = true;
            // Autocast converts the final Ada result immediately before Gemm.
            for (auto* consumer : net.mutable_layers())
                if (consumer->type == "Gemm" && consumer->bottoms == layer->tops)
                {
                    binary->fp32_output = false;
                    projections++;
                }
            if (binary->fp32_output) fp32_blobs.insert(layer->tops.begin(), layer->tops.end());
        }
        else if (!layer->bottoms.empty() && fp32_blobs.count(layer->bottoms[0]) &&
                 (layer->type == "Reduction" || layer->type == "Reshape" ||
                  layer->type == "Squeeze" || layer->type == "Split"))
        {
            layer->featmask |= 1 << 2;
            fp32_blobs.insert(layer->tops.begin(), layer->tops.end());
            if (layer->type == "Reduction") norms++;
        }
    }
    return norms == 1 && projections == 1 && gemms == 1;
}
