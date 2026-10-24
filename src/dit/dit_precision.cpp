// SPDX-License-Identifier: BSD-3-Clause
#include "dit_precision.h"
#include "net.h"
#include "layer/rmsnorm.h"
#include "layer/memorydata.h"
#include "layer/binaryop.h"
#include "layer/concat.h"
#include "layer/unaryop.h"
#include "layer/vulkan/gemm_vulkan.h"
#include "layer/vulkan/innerproduct_vulkan.h"
#include "layer/vulkan/swish_vulkan.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#include "layer/vulkan/shader/gemm_rne.comp.hex.h"
#include "layer/vulkan/shader/innerproduct_gemm_wp4_rne.comp.hex.h"
#include "layer/vulkan/shader/innerproduct_pack4_rne.comp.hex.h"
#include "layer/vulkan/shader/swish_rne.comp.hex.h"
#include "layer/vulkan/shader/dit_rmsnorm.comp.hex.h"
#include "layer/vulkan/shader/dit_binaryop.comp.hex.h"
#include "layer/vulkan/shader/dit_qkv_concat.comp.hex.h"
#include "layer/vulkan/shader/dit_output_unary.comp.hex.h"

namespace
{
bool supports_native_bf16_rne(const ncnn::Option& opt)
{
    return opt.use_bf16_storage && opt.use_bf16_packed && !opt.use_fp16_storage &&
           !opt.use_fp16_packed && !opt.use_fp16_arithmetic && !opt.use_shader_local_memory &&
           !opt.use_subgroup_ops && !opt.use_cooperative_matrix;
}

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

class DitRneGemm final : public ncnn::Gemm_vulkan
{
public:
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!supports_native_bf16_rne(opt) || ncnn::Gemm_vulkan::create_pipeline(opt) != 0 ||
            use_subgroup_ops || use_cooperative_matrix || pipeline_gemm_pack4 || !pipeline_gemm)
            return -1;

        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(seedvr2_gemm_rne_comp_data,
                                       sizeof(seedvr2_gemm_rne_comp_data), opt, spirv) != 0)
            return -1;

        std::vector<ncnn::vk_specialization_type> specs(15);
        specs[0].f = alpha;
        specs[1].f = beta;
        specs[2].i = transA;
        specs[3].i = transB;
        specs[4].i = constantA;
        specs[5].i = constantB;
        specs[6].i = constantC;
        specs[7].i = constantM;
        specs[8].i = constantN;
        specs[9].i = constantK;
        specs[10].i = constant_broadcast_type_C;
        specs[11].i = output_N1M;
        specs[12].i = output_elempack;
        specs[13].i = output_elemtype;
        specs[14].i = output_transpose;

        delete pipeline_gemm;
        pipeline_gemm = new ncnn::Pipeline(vkdev);
        pipeline_gemm->set_optimal_local_size_xyz(ncnn::Mat());
        return pipeline_gemm->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
    }
};

class DitRneInnerProduct final : public ncnn::InnerProduct_vulkan
{
public:
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!supports_native_bf16_rne(opt))
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: requires native BF16 storage/packing\n", name.c_str());
            return -1;
        }
        const int base_status = ncnn::InnerProduct_vulkan::create_pipeline(opt);
        if (base_status != 0 || int8_scale_term)
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: base pipeline status=%d int8=%d\n",
                         name.c_str(), base_status, int8_scale_term);
            return -1;
        }

        const ncnn::Mat& shape = bottom_shapes.empty() ? ncnn::Mat() : bottom_shapes[0];
        const ncnn::Mat& out_shape = top_shapes.empty() ? ncnn::Mat() : top_shapes[0];
        const int num_input = weight_data_size / num_output;
        const int input_pack = num_input % 4 == 0 ? 4 : 1;
        const int output_pack = num_output % 4 == 0 ? 4 : 1;
        if (input_pack != 4 || output_pack != 4)
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: unsupported packs input=%d output=%d\n",
                         name.c_str(), input_pack, output_pack);
            return -1;
        }

        const ncnn::Mat shape_unpacked(shape.w, shape.h * shape.elempack, (void*)0);
        const ncnn::Mat out_shape_unpacked(out_shape.w, out_shape.h * out_shape.elempack, (void*)0);
        std::vector<ncnn::vk_specialization_type> specs(14);
        specs[0].i = bias_term;
        specs[1].i = activation_type;
        specs[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specs[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specs[4].i = shape_unpacked.dims;
        specs[5].i = shape_unpacked.w;
        specs[6].i = shape_unpacked.h;
        specs[7].i = shape_unpacked.c;
        specs[8].i = shape_unpacked.cstep;
        specs[9].i = out_shape_unpacked.dims;
        specs[10].i = out_shape_unpacked.w;
        specs[11].i = out_shape_unpacked.h;
        specs[12].i = out_shape_unpacked.c;
        specs[13].i = out_shape_unpacked.cstep;
        if (shape.dims == 0)
            for (size_t i = 4; i < specs.size(); i++) specs[i].i = 0;

        const bool gemm_path = pipeline_innerproduct_gemm != nullptr;
        if (gemm_path && shape.dims != 0 && (shape.dims != 2 || shape.w != num_input))
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: GEMM shape=%d:%dx%d input=%d\n",
                         name.c_str(), shape.dims, shape.w, shape.h, num_input);
            return -1;
        }
        if (!gemm_path && !pipeline_innerproduct)
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: no supported pack4 pipeline shape=%d:%dx%d\n",
                         name.c_str(), shape.dims, shape.w, shape.h);
            return -1;
        }

        ncnn::Mat local_size_xyz;
        const char* shader_data = nullptr;
        size_t shader_size = 0;
        if (gemm_path)
        {
            local_size_xyz = ncnn::Mat(std::min(16, num_output / output_pack), 4, 1, (void*)0);
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(16, out_shape_unpacked.w);
                local_size_xyz.h = std::min(4, out_shape_unpacked.h);
                local_size_xyz.c = 1;
            }
            shader_data = seedvr2_innerproduct_gemm_wp4_rne_comp_data;
            shader_size = sizeof(seedvr2_innerproduct_gemm_wp4_rne_comp_data);
        }
        else
        {
            local_size_xyz = ncnn::Mat(std::min(64, num_output / output_pack), 1, 1, (void*)0);
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(64, out_shape_unpacked.w);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }
            shader_data = seedvr2_innerproduct_pack4_rne_comp_data;
            shader_size = sizeof(seedvr2_innerproduct_pack4_rne_comp_data);
        }
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(shader_data, shader_size, opt, spirv) != 0)
        {
            std::fprintf(stderr, "DitRneInnerProduct %s: shader compilation failed\n", name.c_str());
            return -1;
        }
        ncnn::Pipeline*& pipeline = gemm_path ? pipeline_innerproduct_gemm : pipeline_innerproduct;
        delete pipeline;
        pipeline = new ncnn::Pipeline(vkdev);
        pipeline->set_optimal_local_size_xyz(local_size_xyz);
        const int status = pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
        if (status != 0)
            std::fprintf(stderr, "DitRneInnerProduct %s: pipeline creation failed: %d\n", name.c_str(), status);
        return status;
    }
};

class DitRneSwish final : public ncnn::Swish_vulkan
{
public:
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!supports_native_bf16_rne(opt))
            return -1;
        const ncnn::Mat& shape = top_shapes.empty() ? ncnn::Mat() : top_shapes[0];
        std::vector<ncnn::vk_specialization_type> specs(1);
        specs[0].u32 = shape.total() * shape.elempack / 4;
        std::vector<uint32_t> spirv;
        if (ncnn::compile_spirv_module(seedvr2_swish_rne_comp_data,
                                       sizeof(seedvr2_swish_rne_comp_data), opt, spirv) != 0)
            return -1;
        pipeline_swish = new ncnn::Pipeline(vkdev);
        pipeline_swish->set_optimal_local_size_xyz(std::max(16, (int)vkdev->info.subgroup_size()), 1, 1);
        return pipeline_swish->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs);
    }
};

ncnn::Layer* create_rne_gemm(void*) { return new DitRneGemm; }
ncnn::Layer* create_rne_innerproduct(void*) { return new DitRneInnerProduct; }
ncnn::Layer* create_rne_swish(void*) { return new DitRneSwish; }

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
void seedvr2::register_dit_block_precision_layers(ncnn::Net& net, bool use_native_bf16_rne)
{
    net.register_custom_layer("RMSNorm", create_rmsnorm);
    net.register_custom_layer("MemoryData", create_memorydata);
    net.register_custom_layer("BinaryOp", create_binaryop);
    net.register_custom_layer("Concat", create_qkv_concat);
    if (use_native_bf16_rne)
    {
        register_dit_projection_precision_layers(net);
        net.register_custom_layer("Swish", create_rne_swish);
    }
}

namespace
{
bool configure_dit_block_precision_impl(ncnn::Net& net, int& sums, int& gemms, int& swishes,
                                        bool require_native_bf16_rne)
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
            if (require_native_bf16_rne && !dynamic_cast<DitRneGemm*>(layer)) return false;
            gemms++;
        }
        else if (layer->type == "Swish")
        {
            if (require_native_bf16_rne && !dynamic_cast<DitRneSwish*>(layer)) return false;
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

bool seedvr2::configure_dit_block_precision_boundaries(ncnn::Net& net)
{
    int sums = 0, gemms = 0, swishes = 0;
    return configure_dit_block_precision_impl(net, sums, gemms, swishes, false);
}

bool seedvr2::configure_dit_block_precision_layers(ncnn::Net& net, bool require_native_bf16_rne)
{
    int sums = 0, gemms = 0, swishes = 0;
    if (!configure_dit_block_precision_impl(net, sums, gemms, swishes, require_native_bf16_rne)) return false;
    return (sums == 12 && gemms == 10 && swishes == 2) ||
           (sums == 6 && gemms == 7 && swishes == 1); // paired blocks, then video-only last block
}

void seedvr2::register_dit_projection_precision_layers(ncnn::Net& net)
{
    net.register_custom_layer("Gemm", create_rne_gemm);
}

bool seedvr2::configure_dit_projection_precision_layers(ncnn::Net& net, int expected_count)
{
    int gemms = 0;
    for (const auto* layer : net.mutable_layers())
    {
        if (layer->type == "Gemm")
        {
            if (!dynamic_cast<const DitRneGemm*>(layer)) return false;
            gemms++;
        }
    }
    return expected_count > 0 && gemms == expected_count;
}

void seedvr2::register_dit_embedding_precision_layers(ncnn::Net& net)
{
    net.register_custom_layer("InnerProduct", create_rne_innerproduct);
    net.register_custom_layer("Swish", create_rne_swish);
}

bool seedvr2::configure_dit_embedding_precision_layers(ncnn::Net& net)
{
    const std::set<std::pair<int, int>> expected{{256, 2560}, {2560, 2560}, {2560, 15360}};
    std::set<std::pair<int, int>> observed;
    int swishes = 0;
    for (const auto* layer : net.mutable_layers())
    {
        if (layer->type == "InnerProduct")
        {
            const auto* projection = dynamic_cast<const DitRneInnerProduct*>(layer);
            if (!projection || projection->num_output <= 0 || projection->weight_data_size % projection->num_output != 0)
                return false;
            observed.emplace(projection->weight_data_size / projection->num_output, projection->num_output);
        }
        else if (layer->type == "Swish")
        {
            if (!dynamic_cast<const DitRneSwish*>(layer)) return false;
            swishes++;
        }
    }
    return observed == expected && swishes == 2;
}

void seedvr2::register_dit_output_precision_layers(ncnn::Net& net, bool use_native_bf16_rne)
{
    net.register_custom_layer("MemoryData", create_memorydata);
    net.register_custom_layer("BinaryOp", create_binaryop);
    net.register_custom_layer("UnaryOp", create_output_unary);
    if (use_native_bf16_rne)
        register_dit_projection_precision_layers(net);
}

bool seedvr2::configure_dit_output_precision_layers(ncnn::Net& net, bool require_native_bf16_rne)
{
    std::set<int> fp32_blobs;
    int norms = 0, projections = 0, gemms = 0;
    for (auto* layer : net.mutable_layers())
    {
        if (layer->type == "Gemm")
        {
            if (require_native_bf16_rne && !dynamic_cast<DitRneGemm*>(layer)) return false;
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
