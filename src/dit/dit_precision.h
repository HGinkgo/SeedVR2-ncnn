#pragma once
namespace ncnn { class Net; }
namespace seedvr2
{
// Net-local Vulkan overrides for the fixed PT2 mixed-precision graphs.
// Register before load_param; configure before load_model. Configure rejects
// graph contracts whose FP32 parameter views / BF16 boundaries are unknown.
void register_dit_block_precision_layers(ncnn::Net& net, bool use_native_bf16_rne = false);
bool configure_dit_block_precision_boundaries(ncnn::Net& net);
bool configure_dit_block_precision_layers(ncnn::Net& net, bool require_native_bf16_rne = false);
void register_dit_projection_precision_layers(ncnn::Net& net);
bool configure_dit_projection_precision_layers(ncnn::Net& net, int expected_count);
void register_dit_embedding_precision_layers(ncnn::Net& net);
bool configure_dit_embedding_precision_layers(ncnn::Net& net);
void register_dit_output_precision_layers(ncnn::Net& net, bool use_native_bf16_rne = false);
bool configure_dit_output_precision_layers(ncnn::Net& net, bool require_native_bf16_rne = false);
}
