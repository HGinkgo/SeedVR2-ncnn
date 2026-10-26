#pragma once
namespace ncnn { class Net; }
namespace seedvr2
{
// Register before load_param; configure before load_model. The checks reject
// unknown FP32 parameter-view and BF16-boundary graph contracts.
void register_dit_block_precision_layers(ncnn::Net& net);
bool configure_dit_block_precision_layers(ncnn::Net& net);
void register_dit_output_precision_layers(ncnn::Net& net);
bool configure_dit_output_precision_layers(ncnn::Net& net);
}
