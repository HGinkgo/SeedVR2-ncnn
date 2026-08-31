#include "inference/image_inference.h"

#include <cassert>
#include <string>
#include <vector>

int main()
{
    seedvr2::ModelGraphSet graphs;
    seedvr2::RgbImage input;
    input.width = 16;
    input.height = 16;
    input.pixels.resize(16u * 16u * 3u, 127u);

    seedvr2::ResolutionPlan plan;
    assert(seedvr2::ResolutionPlan::from_explicit(16, 16, plan));

    seedvr2::RgbImage output;
    std::string error;
    const bool ok = seedvr2::run_image_inference(graphs, input, plan, -1, output, error);
#if NCNN_VULKAN
    (void)ok;
#else
    assert(!ok);
    assert(error == "image inference requires a Vulkan-enabled build");
#endif

    seedvr2::ImageInferenceSession session;
    error.clear();
    const bool session_ok = seedvr2::ImageInferenceSession::open(graphs, plan, -1, session, error);
#if NCNN_VULKAN
    (void)session_ok;
#else
    assert(!session_ok);
    assert(error == "image inference requires a Vulkan-enabled build");
#endif

    std::vector<seedvr2::RgbImage> batch;
    std::vector<seedvr2::RgbImage> outputs(1);
    error.clear();
    assert(!session.run_batch(batch, outputs, error));
    assert(outputs.empty());
    assert(error == "inference batch must contain 1 or 2 frames");

    batch.resize(3);
    outputs.assign(1, seedvr2::RgbImage());
    error.clear();
    assert(!session.run_batch(batch, outputs, error));
    assert(outputs.empty());
    assert(error == "inference batch must contain 1 or 2 frames");
    return 0;
}
