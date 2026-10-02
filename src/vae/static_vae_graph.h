#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

namespace seedvr2
{

enum class VaeGraphMode
{
    Dynamic,
    Static256,
    StaticShape,
};

struct PreparedVaeGraph final
{
    VaeGraphMode mode = VaeGraphMode::Dynamic;
    std::filesystem::path param_path;
    std::filesystem::path model_path;
    // Non-empty only for a materialized fixed graph. The ncnn Net consumes it
    // through load_param_mem(), while the binary remains in the model package.
    std::string param_contents;
};

VaeGraphMode select_vae_graph_mode(int width, int height, int tile_size);
bool vae_graph_uses_light_mode(VaeGraphMode mode);
const char* vae_graph_mode_name(VaeGraphMode mode);

// Remove quoted shape-expression fields from Reshape layers and scale the
// exported 256x256 spatial dimensions to the requested fixed shape.
bool materialize_static_vae_param(std::string_view dynamic_param,
                                  int width,
                                  int height,
                                  std::string& static_param,
                                  std::size_t& removed_fields,
                                  std::string& error);

// Preserve the historical 256x256 helper contract for focused callers.
bool materialize_static_vae_param(std::string_view dynamic_param,
                                  std::string& static_param,
                                  std::size_t& removed_fields,
                                  std::string& error);

// Prepare one VAE graph for the requested route. Supported fixed shapes are
// materialized when the package contains dynamic Reshape metadata; other
// routes keep path-based loading.
bool prepare_vae_graph(const std::filesystem::path& stem,
                       int width,
                       int height,
                       int tile_size,
                       PreparedVaeGraph& prepared,
                       std::string& error,
                       bool enable_fixed_256_pointwise_conv3d = false);

} // namespace seedvr2
