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

// Remove only quoted shape-expression fields from Reshape layers. Numeric
// shape parameters and all fields on other layer types are preserved.
bool materialize_static_vae_param(std::string_view dynamic_param,
                                  std::string& static_param,
                                  std::size_t& removed_fields,
                                  std::string& error);

// Prepare one VAE graph for the requested route. A plain 256x256 route is
// materialized in memory when the package contains dynamic Reshape metadata;
// other routes keep the original path-based load.
bool prepare_vae_graph(const std::filesystem::path& stem,
                       int width,
                       int height,
                       int tile_size,
                       PreparedVaeGraph& prepared,
                       std::string& error);

} // namespace seedvr2
