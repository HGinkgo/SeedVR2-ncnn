#include "vae/static_vae_graph.h"

#include <fstream>
#include <iterator>

#include <algorithm>

namespace seedvr2
{

VaeGraphMode select_vae_graph_mode(int width, int height, int tile_size)
{
    if (width == 256 && height == 256 && tile_size == 0)
        return VaeGraphMode::Static256;
    return VaeGraphMode::Dynamic;
}

bool vae_graph_uses_light_mode(VaeGraphMode mode)
{
    return mode == VaeGraphMode::Static256;
}

const char* vae_graph_mode_name(VaeGraphMode mode)
{
    return mode == VaeGraphMode::Static256 ? "static-256" : "dynamic";
}

bool materialize_static_vae_param(std::string_view dynamic_param,
                                  std::string& static_param,
                                  std::size_t& removed_fields,
                                  std::string& error)
{
    static_param.clear();
    removed_fields = 0;
    error.clear();

    std::size_t line_start = 0;
    while (line_start < dynamic_param.size())
    {
        const std::size_t newline = dynamic_param.find('\n', line_start);
        const bool has_newline = newline != std::string_view::npos;
        const std::size_t line_end = has_newline ? newline : dynamic_param.size();
        const std::string_view line = dynamic_param.substr(line_start, line_end - line_start);
        const bool is_reshape = line.size() >= 8 && line.compare(0, 8, "Reshape ") == 0;

        std::size_t cursor = 0;
        std::size_t copy_cursor = 0;
        while (is_reshape)
        {
            const std::size_t field_start = line.find(" 6=\"", cursor);
            if (field_start == std::string_view::npos)
                break;
            const std::size_t value_end = line.find('\"', field_start + 4);
            if (value_end == std::string_view::npos)
            {
                error = "malformed quoted Reshape shape field";
                static_param.clear();
                removed_fields = 0;
                return false;
            }
            const std::string_view value = line.substr(field_start + 4, value_end - field_start - 4);
            const bool is_dynamic_shape = std::any_of(value.begin(), value.end(), [](char character) {
                return character == 'w' || character == 'h' || character == 'd' || character == 'c';
            });
            if (!is_dynamic_shape)
            {
                cursor = value_end + 1;
                continue;
            }
            static_param.append(line.substr(copy_cursor, field_start - copy_cursor));
            cursor = value_end + 1;
            copy_cursor = cursor;
            removed_fields++;
        }
        if (is_reshape)
            static_param.append(line.substr(copy_cursor));
        else
            static_param.append(line);
        if (has_newline)
            static_param.push_back('\n');
        line_start = has_newline ? newline + 1 : dynamic_param.size();
    }
    return true;
}

bool prepare_vae_graph(const std::filesystem::path& stem,
                       int width,
                       int height,
                       int tile_size,
                       PreparedVaeGraph& prepared,
                       std::string& error)
{
    prepared = PreparedVaeGraph();
    error.clear();
    prepared.param_path = stem;
    prepared.param_path += ".ncnn.param";
    prepared.model_path = stem;
    prepared.model_path += ".ncnn.bin";

    if (select_vae_graph_mode(width, height, tile_size) != VaeGraphMode::Static256)
        return true;

    std::ifstream file(prepared.param_path, std::ios::binary);
    if (!file)
    {
        error = "failed to read VAE parameter file: " + prepared.param_path.string();
        return false;
    }
    const std::string dynamic_param((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::size_t removed_fields = 0;
    if (!materialize_static_vae_param(dynamic_param, prepared.param_contents, removed_fields, error))
        return false;

    // A legacy package may already contain a fixed graph. Keep its historical
    // path-based load because its graph was not validated for light mode here.
    if (removed_fields == 0)
    {
        prepared.param_contents.clear();
        return true;
    }
    prepared.mode = VaeGraphMode::Static256;
    return true;
}

} // namespace seedvr2
