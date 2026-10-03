#include "vae/static_vae_graph.h"

#include <fstream>
#include <iterator>

#include <algorithm>
#include <charconv>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace seedvr2
{
namespace
{

bool replace_numeric_field(std::string& line, std::string_view key, int value)
{
    const std::size_t field_start = line.find(key);
    if (field_start == std::string::npos)
        return false;

    const std::size_t value_start = field_start + key.size();
    std::size_t value_end = value_start;
    if (value_end < line.size() && (line[value_end] == '-' || line[value_end] == '+'))
        value_end++;
    while (value_end < line.size() && line[value_end] >= '0' && line[value_end] <= '9')
        value_end++;
    if (value_end == value_start || (value_end == value_start + 1 &&
                                     (line[value_start] == '-' || line[value_start] == '+')))
        return false;

    line.replace(value_start, value_end - value_start, std::to_string(value));
    return true;
}

int scaled_spatial_dimension(int value,
                             int requested_width,
                             int requested_height,
                             bool width_field,
                             bool flattened_spatial_product)
{
    // This expression flattens both spatial axes, so its first output axis
    // scales with width and height rather than either axis independently.
    if (flattened_spatial_product && width_field)
        return value * requested_width * requested_height / (256 * 256);
    if (value < 32 || value > 256)
        return value;

    const int requested = width_field ? requested_width : requested_height;
    if ((value * requested) % 256 != 0)
        return value;
    return value * requested / 256;
}

bool scale_reshape_dimensions(std::string& line, int width, int height)
{
    const std::size_t width_field = line.find(" 0=");
    const std::size_t height_field = line.find(" 1=");
    if (width_field == std::string::npos || height_field == std::string::npos)
        return true;

    auto read_field = [&line](std::size_t field_start, std::size_t key_size, int& value) {
        const std::size_t value_start = field_start + key_size;
        std::size_t value_end = value_start;
        if (value_end < line.size() && (line[value_end] == '-' || line[value_end] == '+'))
            value_end++;
        while (value_end < line.size() && line[value_end] >= '0' && line[value_end] <= '9')
            value_end++;
        if (value_end == value_start || (value_end == value_start + 1 &&
                                         (line[value_start] == '-' || line[value_start] == '+')))
            return false;
        value = std::stoi(line.substr(value_start, value_end - value_start));
        return true;
    };

    int numeric_width = 0;
    int numeric_height = 0;
    if (!read_field(width_field, 3, numeric_width) || !read_field(height_field, 3, numeric_height))
        return false;

    const bool flattened_spatial_product = line.find(" 6=\"*(") != std::string::npos;
    const int scaled_width =
        scaled_spatial_dimension(numeric_width, width, height, true, flattened_spatial_product);
    const int scaled_height =
        scaled_spatial_dimension(numeric_height, width, height, false, flattened_spatial_product);
    return replace_numeric_field(line, " 0=", scaled_width) && replace_numeric_field(line, " 1=", scaled_height);
}

bool drop_baked_reshape_shape_inputs(std::string& line)
{
    std::vector<std::string_view> fields;
    std::size_t cursor = 0;
    while (cursor < line.size())
    {
        while (cursor < line.size() && (line[cursor] == ' ' || line[cursor] == '\t'))
            cursor++;
        const std::size_t field_start = cursor;
        while (cursor < line.size() && line[cursor] != ' ' && line[cursor] != '\t')
            cursor++;
        if (field_start < cursor)
            fields.emplace_back(line.data() + field_start, cursor - field_start);
    }
    if (fields.size() < 5 || fields[0] != "Reshape")
        return false;

    int bottom_count = 0;
    int top_count = 0;
    const auto bottom_parse = std::from_chars(fields[2].data(), fields[2].data() + fields[2].size(), bottom_count);
    const auto top_parse = std::from_chars(fields[3].data(), fields[3].data() + fields[3].size(), top_count);
    if (bottom_parse.ec != std::errc() || bottom_parse.ptr != fields[2].data() + fields[2].size() ||
        top_parse.ec != std::errc() || top_parse.ptr != fields[3].data() + fields[3].size() ||
        bottom_count < 1 || top_count < 1 || fields.size() < static_cast<std::size_t>(4 + bottom_count + top_count))
        return false;
    if (bottom_count == 1)
        return true;

    const std::size_t top_fields_end = static_cast<std::size_t>(4 + bottom_count + top_count);
    const char* tail = fields[top_fields_end - 1].data() + fields[top_fields_end - 1].size();
    std::string collapsed;
    collapsed.reserve(line.size());
    collapsed.append(fields[0]);
    collapsed.push_back(' ');
    collapsed.append(fields[1]);
    collapsed.append(" 1 ");
    collapsed.append(fields[3]);
    collapsed.push_back(' ');
    collapsed.append(fields[4]);
    for (int i = 0; i < top_count; ++i)
    {
        collapsed.push_back(' ');
        collapsed.append(fields[static_cast<std::size_t>(4 + bottom_count + i)]);
    }
    collapsed.append(tail, line.data() + line.size() - tail);
    line = std::move(collapsed);
    return true;
}

void migrate_legacy_depth_to_space_lines(std::string& param)
{
    std::string migrated;
    migrated.reserve(param.size());
    std::size_t line_start = 0;
    while (line_start < param.size())
    {
        const std::size_t newline = param.find('\n', line_start);
        const bool has_newline = newline != std::string::npos;
        const std::size_t line_end = has_newline ? newline : param.size();
        std::string line = param.substr(line_start, line_end - line_start);
        std::istringstream fields(line);
        std::vector<std::string> tokens;
        std::string token;
        while (fields >> token)
            tokens.push_back(token);
        if (tokens.size() >= 9 && tokens[0] == "SeedVR2DepthToSpace")
        {
            std::string* first = nullptr;
            std::string* third = nullptr;
            for (std::size_t index = 6; index < tokens.size(); ++index)
            {
                if (tokens[index] == "0=1") first = &tokens[index];
                if (tokens[index] == "2=2") third = &tokens[index];
            }
            if (first != nullptr && third != nullptr)
            {
                *first = "0=2";
                *third = "2=1";
                std::ostringstream rewritten;
                for (std::size_t index = 0; index < tokens.size(); ++index)
                {
                    if (index != 0) rewritten << ' ';
                    rewritten << tokens[index];
                }
                line = rewritten.str();
            }
        }
        migrated.append(line);
        if (has_newline)
            migrated.push_back('\n');
        line_start = has_newline ? newline + 1 : param.size();
    }
    param = std::move(migrated);
}

bool migrate_fixed_256_pointwise_conv3d_lines(std::string& param, std::size_t& replaced)
{
    static constexpr std::string_view target_names[] = {
        "conv3d_11", "conv3d_19", "conv3d_23", "conv3d_28", "conv3d_32"};
    static constexpr std::string_view source_type = "Convolution3D";
    static constexpr std::string_view replacement_type = "SeedVR2PointwiseConv3D";
    replaced = 0;
    for (const std::string_view target_name : target_names)
    {
        const std::string needle = std::string(source_type) + " " + std::string(target_name) + " ";
        std::size_t search_start = 0;
        while (true)
        {
            const std::size_t match = param.find(needle, search_start);
            if (match == std::string::npos)
                break;
            if (match == 0 || param[match - 1] == '\n')
            {
                param.replace(match, source_type.size(), replacement_type);
                replaced++;
                break;
            }
            search_start = match + needle.size();
        }
    }
    return replaced == std::size(target_names);
}

} // namespace

VaeGraphMode select_vae_graph_mode(int width, int height, int tile_size)
{
    const bool supported_shape = (width == 128 && height == 128) ||
                                 (width == 128 && height == 256) ||
                                 (width == 256 && height == 128) || (width == 256 && height == 256);
    if (tile_size == 0 && supported_shape)
    {
        if (width == 256 && height == 256)
            return VaeGraphMode::Static256;
        return VaeGraphMode::StaticShape;
    }
    return VaeGraphMode::Dynamic;
}

bool vae_graph_uses_light_mode(VaeGraphMode mode)
{
    return mode == VaeGraphMode::Static256 || mode == VaeGraphMode::StaticShape;
}

const char* vae_graph_mode_name(VaeGraphMode mode)
{
    if (mode == VaeGraphMode::Static256)
        return "static-256";
    if (mode == VaeGraphMode::StaticShape)
        return "static-shape";
    return "dynamic";
}

bool materialize_static_vae_param(std::string_view dynamic_param,
                                  int width,
                                  int height,
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
        std::string line_copy(line);
        if (is_reshape && !scale_reshape_dimensions(line_copy, width, height))
        {
            error = "malformed numeric Reshape shape field";
            static_param.clear();
            removed_fields = 0;
            return false;
        }

        std::size_t cursor = 0;
        std::size_t copy_cursor = 0;
        bool removed_dynamic_shape = false;
        std::string materialized_line;
        while (is_reshape)
        {
            const std::size_t field_start = line_copy.find(" 6=\"", cursor);
            if (field_start == std::string_view::npos)
                break;
            const std::size_t value_end = line_copy.find('\"', field_start + 4);
            if (value_end == std::string_view::npos)
            {
                error = "malformed quoted Reshape shape field";
                static_param.clear();
                removed_fields = 0;
                return false;
            }
            const std::string_view value(line_copy.data() + field_start + 4, value_end - field_start - 4);
            const bool is_dynamic_shape = std::any_of(value.begin(), value.end(), [](char character) {
                return character == 'w' || character == 'h' || character == 'd' || character == 'c';
            });
            if (!is_dynamic_shape)
            {
                cursor = value_end + 1;
                continue;
            }
            materialized_line.append(line_copy.substr(copy_cursor, field_start - copy_cursor));
            cursor = value_end + 1;
            copy_cursor = cursor;
            removed_fields++;
            removed_dynamic_shape = true;
        }
        if (is_reshape)
        {
            materialized_line.append(line_copy.substr(copy_cursor));
            if (removed_dynamic_shape && !drop_baked_reshape_shape_inputs(materialized_line))
            {
                error = "failed to remove baked Reshape shape inputs";
                static_param.clear();
                removed_fields = 0;
                return false;
            }
            static_param.append(materialized_line);
        }
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
                       std::string& error,
                       bool enable_fixed_256_pointwise_conv3d)
{
    prepared = PreparedVaeGraph();
    error.clear();
    prepared.param_path = stem;
    prepared.param_path += ".ncnn.param";
    prepared.model_path = stem;
    prepared.model_path += ".ncnn.bin";

    const VaeGraphMode mode = select_vae_graph_mode(width, height, tile_size);
    if (mode == VaeGraphMode::Dynamic)
        return true;

    std::ifstream file(prepared.param_path, std::ios::binary);
    if (!file)
    {
        error = "failed to read VAE parameter file: " + prepared.param_path.string();
        return false;
    }
    std::string dynamic_param((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    migrate_legacy_depth_to_space_lines(dynamic_param);
    if (enable_fixed_256_pointwise_conv3d && width == 256 && height == 256)
    {
        std::size_t replaced_pointwise_nodes = 0;
        if (!migrate_fixed_256_pointwise_conv3d_lines(dynamic_param, replaced_pointwise_nodes))
        {
            error = "fixed 256x256 VAE graph must contain five pointwise Conv3D nodes; found " +
                    std::to_string(replaced_pointwise_nodes);
            return false;
        }
    }
    std::size_t removed_fields = 0;
    if (!materialize_static_vae_param(dynamic_param, width, height, prepared.param_contents, removed_fields, error))
        return false;

    // A legacy package may already contain a fixed graph. Keep its historical
    // path-based load because its graph was not validated for light mode here.
    if (removed_fields == 0)
    {
        prepared.param_contents.clear();
        return true;
    }
    prepared.mode = mode;
    return true;
}

} // namespace seedvr2
