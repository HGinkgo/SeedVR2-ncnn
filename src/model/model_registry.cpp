#include "model/model_registry.h"

#include "resolution/resolution_plan.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

namespace seedvr2
{
namespace
{

constexpr std::size_t kConditioningWidth = 5120;
constexpr std::size_t kPositiveTokens = 58;
constexpr std::size_t kNegativeTokens = 64;

bool is_regular_model_file(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

class Sha256 final
{
public:
    Sha256()
    {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    }

    void update(const std::uint8_t* data, std::size_t size)
    {
        while (size > 0)
        {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy(data, data + copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size())
            {
                transform(block_.data());
                bit_count_ += 512;
                block_size_ = 0;
            }
        }
    }

    std::string final_hex()
    {
        const std::uint64_t message_bits = bit_count_ + static_cast<std::uint64_t>(block_size_) * 8u;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56)
        {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (int index = 0; index < 8; ++index)
            block_[56 + index] = static_cast<std::uint8_t>(message_bits >> (56 - index * 8));
        transform(block_.data());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (std::uint32_t word : state_)
            output << std::setw(8) << word;
        return output.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> kConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    static std::uint32_t rotate_right(std::uint32_t value, int count)
    {
        return (value >> count) | (value << (32 - count));
    }

    void transform(const std::uint8_t* block)
    {
        std::array<std::uint32_t, 64> schedule{};
        for (int index = 0; index < 16; ++index)
            schedule[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                              (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                              (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                              static_cast<std::uint32_t>(block[index * 4 + 3]);
        for (int index = 16; index < 64; ++index)
        {
            const std::uint32_t s0 = rotate_right(schedule[index - 15], 7) ^
                                     rotate_right(schedule[index - 15], 18) ^ (schedule[index - 15] >> 3);
            const std::uint32_t s1 = rotate_right(schedule[index - 2], 17) ^
                                     rotate_right(schedule[index - 2], 19) ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int index = 0; index < 64; ++index)
        {
            const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + s1 + choose + kConstants[index] + schedule[index];
            const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t bit_count_ = 0;
};

constexpr std::array<std::uint32_t, 64> Sha256::kConstants;

bool sha256_file(const std::filesystem::path& path, std::string& digest, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "failed to open model package artifact: " + path.string();
        return false;
    }
    Sha256 hash;
    std::array<std::uint8_t, 64 * 1024> buffer{};
    while (input)
    {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0)
            hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof())
    {
        error = "failed to read model package artifact: " + path.string();
        return false;
    }
    digest = hash.final_hex();
    return true;
}

bool is_hex_hash(const std::string& hash)
{
    if (hash.size() != 64)
        return false;
    return std::all_of(hash.begin(), hash.end(), [](unsigned char value) { return std::isxdigit(value) != 0; });
}

bool safe_manifest_path(const std::string& name)
{
    if (name.empty() || name.find('\\') != std::string::npos)
        return false;
    const std::filesystem::path path(name);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory())
        return false;
    for (const auto& component : path)
        if (component == ".." || component == ".")
            return false;
    return path.generic_string() == name;
}

bool graph_exists(const std::filesystem::path& stem, std::string& error)
{
    const std::filesystem::path param = stem.string() + ".ncnn.param";
    const std::filesystem::path model = stem.string() + ".ncnn.bin";
    if (!is_regular_model_file(param))
    {
        error = "missing model graph artifact: " + param.string();
        return false;
    }
    if (!is_regular_model_file(model))
    {
        error = "missing model graph artifact: " + model.string();
        return false;
    }
    return true;
}

std::string variant_name(const ResolutionPlan& plan)
{
    return std::to_string(plan.image_height) + "x" + std::to_string(plan.image_width);
}

bool has_flat_graph_artifact(const std::filesystem::path& model_dir)
{
    const std::filesystem::path encode = model_dir / "vae_encode";
    return is_regular_model_file(encode.string() + ".ncnn.param") ||
           is_regular_model_file(encode.string() + ".ncnn.bin");
}

bool validate_manifest(const std::filesystem::path& model_dir, std::string& error)
{
    const std::filesystem::path manifest = model_dir / "manifest.sha256";
    std::ifstream input(manifest);
    if (!input)
    {
        error = "missing model package manifest: " + manifest.string();
        return false;
    }

    std::map<std::string, std::string> expected;
    std::size_t line_number = 0;
    std::string line;
    while (std::getline(input, line))
    {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const std::size_t separator = line.find("  ");
        if (separator != 64 || line.size() <= separator + 2 || !is_hex_hash(line.substr(0, separator)))
        {
            error = "invalid model package manifest record at line " + std::to_string(line_number);
            return false;
        }
        const std::string name = line.substr(separator + 2);
        if (!safe_manifest_path(name))
        {
            error = "invalid model package manifest path at line " + std::to_string(line_number);
            return false;
        }
        std::string digest = line.substr(0, separator);
        std::transform(digest.begin(), digest.end(), digest.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (!expected.emplace(name, digest).second)
        {
            error = "duplicate model package manifest path: " + name;
            return false;
        }
    }
    if (expected.size() != 75)
    {
        error = "model package manifest must contain 75 records, got " + std::to_string(expected.size());
        return false;
    }

    std::map<std::string, std::filesystem::path> actual;
    std::error_code traversal_error;
    for (std::filesystem::recursive_directory_iterator it(model_dir, traversal_error), end; it != end; it.increment(traversal_error))
    {
        if (traversal_error)
        {
            error = "failed to inspect model package: " + model_dir.string();
            return false;
        }
        const bool is_symlink = it->is_symlink(traversal_error);
        if (traversal_error)
        {
            error = "failed to inspect model package: " + model_dir.string();
            return false;
        }
        if (is_symlink)
        {
            error = "model package must not contain symbolic links: " + it->path().string();
            return false;
        }
        if (!it->is_regular_file(traversal_error))
        {
            if (traversal_error)
            {
                error = "failed to inspect model package: " + model_dir.string();
                return false;
            }
            continue;
        }
        const std::string name = std::filesystem::relative(it->path(), model_dir).generic_string();
        if (name == "manifest.sha256")
            continue;
        if (!actual.emplace(name, it->path()).second)
        {
            error = "duplicate model package artifact path: " + name;
            return false;
        }
    }
    for (const auto& [name, digest] : expected)
    {
        const auto actual_it = actual.find(name);
        if (actual_it == actual.end())
        {
            error = "manifest artifact is missing: " + name;
            return false;
        }
        std::string actual_digest;
        if (!sha256_file(actual_it->second, actual_digest, error))
            return false;
        if (actual_digest != digest)
        {
            error = "manifest hash mismatch for artifact: " + name;
            return false;
        }
    }
    for (const auto& [name, path] : actual)
    {
        if (expected.find(name) == expected.end())
        {
            error = "artifact is not listed in model package manifest: " + path.string();
            return false;
        }
    }
    return true;
}

} // namespace

bool ModelRegistry::open(const std::filesystem::path& model_dir, ModelRegistry& registry, std::string& error)
{
    error.clear();
    if (model_dir.empty())
    {
        error = "model directory is empty";
        return false;
    }

    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(model_dir, filesystem_error))
    {
        error = "model directory does not exist or is not a directory: " + model_dir.string();
        return false;
    }

    registry.model_dir_ = model_dir;
    return true;
}

bool ModelRegistry::check_package(std::string& error) const
{
    error.clear();
    if (model_dir_.empty())
    {
        error = "model registry is not open";
        return false;
    }
    if (!validate_manifest(model_dir_, error))
        return false;

    ResolutionPlan plan;
    if (!ResolutionPlan::from_explicit(256, 256, plan, &error))
        return false;
    ModelGraphSet graphs;
    if (!resolve(plan, graphs, error))
        return false;
    return true;
}

bool ModelRegistry::resolve(const ResolutionPlan& plan, ModelGraphSet& graphs, std::string& error) const
{
    graphs = ModelGraphSet();
    error.clear();
    if (model_dir_.empty())
    {
        error = "model registry is not open";
        return false;
    }
    if (plan.image_height <= 0 || plan.image_width <= 0)
    {
        error = "model graph resolution requires positive target dimensions";
        return false;
    }

    std::filesystem::path graph_dir = model_dir_;
    if (!has_flat_graph_artifact(model_dir_))
    {
        graph_dir /= variant_name(plan);
        std::error_code filesystem_error;
        if (!std::filesystem::is_directory(graph_dir, filesystem_error))
        {
            error = "model variant is missing for target " + variant_name(plan) + ": " + graph_dir.string();
            return false;
        }
    }

    const std::filesystem::path vae_encode = graph_dir / "vae_encode";
    const std::filesystem::path vae_decode = graph_dir / "vae_decode";
    const std::filesystem::path dit_stack = graph_dir;
    if (!graph_exists(vae_encode, error) || !graph_exists(vae_decode, error))
        return false;

    const std::vector<std::filesystem::path> stack_graphs = {
        dit_stack / "dit_input",
        dit_stack / "dit_embedding",
        dit_stack / "dit_output",
    };
    for (const std::filesystem::path& graph : stack_graphs)
        if (!graph_exists(graph, error))
            return false;
    for (int index = 0; index < 32; index++)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "dit_block_%02d", index);
        if (!graph_exists(dit_stack / name, error))
            return false;
    }

    const std::filesystem::path conditioning = model_dir_ / "conditioning" / "pos_emb.f32";
    if (!is_regular_model_file(conditioning))
    {
        error = "missing positive conditioning artifact: " + conditioning.string();
        return false;
    }
    std::error_code size_error;
    const std::uintmax_t bytes = std::filesystem::file_size(conditioning, size_error);
    const std::uintmax_t bytes_per_token = kConditioningWidth * sizeof(float);
    if (size_error || (bytes != kPositiveTokens * bytes_per_token && bytes != kNegativeTokens * bytes_per_token))
    {
        std::ostringstream message;
        message << "positive conditioning artifact has unsupported size: " << conditioning.string();
        error = message.str();
        return false;
    }

    graphs.vae_encode_stem = vae_encode;
    graphs.dit_stack_dir = dit_stack;
    graphs.vae_decode_stem = vae_decode;
    graphs.conditioning_path = conditioning;
    graphs.text_tokens = static_cast<int>(bytes / bytes_per_token);
    return true;
}

} // namespace seedvr2
