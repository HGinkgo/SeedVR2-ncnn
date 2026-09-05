#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <unordered_set>

namespace seedvr2
{
namespace
{

std::string lowercase_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

bool valid_image(const RgbImage& image)
{
    if (image.width <= 0 || image.height <= 0)
        return false;
    const std::size_t expected = static_cast<std::size_t>(image.width) *
                                 static_cast<std::size_t>(image.height) * 3u;
    return image.pixels.size() == expected;
}

std::string output_path_key(const std::filesystem::path& path)
{
    std::string key = path.generic_string();
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return key;
}

} // namespace

bool load_rgb_image(const std::filesystem::path& path, RgbImage& image, std::string& error)
{
    image = RgbImage();
    error.clear();
    if (path.empty())
    {
        error = "input image path is empty";
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (!decoded)
    {
        const char* reason = stbi_failure_reason();
        error = "failed to decode image: " + path.string();
        if (reason)
            error += " (" + std::string(reason) + ")";
        return false;
    }

    image.width = width;
    image.height = height;
    image.pixels.assign(decoded, decoded + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);
    stbi_image_free(decoded);
    return true;
}

bool save_rgb_image(const std::filesystem::path& path, const RgbImage& image, std::string& error)
{
    error.clear();
    if (path.empty())
    {
        error = "output image path is empty";
        return false;
    }
    if (!valid_image(image))
    {
        error = "RGB image dimensions and pixel buffer do not match";
        return false;
    }

    const std::string extension = lowercase_extension(path);
    const unsigned char* pixels = image.pixels.data();
    int result = 0;
    if (extension == ".png")
        result = stbi_write_png(path.string().c_str(), image.width, image.height, 3, pixels, image.width * 3);
    else if (extension == ".jpg" || extension == ".jpeg")
        result = stbi_write_jpg(path.string().c_str(), image.width, image.height, 3, pixels, 95);
    else
    {
        error = "unsupported output image format: " + extension;
        return false;
    }

    if (!result)
    {
        error = "failed to encode image: " + path.string();
        return false;
    }
    return true;
}

bool list_rgb_images(const std::filesystem::path& directory,
                     std::vector<std::filesystem::path>& paths,
                     std::string& error)
{
    paths.clear();
    error.clear();
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(directory, filesystem_error))
    {
        error = "input directory does not exist or is not a directory: " + directory.string();
        return false;
    }
    for (std::filesystem::directory_iterator it(directory, filesystem_error), end; it != end;
         it.increment(filesystem_error))
    {
        if (filesystem_error)
        {
            error = "failed to enumerate input directory: " + directory.string();
            paths.clear();
            return false;
        }
        if (!it->is_regular_file(filesystem_error))
            continue;
        const std::string extension = lowercase_extension(it->path());
        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            paths.push_back(it->path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
    {
        error = "input directory contains no PNG or JPEG images: " + directory.string();
        return false;
    }
    return true;
}

bool make_directory_output_paths(const std::filesystem::path& output_directory,
                                 const std::vector<std::filesystem::path>& input_paths,
                                 std::vector<std::filesystem::path>& output_paths,
                                 std::string& error)
{
    output_paths.clear();
    error.clear();
    std::unordered_set<std::string> seen;
    for (const std::filesystem::path& input_path : input_paths)
    {
        const std::string stem = input_path.stem().string();
        if (stem.empty())
        {
            error = "input image has an empty output stem: " + input_path.string();
            output_paths.clear();
            return false;
        }
        const std::filesystem::path output_path = output_directory / (stem + ".png");
        if (!seen.insert(output_path_key(output_path)).second)
        {
            error = "directory output path collision: " + output_path.string();
            output_paths.clear();
            return false;
        }
        output_paths.push_back(output_path);
    }
    return true;
}

} // namespace seedvr2
