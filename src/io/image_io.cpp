#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cstddef>

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

} // namespace seedvr2
