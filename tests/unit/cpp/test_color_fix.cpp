#include "postprocess/color_fix.h"
#include "inference/rgb_spool.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

} // namespace

int main()
{
    seedvr2::RgbImage content;
    content.width = 4;
    content.height = 4;
    content.pixels.resize(4u * 4u * 3u);
    for (std::size_t index = 0; index < content.pixels.size(); index += 3)
    {
        content.pixels[index + 0] = 255;
        content.pixels[index + 1] = (index / 3u) % 2u ? 0 : 255;
        content.pixels[index + 2] = 32;
    }

    seedvr2::RgbImage reference;
    reference.width = 4;
    reference.height = 4;
    reference.pixels.assign(4u * 4u * 3u, 96u);

    seedvr2::RgbImage fixed;
    std::string error;
    require(seedvr2::apply_wavelet_color_fix(content, reference, fixed, error), error.c_str());
    require(fixed.width == 4 && fixed.height == 4, "color fix dimensions");
    require(fixed.pixels.size() == content.pixels.size(), "color fix pixel count");
    require(fixed.pixels[1] > 160, "color fix retains content detail");
    require(fixed.pixels[2] >= 80 && fixed.pixels[2] <= 112, "color fix adopts reference low frequency");

    seedvr2::RgbImage invalid;
    require(!seedvr2::apply_wavelet_color_fix(content, invalid, fixed, error), "reject invalid reference");
    require(error.find("same dimensions") != std::string::npos, "invalid reference error");

    seedvr2::RgbFrameSpool spool;
    require(seedvr2::RgbFrameSpool::create(spool, error), error.c_str());
    require(spool.append(reference, error), error.c_str());
    require(spool.rewind(error), error.c_str());
    seedvr2::RgbImage decoded;
    require(spool.read_next(decoded, error), error.c_str());
    require(decoded.width == reference.width && decoded.height == reference.height &&
                decoded.pixels == reference.pixels,
            "RGB reference spool roundtrip");
    require(!spool.read_next(decoded, error) && error.empty(), "RGB reference spool EOF");

    std::puts("seedvr2-color-fix: ok");
    return 0;
}
