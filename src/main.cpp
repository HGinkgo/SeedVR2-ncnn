#include <cstdio>
#include <cstring>

#include "net.h"
#include "platform.h"

namespace
{

void print_usage()
{
    std::puts("Usage: seedvr2-ncnn [options]");
    std::puts("");
    std::puts("Options:");
    std::puts("  --help       Show this help message");
    std::puts("  --version    Show SeedVR2-ncnn and ncnn versions");
}

void print_version()
{
    // Constructing Net keeps the CLI smoke test tied to the linked ncnn API.
    ncnn::Net net;
    (void)net;
    std::printf("SeedVR2-ncnn %s\n", SEEDVR2_VERSION);
    std::printf("ncnn %s\n", NCNN_VERSION_STRING);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        print_usage();
        return 0;
    }

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0)
    {
        print_usage();
        return 0;
    }

    if (argc == 2 && std::strcmp(argv[1], "--version") == 0)
    {
        print_version();
        return 0;
    }

    std::fprintf(stderr, "error: unsupported arguments\n\n");
    print_usage();
    return 2;
}
