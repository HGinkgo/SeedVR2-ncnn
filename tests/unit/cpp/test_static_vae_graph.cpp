#include "vae/static_vae_graph.h"
#include "inference/performance_profile.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "static VAE graph test failed: " << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main()
{
    using namespace seedvr2;

    require(select_vae_graph_mode(256, 256, 0) == VaeGraphMode::Static256,
            "plain 256x256 should select the fixed graph");
    require(select_vae_graph_mode(128, 128, 0) == VaeGraphMode::Dynamic,
            "other sizes should retain the dynamic graph");
    require(select_vae_graph_mode(256, 256, 128) == VaeGraphMode::Dynamic,
            "tiled 256x256 should retain the dynamic graph");
    require(vae_graph_uses_light_mode(VaeGraphMode::Static256),
            "fixed graph should enable light mode");
    require(!vae_graph_uses_light_mode(VaeGraphMode::Dynamic),
            "dynamic graph should not enable light mode");
    require(std::string(vae_graph_mode_name(VaeGraphMode::Static256)) == "static-256",
            "fixed graph profile name should be stable");
    require(std::string(vae_graph_mode_name(VaeGraphMode::Dynamic)) == "dynamic",
            "dynamic graph profile name should be stable");
    require(format_profile_mode_line("vae-graph", "static-256") ==
                "profile name=vae-graph mode=static-256",
            "profile mode line should be machine-readable");

    const std::string dynamic_param =
        "7767517\n"
        "Input in0 0 1 in0\n"
        "Reshape reshape_0 1 1 in0 out0 0=0 1=0 2=0 6=\"0w,0h,0d\"\n"
        "Convolution conv0 1 1 out0 out1 0=3 1=1 6=\"keep\"\n"
        "Reshape reshape_1 1 1 out1 out2 0=0 1=0 2=0 6=16\n"
        "Reshape reshape_2 1 1 out2 out3 0=0 1=0 2=0 6=\"keep\"\n";
    const std::string expected_static_param =
        "7767517\n"
        "Input in0 0 1 in0\n"
        "Reshape reshape_0 1 1 in0 out0 0=0 1=0 2=0\n"
        "Convolution conv0 1 1 out0 out1 0=3 1=1 6=\"keep\"\n"
        "Reshape reshape_1 1 1 out1 out2 0=0 1=0 2=0 6=16\n"
        "Reshape reshape_2 1 1 out2 out3 0=0 1=0 2=0 6=\"keep\"\n";
    std::string static_param;
    std::size_t removed_fields = 0;
    std::string error;
    require(materialize_static_vae_param(dynamic_param, static_param, removed_fields, error),
            "dynamic parameter should be materialized");
    require(static_param == expected_static_param,
            "only dynamic Reshape shape expressions should be removed");
    require(removed_fields == 1, "one dynamic shape expression should be removed");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "seedvr2-static-vae-graph-test";
    std::filesystem::create_directories(root);
    const std::filesystem::path stem = root / "vae_encode";
    {
        std::ofstream file(stem.string() + ".ncnn.param");
        require(file.good(), "temporary parameter file should open");
        file << dynamic_param;
    }
    PreparedVaeGraph prepared;
    require(prepare_vae_graph(stem, 256, 256, 0, prepared, error),
            "256 dynamic parameter should prepare successfully");
    require(prepared.mode == VaeGraphMode::Static256,
            "prepared dynamic parameter should select static mode");
    require(prepared.param_contents == expected_static_param,
            "prepared parameter should be loaded in memory");
    require(prepared.model_path == stem.string() + ".ncnn.bin",
            "prepared model path should keep the original bin");

    PreparedVaeGraph dynamic;
    require(prepare_vae_graph(stem, 128, 128, 0, dynamic, error),
            "non-256 graph should not inspect the parameter file");
    require(dynamic.mode == VaeGraphMode::Dynamic && dynamic.param_contents.empty(),
            "non-256 graph should use the path-based dynamic load");

    std::filesystem::remove_all(root);
    std::cout << "seedvr2-static-vae-graph-test: ok\n";
    return 0;
}
