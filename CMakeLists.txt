cmake_minimum_required(VERSION 3.15)

project(SeedVR2-ncnn VERSION 0.1.0 LANGUAGES CXX)

option(SEEDVR2_ENABLE_VULKAN "Build ncnn with Vulkan compute support" OFF)
option(SEEDVR2_ENABLE_FFMPEG "Enable optional FFmpeg video input" OFF)
option(SEEDVR2_BUILD_TESTS "Build fast native unit and backend tests" OFF)
option(SEEDVR2_BUILD_INTEGRATION_TESTS "Build model and end-to-end integration tests" OFF)
option(SEEDVR2_BUILD_BENCHMARKS "Build focused native benchmarks" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
    message(FATAL_ERROR
        "The ncnn submodule is missing. Run: git submodule update --init --recursive")
endif()

if(SEEDVR2_ENABLE_FFMPEG)
    find_path(SEEDVR2_FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h)
    find_library(SEEDVR2_FFMPEG_AVFORMAT_LIBRARY NAMES avformat)
    find_library(SEEDVR2_FFMPEG_AVCODEC_LIBRARY NAMES avcodec)
    find_library(SEEDVR2_FFMPEG_AVUTIL_LIBRARY NAMES avutil)
    find_library(SEEDVR2_FFMPEG_SWSCALE_LIBRARY NAMES swscale)
    if(NOT SEEDVR2_FFMPEG_INCLUDE_DIR OR NOT SEEDVR2_FFMPEG_AVFORMAT_LIBRARY OR
       NOT SEEDVR2_FFMPEG_AVCODEC_LIBRARY OR NOT SEEDVR2_FFMPEG_AVUTIL_LIBRARY OR
       NOT SEEDVR2_FFMPEG_SWSCALE_LIBRARY)
        message(FATAL_ERROR
            "SEEDVR2_ENABLE_FFMPEG=ON requires FFmpeg development headers and shared libraries "
            "(libavformat, libavcodec, libavutil, libswscale). Install the development package "
            "or configure with -DSEEDVR2_ENABLE_FFMPEG=OFF.")
    endif()
    set(SEEDVR2_FFMPEG_LIBRARIES
        ${SEEDVR2_FFMPEG_AVFORMAT_LIBRARY}
        ${SEEDVR2_FFMPEG_AVCODEC_LIBRARY}
        ${SEEDVR2_FFMPEG_AVUTIL_LIBRARY}
        ${SEEDVR2_FFMPEG_SWSCALE_LIBRARY})
endif()

# Keep the first application build focused on the library consumed by SeedVR2.
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ${SEEDVR2_ENABLE_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(third_party/ncnn)

add_executable(seedvr2-ncnn
    src/main.cpp
    src/cli/cli.cpp
    src/conditioning/conditioning.cpp
    src/io/image_io.cpp
    src/inference/image_inference.cpp
    src/inference/latent_spool.cpp
    src/inference/memory_diagnostics.cpp
    src/inference/performance_profile.cpp
    src/model/model_registry.cpp
    src/video/video_io.cpp
    src/video/ffmpeg_video_reader.cpp
    src/awa/awa_layers.cpp
    src/resolution/resolution_plan.cpp
    src/vae/temporal_pad.cpp
    src/vae/causal_conv3d.cpp
    src/vae/depth_to_space.cpp)
target_link_libraries(seedvr2-ncnn PRIVATE ncnn)
target_include_directories(seedvr2-ncnn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")
target_compile_definitions(seedvr2-ncnn PRIVATE
    SEEDVR2_VERSION="${PROJECT_VERSION}")
if(SEEDVR2_ENABLE_FFMPEG)
    target_compile_definitions(seedvr2-ncnn PRIVATE SEEDVR2_HAS_FFMPEG=1)
    target_include_directories(seedvr2-ncnn PRIVATE "${SEEDVR2_FFMPEG_INCLUDE_DIR}")
    target_link_libraries(seedvr2-ncnn PRIVATE ${SEEDVR2_FFMPEG_LIBRARIES})
endif()

if(SEEDVR2_ENABLE_VULKAN)
    target_sources(seedvr2-ncnn PRIVATE
        src/dit/dit_stack.cpp
        src/sampler/vulkan_sampler.cpp)
    set(SEEDVR2_AWA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/awa/shader")
    set(SEEDVR2_AWA_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(SEEDVR2_AWA_SHADER_HEADERS)
    foreach(SEEDVR2_AWA_SHADER awa_pack awa_unpack awa_attention_score awa_attention_softmax awa_attention_value mmrope)
        set(SEEDVR2_AWA_SHADER_SRC "${SEEDVR2_AWA_SHADER_DIR}/${SEEDVR2_AWA_SHADER}.comp")
        set(SEEDVR2_AWA_SHADER_HEADER
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SEEDVR2_AWA_SHADER}.comp.hex.h")
        add_custom_command(
            OUTPUT "${SEEDVR2_AWA_SHADER_HEADER}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
            COMMAND ${CMAKE_COMMAND}
                -DSHADER_SRC=${SEEDVR2_AWA_SHADER_SRC}
                -DSHADER_COMP_HEADER=${SEEDVR2_AWA_SHADER_HEADER}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
            DEPENDS "${SEEDVR2_AWA_SHADER_SRC}"
            COMMENT "Preprocessing SeedVR2 AWA shader ${SEEDVR2_AWA_SHADER}.comp"
            VERBATIM)
        set_source_files_properties("${SEEDVR2_AWA_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
        list(APPEND SEEDVR2_AWA_SHADER_HEADERS "${SEEDVR2_AWA_SHADER_HEADER}")
    endforeach()
    set(SEEDVR2_TEMPORAL_PAD_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/temporal_pad.comp")
    set(SEEDVR2_TEMPORAL_PAD_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/temporal_pad.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_TEMPORAL_PAD_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_TEMPORAL_PAD_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE temporal padding shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    set(SEEDVR2_DEPTH_TO_SPACE_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/depth_to_space.comp")
    set(SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/depth_to_space.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_DEPTH_TO_SPACE_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_DEPTH_TO_SPACE_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE depth-to-space shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    set(SEEDVR2_CAUSAL_CONV3D_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/causal_conv3d.comp")
    set(SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/causal_conv3d.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_CAUSAL_CONV3D_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_CAUSAL_CONV3D_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE causal convolution shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    target_sources(seedvr2-ncnn PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
    target_include_directories(seedvr2-ncnn PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")

    if(SEEDVR2_BUILD_BENCHMARKS)
        add_executable(seedvr2-awa-graph-benchmark
            tests/manual/cpp/benchmark_awa_graph.cpp
            src/awa/awa_benchmark.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-awa-graph-benchmark PRIVATE ncnn)
        target_include_directories(seedvr2-awa-graph-benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-awa-graph-benchmark PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS seedvr2-ncnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(SEEDVR2_BUILD_TESTS OR SEEDVR2_BUILD_INTEGRATION_TESTS)
    enable_testing()
endif()

if(SEEDVR2_BUILD_TESTS)
    # Fast unit and backend-contract tests. These are the default CTest surface.
    add_executable(seedvr2-cli-test
        tests/unit/cpp/test_cli.cpp
        src/cli/cli.cpp
        src/inference/memory_diagnostics.cpp
        src/inference/performance_profile.cpp
        src/resolution/resolution_plan.cpp)
    target_include_directories(seedvr2-cli-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-cli-test COMMAND seedvr2-cli-test)

    add_executable(seedvr2-image-io-test
        tests/unit/cpp/test_image_io.cpp
        src/io/image_io.cpp)
    target_include_directories(seedvr2-image-io-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")
    add_test(NAME seedvr2-image-io-test COMMAND seedvr2-image-io-test)

    add_executable(seedvr2-sampler-test
        tests/unit/cpp/test_sampler.cpp
        src/sampler/sampler.cpp)
    target_include_directories(seedvr2-sampler-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-sampler-test COMMAND seedvr2-sampler-test)

    add_executable(seedvr2-resolution-test
        tests/unit/cpp/test_resolution_plan.cpp
        src/resolution/resolution_plan.cpp)
    target_include_directories(seedvr2-resolution-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-resolution-test COMMAND seedvr2-resolution-test)

    add_executable(seedvr2-model-registry-test
        tests/unit/cpp/test_model_registry.cpp
        src/model/model_registry.cpp
        src/resolution/resolution_plan.cpp)
    target_include_directories(seedvr2-model-registry-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-model-registry-test COMMAND seedvr2-model-registry-test)

    if(NOT SEEDVR2_ENABLE_VULKAN)
        add_executable(seedvr2-image-inference-test
            tests/unit/cpp/test_image_inference.cpp
            src/inference/image_inference.cpp
            src/inference/latent_spool.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp)
        target_include_directories(seedvr2-image-inference-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_BINARY_DIR}/third_party/ncnn/src")
        add_test(NAME seedvr2-image-inference-test COMMAND seedvr2-image-inference-test)
    endif()

    add_executable(seedvr2-video-io-test
        tests/unit/cpp/test_video_io.cpp
        src/video/video_io.cpp
        src/video/ffmpeg_video_reader.cpp)
    target_include_directories(seedvr2-video-io-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_FFMPEG)
        target_compile_definitions(seedvr2-video-io-test PRIVATE SEEDVR2_HAS_FFMPEG=1)
        target_include_directories(seedvr2-video-io-test PRIVATE "${SEEDVR2_FFMPEG_INCLUDE_DIR}")
        target_link_libraries(seedvr2-video-io-test PRIVATE ${SEEDVR2_FFMPEG_LIBRARIES})
    endif()
    add_test(NAME seedvr2-video-io-test COMMAND seedvr2-video-io-test)
    if(UNIX AND SEEDVR2_ENABLE_FFMPEG)
        get_filename_component(SEEDVR2_FFMPEG_LIBRARY_DIR "${SEEDVR2_FFMPEG_AVFORMAT_LIBRARY}" DIRECTORY)
        set_tests_properties(seedvr2-video-io-test PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=${SEEDVR2_FFMPEG_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}")
    endif()

    add_executable(seedvr2-awa-benchmark-test
        tests/unit/cpp/test_awa_benchmark.cpp
        src/awa/awa_benchmark.cpp)
    target_include_directories(seedvr2-awa-benchmark-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-awa-benchmark-test COMMAND seedvr2-awa-benchmark-test)

    add_executable(seedvr2-awa-test
        tests/unit/cpp/test_awa_layers.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-awa-test PRIVATE ncnn)
    target_include_directories(seedvr2-awa-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-awa-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_include_directories(seedvr2-awa-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()
    add_test(NAME seedvr2-awa-test COMMAND seedvr2-awa-test)

    add_executable(seedvr2-awa-dynamic-test
        tests/unit/cpp/test_awa_dynamic_metadata.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-awa-dynamic-test PRIVATE ncnn)
    target_include_directories(seedvr2-awa-dynamic-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-awa-dynamic-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_include_directories(seedvr2-awa-dynamic-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()
    add_test(NAME seedvr2-awa-dynamic-test COMMAND seedvr2-awa-dynamic-test)

    add_executable(seedvr2-temporal-pad-test
        tests/unit/cpp/test_temporal_pad.cpp
        src/vae/temporal_pad.cpp
        src/vae/causal_conv3d.cpp
        src/vae/depth_to_space.cpp)
    target_link_libraries(seedvr2-temporal-pad-test PRIVATE ncnn)
    target_include_directories(seedvr2-temporal-pad-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-temporal-pad-test PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
        target_sources(seedvr2-temporal-pad-test PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
        target_include_directories(seedvr2-temporal-pad-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()
    add_test(NAME seedvr2-temporal-pad-test COMMAND seedvr2-temporal-pad-test)

    add_executable(seedvr2-depth-to-space-test
        tests/unit/cpp/test_depth_to_space.cpp
        src/vae/depth_to_space.cpp
        src/vae/causal_conv3d.cpp
        src/vae/temporal_pad.cpp)
    target_link_libraries(seedvr2-depth-to-space-test PRIVATE ncnn)
    target_include_directories(seedvr2-depth-to-space-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-depth-to-space-test PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
        target_include_directories(seedvr2-depth-to-space-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()
    add_test(NAME seedvr2-depth-to-space-test COMMAND seedvr2-depth-to-space-test)
endif()

if(SEEDVR2_BUILD_TESTS AND SEEDVR2_ENABLE_VULKAN)
    add_executable(seedvr2-awa-vulkan-capability-test
        tests/unit/cpp/test_awa_vulkan_capability.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-awa-vulkan-capability-test PRIVATE ncnn)
    target_include_directories(seedvr2-awa-vulkan-capability-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_sources(seedvr2-awa-vulkan-capability-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    target_include_directories(seedvr2-awa-vulkan-capability-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    add_test(NAME seedvr2-awa-vulkan-capability-test COMMAND seedvr2-awa-vulkan-capability-test)

    add_executable(seedvr2-temporal-pad-vulkan-test
        tests/unit/cpp/test_temporal_pad_vulkan.cpp
        src/vae/temporal_pad.cpp
        src/vae/causal_conv3d.cpp
        src/vae/depth_to_space.cpp)
    target_link_libraries(seedvr2-temporal-pad-vulkan-test PRIVATE ncnn)
    target_include_directories(seedvr2-temporal-pad-vulkan-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    target_sources(seedvr2-temporal-pad-vulkan-test PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
    target_sources(seedvr2-temporal-pad-vulkan-test PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
    add_test(NAME seedvr2-temporal-pad-vulkan-test COMMAND seedvr2-temporal-pad-vulkan-test)

    add_executable(seedvr2-depth-to-space-vulkan-test
        tests/unit/cpp/test_depth_to_space_vulkan.cpp
        src/vae/depth_to_space.cpp)
    target_link_libraries(seedvr2-depth-to-space-vulkan-test PRIVATE ncnn)
    target_include_directories(seedvr2-depth-to-space-vulkan-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    target_sources(seedvr2-depth-to-space-vulkan-test PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
    add_test(NAME seedvr2-depth-to-space-vulkan-test COMMAND seedvr2-depth-to-space-vulkan-test)

    add_executable(seedvr2-dit-patch-vulkan-test
        tests/unit/cpp/test_dit_patch_vulkan.cpp
        src/dit/dit_stack.cpp
        src/inference/performance_profile.cpp
        src/resolution/resolution_plan.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-dit-patch-vulkan-test PRIVATE ncnn)
    target_include_directories(seedvr2-dit-patch-vulkan-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    target_sources(seedvr2-dit-patch-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    add_test(NAME seedvr2-dit-patch-vulkan-test COMMAND seedvr2-dit-patch-vulkan-test)

    add_executable(seedvr2-sampler-vulkan-test
        tests/unit/cpp/test_sampler_vulkan.cpp
        src/sampler/sampler.cpp
        src/sampler/vulkan_sampler.cpp)
    target_link_libraries(seedvr2-sampler-vulkan-test PRIVATE ncnn)
    target_include_directories(seedvr2-sampler-vulkan-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-sampler-vulkan-test COMMAND seedvr2-sampler-vulkan-test)

    add_executable(seedvr2-awa-unpack-vulkan-test
        tests/unit/cpp/test_awa_unpack_vulkan.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-awa-unpack-vulkan-test PRIVATE ncnn)
    target_include_directories(seedvr2-awa-unpack-vulkan-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    target_sources(seedvr2-awa-unpack-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    add_test(NAME seedvr2-awa-unpack-vulkan-test COMMAND seedvr2-awa-unpack-vulkan-test)
endif()

if(SEEDVR2_BUILD_INTEGRATION_TESTS)
    # Model-backed graph-load and end-to-end targets. Run these explicitly.
    add_executable(seedvr2-conditioning-test
        tests/integration/cpp/test_conditioning.cpp
        src/conditioning/conditioning.cpp)
    target_link_libraries(seedvr2-conditioning-test PRIVATE ncnn)
    target_include_directories(seedvr2-conditioning-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

    add_executable(seedvr2-awa-graph-load-test
        tests/integration/cpp/test_awa_graph_load.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-awa-graph-load-test PRIVATE ncnn)
    target_include_directories(seedvr2-awa-graph-load-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-awa-graph-load-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_include_directories(seedvr2-awa-graph-load-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()

    add_executable(seedvr2-dit-block-graph-load-test
        tests/integration/cpp/test_dit_block_graph_load.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-dit-block-graph-load-test PRIVATE ncnn)
    target_include_directories(seedvr2-dit-block-graph-load-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-dit-block-graph-load-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_include_directories(seedvr2-dit-block-graph-load-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()

    add_executable(seedvr2-vae-graph-load-test
        tests/integration/cpp/test_vae_graph_load.cpp
        src/vae/temporal_pad.cpp
        src/vae/causal_conv3d.cpp
        src/vae/depth_to_space.cpp)
    target_link_libraries(seedvr2-vae-graph-load-test PRIVATE ncnn)
    target_include_directories(seedvr2-vae-graph-load-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-vae-graph-load-test PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
        target_sources(seedvr2-vae-graph-load-test PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
        target_sources(seedvr2-vae-graph-load-test PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
        target_include_directories(seedvr2-vae-graph-load-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()

    add_executable(seedvr2-vae-end-to-end-cpu-test
        tests/integration/cpp/test_vae_end_to_end_cpu.cpp
        src/vae/temporal_pad.cpp
        src/vae/causal_conv3d.cpp
        src/vae/depth_to_space.cpp)
    target_link_libraries(seedvr2-vae-end-to-end-cpu-test PRIVATE ncnn)
    target_include_directories(seedvr2-vae-end-to-end-cpu-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_VULKAN)
        target_sources(seedvr2-vae-end-to-end-cpu-test PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
        target_sources(seedvr2-vae-end-to-end-cpu-test PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
        target_sources(seedvr2-vae-end-to-end-cpu-test PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
        target_include_directories(seedvr2-vae-end-to-end-cpu-test PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
    endif()

    if(SEEDVR2_ENABLE_VULKAN)
        add_executable(seedvr2-vae-end-to-end-vulkan-test
            tests/integration/cpp/test_vae_end_to_end_vulkan.cpp
            src/vae/temporal_pad.cpp
            src/vae/causal_conv3d.cpp
            src/vae/depth_to_space.cpp)
        target_link_libraries(seedvr2-vae-end-to-end-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-vae-end-to-end-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-vae-end-to-end-vulkan-test PRIVATE
            "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
            "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")

        add_executable(seedvr2-awa-graph-vulkan-test
            tests/integration/cpp/test_awa_graph_vulkan.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-awa-graph-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-awa-graph-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-awa-graph-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-dit-block-end-to-end-vulkan-test
            tests/integration/cpp/test_dit_block_end_to_end_vulkan.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-dit-block-end-to-end-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-dit-block-end-to-end-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-dit-block-end-to-end-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-dit-stack-vulkan-test
            tests/integration/cpp/test_dit_stack_vulkan.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-dit-stack-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-dit-stack-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-dit-stack-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-dit-cfg-vulkan-test
            tests/integration/cpp/test_dit_cfg_vulkan.cpp
            src/dit/dit_stack.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp
            src/sampler/sampler.cpp
            src/sampler/vulkan_sampler.cpp
            src/conditioning/conditioning.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-dit-cfg-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-dit-cfg-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-dit-cfg-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-dynamic-dit-stack-vulkan-test
            tests/integration/cpp/test_dynamic_dit_stack_vulkan.cpp
            src/dit/dit_stack.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp
            src/conditioning/conditioning.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-dynamic-dit-stack-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-dynamic-dit-stack-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-dynamic-dit-stack-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-vae-dit-vae-vulkan-test
            tests/integration/cpp/test_vae_dit_vae_vulkan.cpp
            src/awa/awa_layers.cpp
            src/vae/temporal_pad.cpp
            src/vae/causal_conv3d.cpp
            src/vae/depth_to_space.cpp)
        target_link_libraries(seedvr2-vae-dit-vae-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-vae-dit-vae-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-vae-dit-vae-vulkan-test PRIVATE
            ${SEEDVR2_AWA_SHADER_HEADERS}
            "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
            "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")

        add_executable(seedvr2-vae-dit-stack-vulkan-test
            tests/integration/cpp/test_vae_dit_stack_vulkan.cpp
            src/awa/awa_layers.cpp
            src/dit/dit_stack.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp
            src/conditioning/conditioning.cpp
            src/sampler/sampler.cpp
            src/sampler/vulkan_sampler.cpp
            src/vae/temporal_pad.cpp
            src/vae/causal_conv3d.cpp
            src/vae/depth_to_space.cpp)
        target_link_libraries(seedvr2-vae-dit-stack-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-vae-dit-stack-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-vae-dit-stack-vulkan-test PRIVATE
            ${SEEDVR2_AWA_SHADER_HEADERS}
            "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
            "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")

        add_executable(seedvr2-mmrope-vulkan-test
            tests/integration/cpp/test_mmrope_vulkan.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-mmrope-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-mmrope-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-mmrope-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-awa-attention-vulkan-test
            tests/integration/cpp/test_awa_attention_vulkan.cpp
            src/awa/awa_layers.cpp)
        target_link_libraries(seedvr2-awa-attention-vulkan-test PRIVATE ncnn)
        target_include_directories(seedvr2-awa-attention-vulkan-test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")
        target_sources(seedvr2-awa-attention-vulkan-test PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    endif()
endif()
