cmake_minimum_required(VERSION 3.15)

project(SeedVR2-ncnn VERSION 0.1.2 LANGUAGES CXX)

option(SEEDVR2_ENABLE_VULKAN "Build ncnn with Vulkan compute support" OFF)
option(SEEDVR2_ENABLE_FFMPEG "Enable optional FFmpeg video input" OFF)
option(SEEDVR2_BUILD_PT2_SMOKE "Build the optional PT2/ncnn validation runners" OFF)
option(SEEDVR2_BUILD_TESTS "Build focused contract tests" OFF)

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
    src/inference/rgb_spool.cpp
    src/inference/memory_diagnostics.cpp
    src/inference/performance_profile.cpp
    src/postprocess/color_fix.cpp
    src/model/model_registry.cpp
    src/video/video_io.cpp
    src/video/ffmpeg_video_reader.cpp
    src/awa/awa_layers.cpp
    src/resolution/resolution_plan.cpp
    src/sampler/sampler.cpp
    src/vae/temporal_pad.cpp
    src/vae/causal_conv3d.cpp
    src/vae/pointwise_conv3d.cpp
    src/vae/depth_to_space.cpp
    src/vae/static_vae_graph.cpp)
target_link_libraries(seedvr2-ncnn PRIVATE ncnn)
target_include_directories(seedvr2-ncnn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src/layer")
target_compile_definitions(seedvr2-ncnn PRIVATE
    SEEDVR2_VERSION="${PROJECT_VERSION}")
if(SEEDVR2_ENABLE_FFMPEG)
    target_compile_definitions(seedvr2-ncnn PRIVATE SEEDVR2_HAS_FFMPEG=1)
    target_include_directories(seedvr2-ncnn PRIVATE "${SEEDVR2_FFMPEG_INCLUDE_DIR}")
    target_link_libraries(seedvr2-ncnn PRIVATE ${SEEDVR2_FFMPEG_LIBRARIES})
endif()

if(SEEDVR2_BUILD_TESTS)
    enable_testing()
    add_executable(seedvr2-sampler-smoke
        tools/sampler_smoke.cpp
        src/sampler/sampler.cpp)
    target_include_directories(seedvr2-sampler-smoke PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-sampler-smoke COMMAND seedvr2-sampler-smoke)
    add_executable(seedvr2-conditioning-smoke
        tools/conditioning_smoke.cpp
        src/cli/cli.cpp
        src/model/model_registry.cpp
        src/resolution/resolution_plan.cpp)
    target_include_directories(seedvr2-conditioning-smoke PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME seedvr2-conditioning-smoke COMMAND seedvr2-conditioning-smoke)
endif()

if(SEEDVR2_BUILD_PT2_SMOKE)
    add_executable(seedvr2-pt2-smoke tools/pt2_ncnn_smoke.cpp)
    target_link_libraries(seedvr2-pt2-smoke PRIVATE ncnn)
    target_include_directories(seedvr2-pt2-smoke PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")
    target_sources(seedvr2-pt2-smoke PRIVATE src/awa/awa_layers.cpp)

    add_executable(seedvr2-pt2-awa-smoke
        tools/pt2_awa_smoke.cpp
        src/awa/awa_layers.cpp)
    target_link_libraries(seedvr2-pt2-awa-smoke PRIVATE ncnn)
    target_include_directories(seedvr2-pt2-awa-smoke PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")

    add_executable(seedvr2-vae-smoke
        tools/vae_ncnn_smoke.cpp
        src/vae/temporal_pad.cpp
        src/vae/causal_conv3d.cpp
        src/vae/pointwise_conv3d.cpp
        src/vae/depth_to_space.cpp
        src/vae/static_vae_graph.cpp)
    target_link_libraries(seedvr2-vae-smoke PRIVATE ncnn)
    target_include_directories(seedvr2-vae-smoke PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")
endif()

if(SEEDVR2_ENABLE_VULKAN)
    target_sources(seedvr2-ncnn PRIVATE
        src/dit/dit_stack.cpp
        src/sampler/vulkan_sampler.cpp)
    set(SEEDVR2_AWA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/awa/shader")
    set(SEEDVR2_AWA_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(SEEDVR2_AWA_SHADER_HEADERS)
    foreach(SEEDVR2_AWA_SHADER awa_pack awa_unpack awa_attention_score awa_attention_softmax awa_attention_value mmrope
            cast_bf16_f32 cast_f32_bf16)
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
    set(SEEDVR2_DIT_SHADER_HEADERS)
    foreach(SEEDVR2_DIT_SHADER dit_rmsnorm dit_binaryop dit_qkv_concat dit_output_unary)
        set(SEEDVR2_DIT_SHADER_HEADER
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SEEDVR2_DIT_SHADER}.comp.hex.h")
        add_custom_command(
            OUTPUT "${SEEDVR2_DIT_SHADER_HEADER}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
            COMMAND ${CMAKE_COMMAND}
                -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/src/dit/shader/${SEEDVR2_DIT_SHADER}.comp
                -DSHADER_COMP_HEADER=${SEEDVR2_DIT_SHADER_HEADER}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dit/shader/${SEEDVR2_DIT_SHADER}.comp"
            VERBATIM)
        list(APPEND SEEDVR2_DIT_SHADER_HEADERS "${SEEDVR2_DIT_SHADER_HEADER}")
    endforeach()
    set(SEEDVR2_RNE_SHADER_HEADERS)
    foreach(SEEDVR2_RNE_SHADER gemm innerproduct_gemm_wp4 innerproduct_pack4 swish)
        set(SEEDVR2_RNE_SHADER_SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src/layer/vulkan/shader/${SEEDVR2_RNE_SHADER}.comp")
        set(SEEDVR2_RNE_SHADER_SYMBOL "seedvr2_${SEEDVR2_RNE_SHADER}_rne_comp_data")
        set(SEEDVR2_RNE_SHADER_HEADER
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SEEDVR2_RNE_SHADER}_rne.comp.hex.h")
        add_custom_command(
            OUTPUT "${SEEDVR2_RNE_SHADER_HEADER}"
            COMMAND ${CMAKE_COMMAND}
                -DSHADER_SOURCE=${SEEDVR2_RNE_SHADER_SOURCE}
                -DSHADER_HEADER=${SEEDVR2_RNE_SHADER_HEADER}
                -DSHADER_SYMBOL=${SEEDVR2_RNE_SHADER_SYMBOL}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_bf16_rne_shader_header.cmake"
            DEPENDS "${SEEDVR2_RNE_SHADER_SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_bf16_rne_shader_header.cmake"
            COMMENT "Embedding BF16 RNE shader ${SEEDVR2_RNE_SHADER}.comp"
            VERBATIM)
        set_source_files_properties("${SEEDVR2_RNE_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
        list(APPEND SEEDVR2_RNE_SHADER_HEADERS "${SEEDVR2_RNE_SHADER_HEADER}")
    endforeach()
    target_sources(seedvr2-ncnn PRIVATE src/dit/dit_precision.cpp
        ${SEEDVR2_DIT_SHADER_HEADERS} ${SEEDVR2_RNE_SHADER_HEADERS})
    if(SEEDVR2_BUILD_PT2_SMOKE)
        add_executable(seedvr2-dit-precision-smoke tools/dit_precision_smoke.cpp
            src/dit/dit_precision.cpp ${SEEDVR2_DIT_SHADER_HEADERS} ${SEEDVR2_RNE_SHADER_HEADERS})
        target_link_libraries(seedvr2-dit-precision-smoke PRIVATE ncnn)
        target_include_directories(seedvr2-dit-precision-smoke PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src/layer")

        target_sources(seedvr2-pt2-awa-smoke PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_include_directories(seedvr2-pt2-awa-smoke PRIVATE
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")

        add_executable(seedvr2-pt2-dit-stack-smoke
            tools/pt2_dit_stack_smoke.cpp
            src/dit/dit_stack.cpp
            src/awa/awa_layers.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp)
        target_link_libraries(seedvr2-pt2-dit-stack-smoke PRIVATE ncnn)
        target_include_directories(seedvr2-pt2-dit-stack-smoke PRIVATE
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")
        target_sources(seedvr2-pt2-dit-stack-smoke PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})

        add_executable(seedvr2-reference-sampling-smoke
            tools/reference_sampling_smoke.cpp
            src/vae/temporal_pad.cpp
            src/vae/causal_conv3d.cpp
            src/vae/pointwise_conv3d.cpp
            src/vae/depth_to_space.cpp
            src/vae/static_vae_graph.cpp
            src/dit/dit_stack.cpp
            src/awa/awa_layers.cpp
            src/sampler/sampler.cpp
            src/sampler/vulkan_sampler.cpp
            src/inference/performance_profile.cpp
            src/resolution/resolution_plan.cpp
            ${SEEDVR2_AWA_SHADER_HEADERS})
        target_link_libraries(seedvr2-reference-sampling-smoke PRIVATE ncnn)
        target_include_directories(seedvr2-reference-sampling-smoke PRIVATE
            "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src")

    endif()
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
    set(SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/causal_conv3d_oc4.comp")
    set(SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/causal_conv3d_oc4.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE causal output-channel-4 shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    set(SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/causal_conv3d_oc8.comp")
    set(SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/causal_conv3d_oc8.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE causal output-channel-8 shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    set(SEEDVR2_POINTWISE_CONV3D_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/vae/shader/pointwise_conv3d.comp")
    set(SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/pointwise_conv3d.comp.hex.h")
    add_custom_command(
        OUTPUT "${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader"
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_SRC=${SEEDVR2_POINTWISE_CONV3D_SHADER_SRC}
            -DSHADER_COMP_HEADER=${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS "${SEEDVR2_POINTWISE_CONV3D_SHADER_SRC}"
        COMMENT "Preprocessing SeedVR2 VAE pointwise convolution shader"
        VERBATIM)
    set_source_files_properties("${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}" PROPERTIES GENERATED TRUE)
    target_sources(seedvr2-ncnn PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}")
    target_sources(seedvr2-ncnn PRIVATE "${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}")
    target_include_directories(seedvr2-ncnn PRIVATE "${SEEDVR2_AWA_SHADER_INCLUDE_DIR}")

    if(SEEDVR2_BUILD_PT2_SMOKE)
        target_sources(seedvr2-reference-sampling-smoke PRIVATE
            "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
            "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}"
            "${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}")
        target_sources(seedvr2-vae-smoke PRIVATE ${SEEDVR2_AWA_SHADER_HEADERS})
        target_sources(seedvr2-vae-smoke PRIVATE
            "${SEEDVR2_TEMPORAL_PAD_SHADER_HEADER}"
            "${SEEDVR2_DEPTH_TO_SPACE_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_OC4_SHADER_HEADER}"
            "${SEEDVR2_CAUSAL_CONV3D_OC8_SHADER_HEADER}"
            "${SEEDVR2_POINTWISE_CONV3D_SHADER_HEADER}")
    endif()
endif()

if(SEEDVR2_ENABLE_VULKAN AND SEEDVR2_BUILD_PT2_SMOKE)
    foreach(SEEDVR2_DIT_TARGET seedvr2-pt2-dit-stack-smoke seedvr2-reference-sampling-smoke)
        target_sources(${SEEDVR2_DIT_TARGET} PRIVATE src/dit/dit_precision.cpp
            ${SEEDVR2_DIT_SHADER_HEADERS} ${SEEDVR2_RNE_SHADER_HEADERS})
        target_include_directories(${SEEDVR2_DIT_TARGET} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/src/layer")
    endforeach()
endif()

include(GNUInstallDirs)
install(TARGETS seedvr2-ncnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
