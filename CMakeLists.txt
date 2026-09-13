cmake_minimum_required(VERSION 3.15)

project(SeedVR2-ncnn VERSION 0.1.2 LANGUAGES CXX)

option(SEEDVR2_ENABLE_VULKAN "Build ncnn with Vulkan compute support" OFF)
option(SEEDVR2_ENABLE_FFMPEG "Enable optional FFmpeg video input" OFF)
option(SEEDVR2_BUILD_TESTS "Build the compact runtime contract check" OFF)

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
    src/vae/depth_to_space.cpp
    src/vae/static_vae_graph.cpp)
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
endif()

include(GNUInstallDirs)
install(TARGETS seedvr2-ncnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(SEEDVR2_BUILD_TESTS)
    enable_testing()
    add_executable(seedvr2-runtime-contract-test
        tests/runtime_contract.cpp
        src/cli/cli.cpp
        src/inference/performance_profile.cpp
        src/resolution/resolution_plan.cpp
        src/sampler/sampler.cpp
        src/video/video_io.cpp
        src/video/ffmpeg_video_reader.cpp)
    target_include_directories(seedvr2-runtime-contract-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(SEEDVR2_ENABLE_FFMPEG)
        target_compile_definitions(seedvr2-runtime-contract-test PRIVATE SEEDVR2_HAS_FFMPEG=1)
        target_include_directories(seedvr2-runtime-contract-test PRIVATE "${SEEDVR2_FFMPEG_INCLUDE_DIR}")
        target_link_libraries(seedvr2-runtime-contract-test PRIVATE ${SEEDVR2_FFMPEG_LIBRARIES})
    endif()
    add_test(NAME seedvr2-runtime-contract-test COMMAND seedvr2-runtime-contract-test)
    if(UNIX AND SEEDVR2_ENABLE_FFMPEG)
        get_filename_component(SEEDVR2_FFMPEG_LIBRARY_DIR "${SEEDVR2_FFMPEG_AVFORMAT_LIBRARY}" DIRECTORY)
        set_tests_properties(seedvr2-runtime-contract-test PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=${SEEDVR2_FFMPEG_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}")
    endif()
endif()
