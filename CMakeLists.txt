cmake_minimum_required(VERSION 3.15)

project(SeedVR2-ncnn VERSION 0.1.0 LANGUAGES CXX)

option(SEEDVR2_ENABLE_VULKAN "Build ncnn with Vulkan compute support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
    message(FATAL_ERROR
        "The ncnn submodule is missing. Run: git submodule update --init --recursive")
endif()

# Keep the first application build focused on the library consumed by SeedVR2.
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ${SEEDVR2_ENABLE_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(third_party/ncnn)

add_executable(seedvr2-ncnn src/main.cpp)
target_link_libraries(seedvr2-ncnn PRIVATE ncnn)
target_compile_definitions(seedvr2-ncnn PRIVATE
    SEEDVR2_VERSION="${PROJECT_VERSION}")

include(GNUInstallDirs)
install(TARGETS seedvr2-ncnn RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
