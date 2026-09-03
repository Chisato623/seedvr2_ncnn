cmake_minimum_required(VERSION 3.18)

project(seedvr2_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SEEDVR2_NCNN_COMMIT "946fe3fb14a8dff8c06df763f67be522167b2f00")
set(NCNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn")

if(NOT EXISTS "${NCNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "third_party/ncnn is missing. Clone with --recurse-submodules or run "
        "git submodule update --init --recursive.")
endif()

execute_process(
    COMMAND git -C "${NCNN_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE NCNN_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NCNN_HEAD_RESULT)
if(NOT NCNN_HEAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot inspect the ncnn submodule revision.")
endif()
if(NOT NCNN_HEAD STREQUAL SEEDVR2_NCNN_COMMIT)
    message(FATAL_ERROR
        "Unsupported ncnn revision ${NCNN_HEAD}; expected ${SEEDVR2_NCNN_COMMIT}.")
endif()

function(apply_seedvr2_patch patch_path)
    execute_process(
        COMMAND git -C "${NCNN_SOURCE_DIR}" apply --check "${patch_path}"
        RESULT_VARIABLE PATCH_RESULT)
    if(PATCH_RESULT EQUAL 0)
        execute_process(
            COMMAND git -C "${NCNN_SOURCE_DIR}" apply "${patch_path}"
            RESULT_VARIABLE APPLY_RESULT)
        if(NOT APPLY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${patch_path}.")
        endif()
        return()
    endif()

    execute_process(
        COMMAND git -C "${NCNN_SOURCE_DIR}" apply --reverse --check "${patch_path}"
        RESULT_VARIABLE REVERSE_RESULT)
    if(NOT REVERSE_RESULT EQUAL 0)
        message(FATAL_ERROR "ncnn patch is neither applied nor applicable: ${patch_path}")
    endif()
endfunction()

apply_seedvr2_patch(
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/ncnn/0001-nvidia-sdpa-softmax-sync.patch")
apply_seedvr2_patch(
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/ncnn/0002-seedvr2-large-vulkan-workloads.patch")

set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_C_API OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
set(NCNN_SIMPLEVK ON CACHE BOOL "" FORCE)

add_subdirectory("${NCNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ncnn")

add_library(seedvr2_runtime STATIC
    src/seedvr2/adaptive_window_attention.cpp
    src/seedvr2/decomposed_conv3d.cpp
    src/seedvr2/seedvr2_dit.cpp
    src/seedvr2/seedvr2_pipeline.cpp
    src/seedvr2/seedvr2_vae.cpp
    src/seedvr2/shared_linear.cpp
    src/seedvr2/vae_chunked_mha.cpp
    src/seedvr2/vae_downsample3d.cpp
    src/seedvr2/vae_upsample3d.cpp)
target_include_directories(seedvr2_runtime PUBLIC src)
target_link_libraries(seedvr2_runtime PUBLIC ncnn)

add_executable(seedvr2_main
    examples/seedvr2_main.cpp
    src/seedvr2/seedvr2_video_io.cpp)
target_link_libraries(seedvr2_main PRIVATE seedvr2_runtime)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
target_link_libraries(seedvr2_main PRIVATE PkgConfig::FFMPEG)

if(UNIX AND NOT APPLE)
    target_link_libraries(seedvr2_main PRIVATE dl pthread)
endif()
