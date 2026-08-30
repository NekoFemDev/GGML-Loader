cmake_minimum_required(VERSION 3.14)
project(ggml-loader LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_LOADER_VULKAN "Build ggml with the Vulkan backend (baseline)" ON)
option(GGML_LOADER_OPENCL "Build ggml with the OpenCL backend" OFF)

set(GGML_VULKAN ${GGML_LOADER_VULKAN} CACHE BOOL "" FORCE)
set(GGML_OPENCL ${GGML_LOADER_OPENCL} CACHE BOOL "" FORCE)

# SDXL/diffusion-model gguf files (from stable-diffusion.cpp's conversion) carry deeply-nested
# original-checkpoint tensor names (e.g. U-Net block paths) that routinely exceed ggml's default
# 64-char GGML_MAX_NAME -- observed directly: a 76-char name failed to load with the default.
# GGML_MAX_NAME is intentionally #ifndef-guarded in ggml.h for exactly this override; must be
# defined before add_subdirectory so ggml's own compiled objects (not just our code) agree on the
# struct layout -- a mismatch here would be a real, silent ABI bug, not just a compile warning.
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(third_party/ggml)

add_executable(ggml-loader
    src/main.cpp
    src/model.cpp
    src/ollama_model.cpp
    src/cmd_inspect.cpp
    src/cmd_run.cpp
    src/cmd_chat.cpp
    src/cmd_bench.cpp
    src/cmd_sdxl.cpp
    src/cmd_clip.cpp
    src/cmd_unet.cpp
    src/cmd_txt2img.cpp
    src/sdxl_vae.cpp
    src/sdxl_common.cpp
    src/clip_tokenizer.cpp
    src/clip_text.cpp
    src/sdxl_unet.cpp
)
target_link_libraries(ggml-loader PRIVATE ggml)
target_include_directories(ggml-loader PRIVATE src)
