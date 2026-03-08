cmake_minimum_required(VERSION 3.18)
project(qwen3_from_scratch_kernels LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Release)

include(cmake/find_pytorch_vars.cmake)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

include(CheckLanguage)
check_language(CUDA)

file(GLOB_RECURSE SOURCES "csrc/*/*.cpp")
list(APPEND SOURCES csrc/py11bind.cpp)
file(GLOB_RECURSE CUDA_SOURCES "csrc/*/*.cu")
# 如果有 CUDA，添加 CUDA 语言支持
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "✅ 检测到 CUDA，将编译 CUDA 版本算子）")
    add_definitions(-DUSE_CUDA)
    list(APPEND SOURCES ${CUDA_SOURCES})
else()
    message(STATUS "⚠️ 未检测到 CUDA，仅编译 CPU 版本算子")
endif()

message(STATUS "Found sources: ${SOURCES}")
message(STATUS "PyTorch ABI: ${TORCH_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})

add_library(ops SHARED ${SOURCES})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}")
message(STATUS "✅ 添加依赖项：${TORCH_LIBRARIES}")
target_link_libraries(ops PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(ops PRIVATE csrc/include)
target_include_directories(ops PRIVATE "${TORCH_INCLUDE_DIRS}")
target_include_directories(ops PRIVATE "${PYTHON_INCLUDE_DIR}")
set_target_properties(ops PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    PREFIX ""
    INSTALL_RPATH "${TORCH_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_definitions(ops PRIVATE
    TORCH_EXTENSION_NAME=ops
)
set(PYTHON_OPS_DIR "${CMAKE_SOURCE_DIR}/src/qwen3_from_scratch/kernels")

add_custom_command(
    TARGET ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    $<TARGET_FILE:ops>
    ${PYTHON_OPS_DIR}/$<TARGET_FILE_NAME:ops>
)
