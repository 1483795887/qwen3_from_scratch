
find_program(UV_PYTHON
    NAMES python3 python
    PATHS "${CMAKE_SOURCE_DIR}/.venv/bin"
    NO_DEFAULT_PATH
)
if(NOT UV_PYTHON)
    execute_process(
        COMMAND uv python which
        OUTPUT_VARIABLE UV_PYTHON
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

execute_process(COMMAND ${UV_PYTHON} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND "${UV_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND "${UV_PYTHON}" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${UV_PYTHON} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${UV_PYTHON} -c "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${UV_PYTHON} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
