cmake_minimum_required(VERSION 3.18)
project(Reverie LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Global build options
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA
if (NOT FORCE_CPU)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES 75)

        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

# Python bindings
add_subdirectory(ext/nanobind)
find_package(Python 3.8 
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# Debug Python configuration
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python development found: ${Python_Development.Module_FOUND}")

# -----------------------------------------------------------------------------
# Build module libraries
# -----------------------------------------------------------------------------

add_subdirectory(src/core)
add_subdirectory(src/fdtd)

# -----------------------------------------------------------------------------
# Build main executable
# -----------------------------------------------------------------------------

add_executable(main src/main.cpp)
target_link_libraries(main reverie_fdtd)

# Organize files in Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)