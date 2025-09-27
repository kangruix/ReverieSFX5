# -----------------------------------------------------------------------------
# Reverie core module
# -----------------------------------------------------------------------------

# Header files
file(GLOB REVERIE_CORE_H
    ${CMAKE_SOURCE_DIR}/include/reverie/core/*.h
	${CMAKE_SOURCE_DIR}/include/reverie/core/backend/*.h
	${CMAKE_SOURCE_DIR}/include/reverie/core/geometry/*.h
	${CMAKE_SOURCE_DIR}/include/reverie/core/signal/*.h)

# Source files
file(GLOB REVERIE_CORE_C
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.inl
	${CMAKE_CURRENT_SOURCE_DIR}/signal/*.cpp)

# CUDA files
if (CMAKE_CUDA_COMPILER)
	file (GLOB REVERIE_CORE_CU
		${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.cu
		${CMAKE_CURRENT_SOURCE_DIR}/signal/*.cu)
	list(APPEND REVERIE_CORE_C ${REVERIE_CORE_CU})
endif()

# Create library
add_library(reverie_core STATIC ${REVERIE_CORE_C} ${REVERIE_CORE_H})

# Dependencies
target_include_directories(reverie_core
    PUBLIC ${CMAKE_SOURCE_DIR}/include
	PUBLIC ${CMAKE_SOURCE_DIR}/ext/nanobind/include)

# OpenMP support
find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
	target_link_libraries(reverie_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA support
if (CMAKE_CUDA_COMPILER)
	target_include_directories(reverie_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
	target_compile_definitions(reverie_core PUBLIC REV_ENABLE_CUDA)
	target_compile_options(reverie_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# Python bindings
file(GLOB REVERIE_CORE_PY ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)

nanobind_add_module(revpy_core STABLE_ABI ${REVERIE_CORE_PY})
set_target_properties(revpy_core PROPERTIES OUTPUT_NAME "core")
target_link_libraries(revpy_core PRIVATE reverie_core)

# Install directive for scikit-build-core
install(TARGETS revpy_core LIBRARY DESTINATION reverie/core)

# Organize files in Visual Studio
source_group(TREE ${CMAKE_SOURCE_DIR}/include/reverie/core PREFIX "Header Files" FILES ${REVERIE_CORE_H})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files" FILES ${REVERIE_CORE_C})