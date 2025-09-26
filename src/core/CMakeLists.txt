# -----------------------------------------------------------------------------
# Reverie core module
# -----------------------------------------------------------------------------

# Header files
file(GLOB REVERIE_CORE_H
    ${CMAKE_SOURCE_DIR}/include/reverie/core/*.h)

# Source files
file(GLOB REVERIE_CORE_C
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create library
add_library(reverie_core STATIC 
    ${REVERIE_CORE_C} ${REVERIE_CORE_H})

# Dependencies
target_include_directories(reverie_core
	PUBLIC ${CMAKE_SOURCE_DIR}/ext/nanobind/include)

# OpenMP support
find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
	target_link_libraries(reverie_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python bindings
file(GLOB REVERIE_CORE_PY 
    ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)

nanobind_add_module(pyreverie_core STABLE_ABI ${REVERIE_CORE_PY})
set_target_properties(pyreverie_core PROPERTIES OUTPUT_NAME "core")
target_link_libraries(pyreverie_core PRIVATE reverie_core)

# Install directive for scikit-build-core
install(TARGETS pyreverie_core LIBRARY DESTINATION reverie/core)

# Organize files in Visual Studio
source_group(TREE ${CMAKE_SOURCE_DIR}/include/reverie/core PREFIX "Header Files" FILES ${REVERIE_CORE_H})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files" FILES ${REVERIE_CORE_C})