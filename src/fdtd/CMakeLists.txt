# -----------------------------------------------------------------------------
# Reverie fdtd module
# -----------------------------------------------------------------------------

# Header files
file(GLOB REVERIE_FDTD_H
    ${CMAKE_SOURCE_DIR}/include/reverie/fdtd/*.h)

# Source files
file(GLOB REVERIE_FDTD_C
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create library
add_library(reverie_fdtd STATIC 
    ${REVERIE_FDTD_C} ${REVERIE_FDTD_H})

# Dependencies
target_link_libraries(reverie_fdtd PUBLIC reverie_core)

# Python bindings
file(GLOB REVERIE_FDTD_PY
    ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)

nanobind_add_module(pyreverie_fdtd STABLE_ABI ${REVERIE_FDTD_PY})
set_target_properties(pyreverie_fdtd PROPERTIES OUTPUT_NAME "fdtd")
target_link_libraries(pyreverie_fdtd PRIVATE reverie_fdtd)

# Install directive for scikit-build-core
install(TARGETS pyreverie_fdtd LIBRARY DESTINATION reverie/fdtd)

# Organize files in Visual Studio
source_group(TREE ${CMAKE_SOURCE_DIR}/include/reverie/fdtd PREFIX "Header Files" FILES ${REVERIE_FDTD_H})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files" FILES ${REVERIE_FDTD_C})