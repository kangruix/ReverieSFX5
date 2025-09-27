# -----------------------------------------------------------------------------
# Reverie gui module
# -----------------------------------------------------------------------------

# GLAD
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/glad 
    ${CMAKE_BINARY_DIR}/ext/glad)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/glfw 
    ${CMAKE_BINARY_DIR}/ext/glfw)

# ImGui
set(IMGUI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/imgui)
file(GLOB IMGUI_FILES ${IMGUI_SOURCE_DIR}/*.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_FILES})
target_link_libraries(imgui PUBLIC glad glfw)
target_include_directories(imgui PUBLIC 
    ${IMGUI_SOURCE_DIR} ${IMGUI_SOURCE_DIR}/backends)

# Header files
file(GLOB_RECURSE REVERIE_GUI_H 
    ${CMAKE_SOURCE_DIR}/include/reverie/gui/*.h)

# Source files
file(GLOB_RECURSE REVERIE_GUI_C 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create library
add_library(reverie_gui STATIC 
    ${REVERIE_GUI_C} ${REVERIE_GUI_H})

# Dependencies
target_link_libraries(reverie_gui 
    PUBLIC reverie_core imgui)

# Organize files in Visual Studio
source_group(TREE ${CMAKE_SOURCE_DIR}/include/reverie/gui PREFIX "Header Files" FILES ${REVERIE_GUI_H})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source Files" FILES ${REVERIE_GUI_C})