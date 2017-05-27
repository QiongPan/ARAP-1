get_filename_component(PROJECTNAME ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")
file(GLOB_RECURSE SHADERS "*.glsl")

add_executable(${PROJECTNAME} ${SOURCES} ${HEADERS} ${SHADERS})
target_link_libraries(${PROJECTNAME} surface_mesh)
# Lastly, additional libraries may have been built for you.  In addition to linking
# against NanoGUI, we need to link against those as well.
target_link_libraries(${PROJECTNAME} nanogui ${NANOGUI_EXTRA_LIBS})

