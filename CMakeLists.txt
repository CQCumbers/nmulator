cmake_minimum_required(VERSION 3.9)
project(nmulator VERSION 0.1.0 LANGUAGES CXX)

# Add asmjit and SDL2
set(ASMJIT_STATIC TRUE)
add_subdirectory(extern/asmjit)
add_subdirectory(extern/glad)
find_package(SDL2 REQUIRED)

# Compile shader to spir-v
find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
set(GLSL  "${CMAKE_CURRENT_SOURCE_DIR}/src/rdp.comp")
set(SPIRV "${CMAKE_CURRENT_SOURCE_DIR}/src/comp.spv")
add_custom_command(OUTPUT ${SPIRV}
    COMMAND ${GLSLANG_VALIDATOR} -V ${GLSL} --vn g_main -o ${SPIRV}
    DEPENDS ${GLSL})
add_custom_target(shader DEPENDS ${SPIRV})

# Link nmulator against dependencies
add_executable(nmulator src/main.cpp src/r4300.cpp src/rsp.cpp
    src/rdp.cpp src/scheduler.cpp src/debugger.cpp src/mipsjit.cpp)
target_include_directories(nmulator PRIVATE ${SDL2_INCLUDE_DIRS})
string(STRIP ${SDL2_LIBRARIES} SDL2_LIBRARIES)
target_link_libraries(nmulator PRIVATE asmjit glad ${SDL2_LIBRARIES})
add_dependencies(nmulator shader)

# Set nmulator compiler options
target_compile_options(nmulator PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -msse4.2
        -fno-exceptions -fno-rtti -fno-strict-aliasing>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /EHsc /GR->)
target_compile_features(nmulator PRIVATE cxx_std_20)
