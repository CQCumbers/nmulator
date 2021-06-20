cmake_minimum_required(VERSION 3.9)
project(nmulator VERSION 0.1.0 LANGUAGES CXX)

# Add asmjit, glad, and SDL2
set(ASMJIT_STATIC TRUE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/asmjit)
add_subdirectory(extern/glad)
add_subdirectory(extern/sdl2)

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
target_compile_definitions(SDL2-static PRIVATE "-DSDL_LEAN_AND_MEAN=1")
target_link_libraries(nmulator PRIVATE asmjit glad SDL2main SDL2-static)
add_dependencies(nmulator shader)

# Set nmulator compiler options
target_compile_options(nmulator PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -msse4.2
        -fno-exceptions -fno-rtti -fno-strict-aliasing>
    $<$<CXX_COMPILER_ID:MSVC>:/W2 /EHsc /GR->)
target_compile_features(nmulator PRIVATE cxx_std_20)
