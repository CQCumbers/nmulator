cmake_minimum_required(VERSION 3.9)
project(nmulator VERSION 0.1.0 LANGUAGES CXX)

# Add Vulkan, SDL2, asmjit, and shader
find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)
add_subdirectory(extern/asmjit)

# compile shader into array
set(HLSL "${CMAKE_CURRENT_SOURCE_DIR}/src/rdp.comp.hlsl")
set(SPIRV "${CMAKE_CURRENT_SOURCE_DIR}/src/comp.spv")
add_custom_command(OUTPUT ${SPIRV}
    COMMAND glslangValidator -V -D ${HLSL} -e main --vn comp_code -o ${SPIRV}
    DEPENDS ${HLSL})
add_custom_target(shader DEPENDS ${SPIRV})

# Link nmulator against dependencies
add_executable(nmulator src/main.cpp src/r4300.cpp src/rsp.cpp
    src/rdp.cpp src/scheduler.cpp src/debugger.cpp src/mipsjit.cpp)
target_include_directories(nmulator PRIVATE ${SDL2_INCLUDE_DIRS})
string(STRIP ${SDL2_LIBRARIES} SDL2_LIBRARIES)
target_link_libraries(nmulator PRIVATE Vulkan::Vulkan ${SDL2_LIBRARIES} asmjit)
add_dependencies(nmulator shader)

# set nmulator compiler options
target_compile_options(nmulator PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -msse4.2
        -fno-exceptions -fno-rtti -fno-strict-aliasing>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /EHsc /GR->)
target_compile_features(nmulator PRIVATE cxx_std_20)
