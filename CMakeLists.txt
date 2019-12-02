cmake_minimum_required(VERSION 3.9)
project(nmulator VERSION 0.1.0 LANGUAGES CXX)

# Add Vulkan, SDL2, asmjit, and shader
find_package(vulkan REQUIRED)
find_package(SDL2 REQUIRED)
add_subdirectory(extern/asmjit)

# compile shader into array
set(HLSL "${CMAKE_CURRENT_SOURCE_DIR}/src/rdp.comp.hlsl")
set(SPIRV "${CMAKE_CURRENT_SOURCE_DIR}/src/rdp.spv.array")
add_custom_command(OUTPUT ${SPIRV}
    COMMAND glslangValidator -V -D ${HLSL} -e main -o rdp.spv
    COMMAND xxd -i rdp.spv > ${SPIRV}
    DEPENDS ${HLSL})
add_custom_target(shader DEPENDS ${SPIRV})

# Link nmulator against dependencies
add_executable(nmulator src/main.cpp)
target_include_directories(nmulator PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(nmulator PRIVATE Vulkan::Vulkan ${SDL2_LIBRARIES} asmjit)
add_dependencies(nmulator shader)

# set nmulator compiler options
target_compile_options(nmulator
    PRIVATE -Wall -Wextra -march=x86-64 -mavx -mbmi2
    PRIVATE -fno-exceptions -fno-rtti -fno-strict-aliasing)
target_compile_features(nmulator PRIVATE cxx_std_17)

# clean up generated files
add_custom_command(TARGET nmulator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove rdp.spv ${SPIRV})
