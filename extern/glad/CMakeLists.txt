cmake_minimum_required(VERSION 3.9)
project(glad VERSION 2.0.0 LANGUAGES C)

add_library(glad STATIC include/glad/vulkan.h src/vulkan.c)
target_include_directories(glad PUBLIC include)
