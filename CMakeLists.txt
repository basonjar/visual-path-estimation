cmake_minimum_required(VERSION 3.0)
project(EECS442_Final_Project CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
#find_package(PCL REQUIRED COMPONENTS common io)

add_subdirectory(gtsam)

add_executable(program main.cpp)
target_link_options(program PRIVATE -fuse-ld=lld)
target_link_libraries(program gtsam ${OpenCV_LIBS})
target_include_directories(program PRIVATE ${OpenCV_INCLUDE_DIRS})
