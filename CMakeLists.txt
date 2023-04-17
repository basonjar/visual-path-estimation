cmake_minimum_required(VERSION 3.0)
project(EECS442_Final_Project C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
add_library(JsonCpp::JsonCpp INTERFACE IMPORTED)
set_target_properties(
        JsonCpp::JsonCpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${JSONCPP_LIBRARIES}"
)
#find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED PATHS "/opt/homebrew/Cellar/qt@5/5.15.8_2/lib/cmake")

add_subdirectory(gtsam SYSTEM)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common visualization)

add_executable(program main.cpp)
target_precompile_headers(program PRIVATE pch.h)
target_link_options(program PRIVATE -fuse-ld=lld)
target_link_libraries(program PRIVATE gtsam ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_include_directories(program PRIVATE SYSTEM ${GTSAM_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
