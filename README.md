# EECS 442 Final Project

Structure from Motion (SfM)

## Requirements

LLVM 16 or another compiler capable of supporting C++20 and some C++23 features.

OpenCV and PCL are also required.

### macOS

`brew install llvm opencv pcl`

### Ubuntu 20.04 LTS

`sudo cp -v llvm.list /etc/apt/sources.list.d/`

`sudo apt update`

`sudo apt install clang-16 libopencv-dev libpcl-dev`

## Build

`cmake -Bbuild -GNinja .`
`cmake --build build`
