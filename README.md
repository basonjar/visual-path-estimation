# EECS 442 Final Project

## Requirements

LLVM 16 or another compiler capable of supporting C++20 and some C++23 features.

### macOS

`brew install llvm`

### Ubuntu 20.04 LTS

`sudo cp -v llvm.list /etc/apt/sources.list.d/`
`sudo apt update`
`sudo apt install llvm-16`

## Build

`cmake -Bbuild -GNinja .`
`cmake --build build`
