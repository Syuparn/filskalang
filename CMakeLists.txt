cmake_minimum_required(VERSION 3.20.0)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(Filskalang)

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  find_package(MLIR REQUIRED HINTS "${MLIR_CMAKE_PATH}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/include)
  include_directories(${PROJECT_BINARY_DIR}/include)
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  set(FILSKALANG_BUILT_STANDALONE 1)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(FILSKALANG_VERSION_STRING "0.1")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filskalang/Basic/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/filskalang/Basic/Version.inc)

include(AddFilskalang)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

add_subdirectory(lib)
add_subdirectory(tools)
