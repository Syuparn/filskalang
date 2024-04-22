set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangCodeGen
  MLIRGen.cpp

  LINK_LIBS
  filskalangBasic
  )
