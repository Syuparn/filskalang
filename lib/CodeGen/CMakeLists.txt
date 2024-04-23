set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangCodeGen
  Dialect.cpp
  MLIRGen.cpp

  LINK_LIBS
  filskalangBasic
  )
