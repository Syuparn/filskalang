set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangCodeGen
  Dialect.cpp
  LowerToArith.cpp
  LowerToLLVM.cpp
  MLIRGen.cpp
  Ops.cpp

  LINK_LIBS
  filskalangBasic
  )
