set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangSema
  Sema.cpp

  LINK_LIBS
  filskalangBasic
  )
