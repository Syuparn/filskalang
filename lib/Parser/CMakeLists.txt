set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangParser
  Parser.cpp

  LINK_LIBS
  filskalangBasic
  filskalangLexer
  filskalangSema
  )
