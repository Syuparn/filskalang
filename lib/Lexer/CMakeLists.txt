set(LLVM_LINK_COMPONENTS support)

add_filskalang_library(filskalangLexer
  Lexer.cpp

  LINK_LIBS
  filskalangBasic
  )
