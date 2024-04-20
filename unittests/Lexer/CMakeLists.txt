enable_testing()

add_filskalang_tool(lexer_test
  LexerTest.cpp
  )

target_link_libraries(lexer_test
  PRIVATE
  GTest::gtest_main
  filskalangLexer
  filskalangBasic
  )

include(GoogleTest)
gtest_discover_tests(lexer_test)
