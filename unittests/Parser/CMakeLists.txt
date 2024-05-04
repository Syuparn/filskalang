enable_testing()

add_filskalang_tool(parser_test
  ParserTest.cpp
  ParserErrorTest.cpp
  )

target_link_libraries(parser_test
  PRIVATE
  GTest::gtest_main
  filskalangSema
  filskalangParser
  filskalangLexer
  filskalangBasic
  )

include(GoogleTest)
gtest_discover_tests(parser_test)
