enable_testing()

add_filskalang_tool(Test
  LexerTest.cpp
  )

target_link_libraries(Test
  PRIVATE
  GTest::gtest_main
  filskalangLexer
  filskalangBasic
  )

include(GoogleTest)
gtest_discover_tests(Test)
