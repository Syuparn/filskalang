enable_testing()

add_executable(
  Test
  LexerTest.cpp
)
target_link_libraries(
  Test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(Test)
