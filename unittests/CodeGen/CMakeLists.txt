enable_testing()

add_filskalang_tool(codegen_test
  CodeGenTest.cpp
  )

target_link_libraries(codegen_test
  PRIVATE
  GTest::gtest_main
  filskalangCodeGen
  filskalangParser
  filskalangSema
  filskalangLexer
  filskalangBasic
  )

include(GoogleTest)
gtest_discover_tests(codegen_test)
