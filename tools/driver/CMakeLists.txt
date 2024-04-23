# TODO: set deps

add_filskalang_tool(filskalang
  Driver.cpp
  )

# set target_link_libraries
target_link_libraries(filskalang
  PRIVATE
  filskalangBasic
  filskalangLexer
  filskalangParser
  filskalangSema
  filskalangCodeGen
  )
