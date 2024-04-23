#include "filskalang/CodeGen/Dialect.h"

using namespace filskalang;

#include "filskalang/CodeGen/Dialect.cpp.inc"

void FilskalangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "filskalang/CodeGen/Ops.cpp.inc"
      >();
}
