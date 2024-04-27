#include "filskalang/CodeGen/Dialect.h"

using namespace mlir::filskalang;

#include "filskalang/CodeGen/Dialect.cpp.inc"

#define GET_OP_CLASSES
#include "filskalang/CodeGen/Ops.cpp.inc"

void FilskalangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "filskalang/CodeGen/Ops.cpp.inc"
      >();
}
