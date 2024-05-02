#ifndef FILSKALANG_CODEGEN_LOWERTOLLVM_H
#define FILSKALANG_CODEGEN_LOWERTOLLVM_H

#include <memory>

namespace mlir {
class Pass;

namespace filskalang {
/// lower filskalang to llvm
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace filskalang
} // namespace mlir

#endif
