#ifndef FILSKALANG_CODEGEN_LOWERTOARITH_H
#define FILSKALANG_CODEGEN_LOWERTOARITH_H

#include <memory>

namespace mlir {
class Pass;

namespace filskalang {
/// lower filskalang to arith
std::unique_ptr<mlir::Pass> createLowerToArithPass();
} // namespace filskalang
} // namespace mlir

#endif
