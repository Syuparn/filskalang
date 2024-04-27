#ifndef FILSKALANG_CODEGEN_MLIRGEN_H
#define FILSKALANG_CODEGEN_MLIRGEN_H

#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace filskalang {
namespace ast {
class Program;
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          const char *InputFileName,
                                          llvm::SourceMgr &SrcMgr,
                                          ast::Program &ModuleAST);
}; // namespace filskalang

#endif
