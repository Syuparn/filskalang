#ifndef FILSKALANG_CODEGEN_MLIRGEN_H
#define FILSKALANG_CODEGEN_MLIRGEN_H

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
                                          ast::Program &ModuleAST);
}; // namespace filskalang

#endif
