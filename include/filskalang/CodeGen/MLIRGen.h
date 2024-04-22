#ifndef FILSKALANG_CODEGEN_MLIRGEN_H
#define FILSKALANG_CODEGEN_MLIRGEN_H

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace filskalang {
class Program;

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          Program &ModuleAST);
}; // namespace filskalang

#endif
