#include "filskalang/CodeGen/MLIRGen.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

class MLIRGenImpl {

private:
  mlir::ModuleOp Module;
  mlir::OpBuilder builder;

public:
  MLIRGenImpl(mlir::MLIRContext &Context) : builder(&Context) {}

  mlir::ModuleOp mlirGen(filskalang::Program &Program) {
    Module = mlir::ModuleOp::create(builder.getUnknownLoc());
    // TODO: handle program

    return Module;
  }
};

namespace filskalang {
// The public API for codegen
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          filskalang::Program &Program) {
  return MLIRGenImpl(Context).mlirGen(Program);
}
} // namespace filskalang
