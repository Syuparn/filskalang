/*
  Source code in this file is inherited and modified from toy language
  https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy

  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  See https://llvm.org/LICENSE.txt for license information.
  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "filskalang/CodeGen/MLIRGen.h"
#include "filskalang/AST/AST.h"
#include "filskalang/CodeGen/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/Casting.h"

class MLIRGenImpl {

private:
  mlir::ModuleOp Module;
  mlir::OpBuilder Builder;

public:
  MLIRGenImpl(mlir::MLIRContext &Context) : Builder(&Context) {}

  mlir::ModuleOp mlirGen(filskalang::ast::Program &Program) {
    Module = mlir::ModuleOp::create(Builder.getUnknownLoc());

    for (filskalang::ast::Subprogram *Subprogram : Program.getSubprograms()) {
      mlirGen(*Subprogram);
    }

    return Module;
  }

private:
  mlir::filskalang::SubprogramOp
  mlirGen(filskalang::ast::Subprogram &Subprogram) {
    mlir::Location Loc = Subprogram.getLocation().getLocation(Builder);

    // create dummy parameters
    // NOTE: since subprogram does not have parameters, this must be empty
    llvm::SmallVector<mlir::Type, 0> ArgTypes;
    auto funcType = Builder.getFunctionType(ArgTypes, std::nullopt);

    // start subprogram
    Builder.setInsertionPointToEnd(Module.getBody());
    mlir::filskalang::SubprogramOp Sub =
        Builder.create<mlir::filskalang::SubprogramOp>(
            Loc, Subprogram.getName(), funcType);

    // subprogram body
    mlir::Block &EntryBlock = Sub.front();
    Builder.setInsertionPointToStart(&EntryBlock);

    for (filskalang::ast::Instruction *Instruction :
         Subprogram.getInstructions()) {
      switch (Instruction->getKind()) {
      case filskalang::ast::Instruction::IK_Nullary: {
        mlirGenPrt(
            *llvm::cast<filskalang::ast::NullaryInstruction>(Instruction));
        break;
      }
      case filskalang::ast::Instruction::IK_Unary: {
        mlirGenSet(*llvm::cast<filskalang::ast::UnaryInstruction>(Instruction));
        break;
      }
      default: {
        // TODO: impl
      }
      }
    }

    return Sub;
  }

  mlir::filskalang::PrtOp
  mlirGenPrt(filskalang::ast::NullaryInstruction &Instruction) {
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);
    return Builder.create<mlir::filskalang::PrtOp>(Loc);
  }

  mlir::filskalang::SetOp
  mlirGenSet(filskalang::ast::UnaryInstruction &Instruction) {
    auto Type = Builder.getF64Type();
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);
    auto Attr =
        mlir::FloatAttr::get(Type, Instruction.getOperand()->getValue());
    return Builder.create<mlir::filskalang::SetOp>(Loc, Attr);
  }
};

namespace filskalang {
// The public API for codegen
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          filskalang::ast::Program &Program) {
  return MLIRGenImpl(Context).mlirGen(Program);
}
} // namespace filskalang
