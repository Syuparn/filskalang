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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/Casting.h"

class MLIRGenImpl {

private:
  mlir::ModuleOp Module;
  const char *InputFileName;
  llvm::SourceMgr &SrcMgr;
  mlir::OpBuilder Builder;

public:
  MLIRGenImpl(mlir::MLIRContext &Context, const char *InputFileName,
              llvm::SourceMgr &SrcMgr)
      : InputFileName(InputFileName), SrcMgr(SrcMgr), Builder(&Context) {}

  mlir::ModuleOp mlirGen(filskalang::ast::Program &Program) {
    Module = mlir::ModuleOp::create(Builder.getUnknownLoc());

    for (filskalang::ast::Subprogram *Subprogram : Program.getSubprograms()) {
      mlirGen(*Subprogram);
    }

    return Module;
  }

private:
  mlir::Location loc(mlir::SMLoc &Loc) {
    auto LineColPair = SrcMgr.getLineAndColumn(Loc, 0);
    return mlir::FileLineColLoc::get(Builder.getStringAttr(InputFileName),
                                     LineColPair.first, LineColPair.second);
  }

  mlir::filskalang::SubprogramOp
  mlirGen(filskalang::ast::Subprogram &Subprogram) {
    mlir::SMLoc SMLoc = Subprogram.getLocation();

    // create dummy parameters
    // NOTE: since subprogram does not have parameters, this must be empty
    llvm::SmallVector<mlir::Type, 0> ArgTypes;
    auto funcType = Builder.getFunctionType(ArgTypes, std::nullopt);

    // start subprogram
    Builder.setInsertionPointToEnd(Module.getBody());
    mlir::filskalang::SubprogramOp Sub =
        Builder.create<mlir::filskalang::SubprogramOp>(
            loc(SMLoc), Subprogram.getName(), funcType);

    // subprogram body
    mlir::Block &EntryBlock = Sub.front();
    Builder.setInsertionPointToStart(&EntryBlock);

    for (filskalang::ast::Instruction *Instruction :
         Subprogram.getInstructions()) {
      switch (Instruction->getKind()) {
      case filskalang::ast::Instruction::IK_Nullary: {
        mlirGen(*llvm::cast<filskalang::ast::NullaryInstruction>(Instruction));
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
  mlirGen(filskalang::ast::NullaryInstruction &Instruction) {
    mlir::SMLoc SMLoc = Instruction.getLocation();
    return Builder.create<mlir::filskalang::PrtOp>(loc(SMLoc));
  }
};

namespace filskalang {
// The public API for codegen
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          const char *InputFileName,
                                          llvm::SourceMgr &SrcMgr,
                                          filskalang::ast::Program &Program) {
  return MLIRGenImpl(Context, InputFileName, SrcMgr).mlirGen(Program);
}
} // namespace filskalang
