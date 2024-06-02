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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

class MLIRGenImpl {

private:
  mlir::ModuleOp Module;
  mlir::OpBuilder Builder;

public:
  MLIRGenImpl(mlir::MLIRContext &Context) : Builder(&Context) {}

  mlir::ModuleOp mlirGen(filskalang::ast::Program &Program) {
    mlir::Location Loc = Program.getLocation().getLocation(Builder);
    Module = mlir::ModuleOp::create(Loc);

    // create dummy parameters
    // NOTE: since subprogram does not have parameters, this must be empty
    llvm::SmallVector<mlir::Type, 0> ArgTypes;
    auto funcType = Builder.getFunctionType(ArgTypes, std::nullopt);

    // start subprogram
    Builder.setInsertionPointToEnd(Module.getBody());
    mlir::filskalang::ProgramOp ProgramOp =
        Builder.create<mlir::filskalang::ProgramOp>(Loc, funcType);

    // program body
    mlir::Block &EntryBlock = ProgramOp.front();
    // HACK: add dummy terminator at the end to satisfy the terminator
    // constraint in a mlir block; the block must have one and only one
    // terminator operator at the end.
    Builder.setInsertionPointToStart(&EntryBlock);
    Builder.create<mlir::filskalang::DummyTerminatorOp>(Loc);

    for (filskalang::ast::Subprogram *Subprogram : Program.getSubprograms()) {
      mlirGen(*Subprogram);
    }

    return Module;
  }

private:
  void mlirGen(filskalang::ast::Subprogram &Subprogram) {
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
        auto Nullary =
            llvm::cast<filskalang::ast::NullaryInstruction>(Instruction);
        switch (Nullary->getOperator()) {
        case filskalang::ast::NullaryInstruction::OP_HLT: {
          mlirGenHlt(*Nullary);
          break;
        }
        case filskalang::ast::NullaryInstruction::OP_NEG: {
          mlirGenNeg(*Nullary, Subprogram.getName());
          break;
        }
        case filskalang::ast::NullaryInstruction::OP_PRT: {
          mlirGenPrt(*Nullary, Subprogram.getName());
          break;
        }
        }
        break;
      }
      case filskalang::ast::Instruction::IK_Unary: {
        mlirGenSet(*llvm::cast<filskalang::ast::UnaryInstruction>(Instruction),
                   Subprogram.getName());
        break;
      }
      default: {
        // TODO: impl
      }
      }
    }

    // HACK: add dummy terminator at the end to satisfy the terminator
    // constraint in a mlir block; the block must have one and only one
    // terminator operator at the end.
    Builder.create<mlir::filskalang::DummyTerminatorOp>(Loc);
  }

  void mlirGenHlt(filskalang::ast::NullaryInstruction &Instruction) {
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);
    Builder.create<mlir::filskalang::HltOp>(Loc);
  }

  void mlirGenNeg(filskalang::ast::NullaryInstruction &Instruction,
                  llvm::StringRef SubprogramName) {
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);

    auto Memory = Builder.create<mlir::filskalang::RegisterOp>(
        Loc, Builder.getF64Type(), Builder.getStringAttr(SubprogramName));

    auto NegOp = Builder.create<mlir::filskalang::NegOp>(
        Loc, Builder.getF64Type(), Memory);
    // add setOp to store the result to the register `m` again
    Builder.create<mlir::filskalang::MetaSetOp>(
        Loc, NegOp, Builder.getStringAttr(SubprogramName));
  }

  void mlirGenPrt(filskalang::ast::NullaryInstruction &Instruction,
                  llvm::StringRef SubprogramName) {
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);

    auto Memory = Builder.create<mlir::filskalang::RegisterOp>(
        Loc, Builder.getF64Type(), Builder.getStringAttr(SubprogramName));

    Builder.create<mlir::filskalang::PrtOp>(Loc, Memory);
  }

  void mlirGenSet(filskalang::ast::UnaryInstruction &Instruction,
                  llvm::StringRef SubprogramName) {
    auto Type = Builder.getF64Type();
    mlir::Location Loc = Instruction.getLocation().getLocation(Builder);
    auto Attr =
        mlir::FloatAttr::get(Type, Instruction.getOperand()->getValue());
    Builder.create<mlir::filskalang::SetOp>(
        Loc, Attr, Builder.getStringAttr(SubprogramName));
  }
};

namespace filskalang {
// The public API for codegen
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          filskalang::ast::Program &Program) {
  return MLIRGenImpl(Context).mlirGen(Program);
}
} // namespace filskalang
