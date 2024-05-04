/*
  Source code in this file is inherited and modified from toy language
  https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy

  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  See https://llvm.org/LICENSE.txt for license information.
  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "filskalang/CodeGen/LowerToLLVM.h"
#include "filskalang/CodeGen/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <llvm-18/llvm/ADT/APFloat.h>
#include <memory>
#include <unordered_map>

namespace {
// TODO: move to a member variable in any class
std::unordered_map<std::string, mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>
    SubprogramMemory;

class HltOpLowering : public mlir::ConversionPattern {
public:
  explicit HltOpLowering(mlir::MLIRContext *Context)
      : ConversionPattern(mlir::filskalang::HltOp::getOperationName(), 1,
                          Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op, mlir::ArrayRef<mlir::Value> Operands,
                  mlir::ConversionPatternRewriter &Rewriter) const override {

    // TODO: add returnop

    // Notify the rewriter that this operation has been removed.
    Rewriter.eraseOp(Op);
    return mlir::success();
  }
};

class PrtOpLowering : public mlir::ConversionPattern {
public:
  explicit PrtOpLowering(mlir::MLIRContext *Context)
      : ConversionPattern(mlir::filskalang::PrtOp::getOperationName(), 1,
                          Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op, mlir::ArrayRef<mlir::Value> Operands,
                  mlir::ConversionPatternRewriter &Rewriter) const override {
    auto *Context = Rewriter.getContext();
    auto Loc = Op->getLoc();
    auto PrtOp = mlir::cast<mlir::filskalang::PrtOp>(Op);

    mlir::ModuleOp ParentModule = Op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto PrintfRef = getOrInsertPrintf(Rewriter, ParentModule);
    mlir::Value FormatSpecifierCst = getOrCreateGlobalString(
        Loc, Rewriter, "frmt_spec", mlir::StringRef("%f\0", 3), ParentModule);

    auto MemoryPointer = SubprogramMemory.at(PrtOp.getSubprogramName().str());

    // Generate a call to printf
    auto Load = Rewriter.create<mlir::LLVM::LoadOp>(
        Loc, mlir::Float64Type::get(Context), MemoryPointer);

    Rewriter.create<mlir::LLVM::CallOp>(
        Loc, getPrintfType(Context), PrintfRef,
        mlir::ArrayRef<mlir::Value>({FormatSpecifierCst, Load}));

    // Notify the rewriter that this operation has been removed.
    Rewriter.eraseOp(Op);
    return mlir::success();
  }

private:
  // signature of printf
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *Context) {
    auto LLVMF64Ty = mlir::FloatType::getF64(Context);
    auto LLVMPtrTy = mlir::LLVM::LLVMPointerType::get(Context);
    auto LLVMFnType = mlir::LLVM::LLVMFunctionType::get(LLVMF64Ty, LLVMPtrTy,
                                                        /*isVarArg=*/true);
    return LLVMFnType;
  }

  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &Rewriter, mlir::ModuleOp Module) {
    auto *Context = Module.getContext();
    if (Module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(Context, "printf");
    }

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(Rewriter);
    Rewriter.setInsertionPointToStart(Module.getBody());
    Rewriter.create<mlir::LLVM::LLVMFuncOp>(Module.getLoc(), "printf",
                                            getPrintfType(Context));
    return mlir::SymbolRefAttr::get(Context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static mlir::Value getOrCreateGlobalString(mlir::Location Loc,
                                             mlir::OpBuilder &Builder,
                                             mlir::StringRef Name,
                                             mlir::StringRef Value,
                                             mlir::ModuleOp Module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp Global;
    if (!(Global = Module.lookupSymbol<mlir::LLVM::GlobalOp>(Name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(Builder);
      Builder.setInsertionPointToStart(Module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(Builder.getContext(), 8), Value.size());
      Global = Builder.create<mlir::LLVM::GlobalOp>(
          Loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, Name,
          Builder.getStringAttr(Value),
          /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value GlobalPtr =
        Builder.create<mlir::LLVM::AddressOfOp>(Loc, Global);
    mlir::Value Cst0 = Builder.create<mlir::LLVM::ConstantOp>(
        Loc, Builder.getI64Type(), Builder.getIndexAttr(0));
    return Builder.create<mlir::LLVM::GEPOp>(
        Loc, mlir::LLVM::LLVMPointerType::get(Builder.getContext()),
        Global.getType(), GlobalPtr, mlir::ArrayRef<mlir::Value>({Cst0, Cst0}));
  }
};

class SetOpLowering : public mlir::ConversionPattern {
public:
  explicit SetOpLowering(mlir::MLIRContext *Context)
      : ConversionPattern(mlir::filskalang::SetOp::getOperationName(), 1,
                          Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op, mlir::ArrayRef<mlir::Value> Operands,
                  mlir::ConversionPatternRewriter &Rewriter) const override {
    auto Loc = Op->getLoc();
    auto SetOp = mlir::cast<mlir::filskalang::SetOp>(Op);

    mlir::Value ConstantOp = Rewriter.create<mlir::LLVM::ConstantOp>(
        Loc, Rewriter.getF64Type(),
        Rewriter.getF64FloatAttr(SetOp.getValue().convertToDouble()));

    auto MemoryPointer = SubprogramMemory.at(SetOp.getSubprogramName().str());
    auto Value = llvm::APFloat(SetOp.getValue());
    Rewriter.create<mlir::LLVM::StoreOp>(Loc, ConstantOp, MemoryPointer);

    // Notify the rewriter that this operation has been removed.
    Rewriter.eraseOp(Op);
    return mlir::success();
  }
};

class SubprogramOpLowering : public mlir::ConversionPattern {
public:
  explicit SubprogramOpLowering(mlir::MLIRContext *Context)
      : ConversionPattern(mlir::filskalang::SubprogramOp::getOperationName(), 1,
                          Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op, mlir::ArrayRef<mlir::Value> Operands,
                  mlir::ConversionPatternRewriter &Rewriter) const override {
    auto Loc = Op->getLoc();
    auto *Context = Op->getContext();

    mlir::filskalang::SubprogramOp Subprogram =
        mlir::cast<mlir::filskalang::SubprogramOp>(Op);
    Rewriter.setInsertionPointToEnd(Subprogram->getBlock());

    // HACK: since builtin.module cannot hold multiple blocks, inline
    // subprograms into a function and call it from the module
    auto DummyType = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(Context), mlir::ArrayRef<mlir::Type>{});
    // NOTE: name should be "main" because LLVM IR recognize the name as an
    // entrypoint
    auto Func = Rewriter.create<mlir::LLVM::LLVMFuncOp>(Loc, "main", DummyType);
    Rewriter.inlineRegionBefore(Subprogram.getBody(), Func.getBody(),
                                Func.end());
    mlir::Block *EntryBlock = &Func.getBody().front();

    // define subprogram register `m`
    Rewriter.setInsertionPointToStart(EntryBlock);
    mlir::Value Cst0 = Rewriter.create<mlir::LLVM::ConstantOp>(
        Loc, Rewriter.getI64Type(), Rewriter.getIndexAttr(0));
    auto Alloca = Rewriter.create<mlir::LLVM::AllocaOp>(
        Loc, /*resultType*/ mlir::LLVM::LLVMPointerType::get(Context),
        /*elementType*/ Rewriter.getF64Type(), Cst0);
    // set to SubprogramMemory map (this is used when the subprogram register
    // `m` is referred)
    SubprogramMemory[Subprogram.getName().str()] = Alloca.getRes();

    // add return because llvm block must ends with terminator operator
    // TODO: replace with infinite loop
    Rewriter.setInsertionPointToEnd(EntryBlock);
    Rewriter.create<mlir::LLVM::ReturnOp>(Loc, mlir::ArrayRef<mlir::Value>());

    Rewriter.setInsertionPointToEnd(Subprogram->getBlock());

    // Notify the rewriter that this operation has been removed.
    Rewriter.eraseOp(Op);

    return mlir::success();
  }
};

struct FilskalangToLLVMLoweringPass
    : public mlir::PassWrapper<FilskalangToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FilskalangToLLVMLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void FilskalangToLLVMLoweringPass::runOnOperation() {
  // define final target
  mlir::LLVMConversionTarget Target(getContext());
  Target.addLegalOp<mlir::ModuleOp>();

  // define lowering patterns
  mlir::RewritePatternSet Patterns(&getContext());
  // TODO: add transitive lowering patterns

  // lower from filskalang dialect
  Patterns.add<SubprogramOpLowering>(&getContext());
  Patterns.add<HltOpLowering>(&getContext());
  Patterns.add<PrtOpLowering>(&getContext());
  Patterns.add<SetOpLowering>(&getContext());

  // completely lower to LLVM
  auto Module = getOperation();
  if (failed(applyFullConversion(Module, Target, std::move(Patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::filskalang::createLowerToLLVMPass() {
  return std::make_unique<FilskalangToLLVMLoweringPass>();
}
