/*
  Source code in this file is inherited and modified from toy language
  https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy

  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  See https://llvm.org/LICENSE.txt for license information.
  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "filskalang/CodeGen/Dialect.h"

void mlir::filskalang::SubprogramOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
    mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs) {

  // FunctionOpInterface provides a convenient `build` method that will
  // populate the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

void mlir::filskalang::ProgramOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs) {

  // FunctionOpInterface provides a convenient `build` method that will
  // populate the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, llvm::StringRef("program"), type, attrs,
                      type.getInputs());
}
