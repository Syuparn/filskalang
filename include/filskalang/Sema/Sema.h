/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_SEMA_SEMA_H
#define FILSKALANG_SEMA_SEMA_H

#include "filskalang/AST/AST.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/TokenKinds.h"

namespace filskalang {

class Sema {
  DiagnosticsEngine &Diags;

public:
  Sema(DiagnosticsEngine &Diags) : Diags(Diags) { initialize(Diags); }

  void initialize(filskalang::DiagnosticsEngine &Diags);

  ast::Program *actOnProgram(mlir::SMLoc Loc,
                             std::vector<ast::Subprogram *> &Subprograms);
  // NOTE: result is pushed to Subprograms destructively
  void actOnSubprogram(mlir::SMLoc Loc, llvm::StringRef Name,
                       std::vector<ast::Instruction *> &Instructions,
                       std::vector<ast::Subprogram *> &Subprograms);
  // NOTE: result is pushed to Instructions destructively
  void actOnNullaryInstruction(mlir::SMLoc Loc, tok::TokenKind OperatorKind,
                               std::vector<ast::Instruction *> &Instructions);
  // NOTE: result is pushed to Instructions destructively
  void actOnUnaryInstruction(mlir::SMLoc Loc, tok::TokenKind OperatorKind,
                             ast::NumberLiteral *&Operand,
                             std::vector<ast::Instruction *> &Instructions);
  ast::NumberLiteral *actOnNumberLiteral(mlir::SMLoc Loc, llvm::StringRef Data);
};
} // namespace filskalang

#endif
