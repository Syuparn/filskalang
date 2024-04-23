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

  Program *actOnProgram(llvm::SMLoc Loc,
                        std::vector<Subprogram *> &Subprograms);
  // NOTE: result is pushed to Subprograms destructively
  void actOnSubprogram(llvm::SMLoc Loc, llvm::StringRef Name,
                       std::vector<Instruction *> &Instructions,
                       std::vector<Subprogram *> &Subprograms);
  // NOTE: result is pushed to Instructions destructively
  void actOnNullaryInstruction(llvm::SMLoc Loc, tok::TokenKind OperatorKind,
                               std::vector<Instruction *> &Instructions);
  // NOTE: result is pushed to Instructions destructively
  void actOnUnaryInstruction(llvm::SMLoc Loc, tok::TokenKind OperatorKind,
                             NumberLiteral *&Operand,
                             std::vector<Instruction *> &Instructions);
  NumberLiteral *actOnNumberLiteral(llvm::SMLoc Loc, llvm::StringRef Data);
};
} // namespace filskalang

#endif
