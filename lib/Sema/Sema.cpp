/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Sema/Sema.h"
#include "filskalang/AST/AST.h"
#include "filskalang/Basic/TokenKinds.h"
#include <llvm/ADT/APFloat.h>

using namespace filskalang;

void Sema::initialize() {}

Program *Sema::actOnProgram(llvm::SMLoc Loc,
                            std::vector<Subprogram *> &Subprograms) {
  // TODO: validate

  return new Program(Loc, Subprograms);
}

void Sema::actOnSubprogram(llvm::SMLoc Loc, llvm::StringRef Name,
                           std::vector<Instruction *> &Instructions,
                           std::vector<Subprogram *> &Subprograms) {
  // TODO: validate
  Subprogram *Sub = new Subprogram(Loc, Name, Instructions);
  Subprograms.push_back(Sub);
}

void Sema::actOnNullaryInstruction(llvm::SMLoc Loc, tok::TokenKind OperatorKind,
                                   std::vector<Instruction *> &Instructions) {
  // TODO: validate

  NullaryInstruction::NullaryOperator Op;
  switch (OperatorKind) {
  case tok::kw_prt:
    Op = NullaryInstruction::OP_PRT;
    break;
  // TODO: handle other ops
  default:
    Diags.report(Loc, diag::err_unexpected);
    return;
  }

  NullaryInstruction *NI = new NullaryInstruction(Loc, Op);
  Instructions.push_back(NI);
}

void Sema::actOnUnaryInstruction(llvm::SMLoc Loc, tok::TokenKind OperatorKind,
                                 NumberLiteral *&Operand,
                                 std::vector<Instruction *> &Instructions) {
  // TODO: validate
  UnaryInstruction::UnaryOperator Op;
  switch (OperatorKind) {
  case tok::kw_set:
    Op = UnaryInstruction::OP_SET;
    break;
  // TODO: handle other ops
  default:
    Diags.report(Loc, diag::err_unexpected);
    return;
  }

  UnaryInstruction *UI = new UnaryInstruction(Loc, Op, Operand);
  Instructions.push_back(UI);
}

NumberLiteral *Sema::actOnNumberLiteral(llvm::SMLoc Loc, llvm::StringRef Data) {
  // TODO: parse exponent
  // NOTE: use the same precision as the original implementation
  llvm::APFloat Value(llvm::APFloat::IEEEdouble(), Data);

  return new NumberLiteral(Loc, Value);
}
