/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Sema/Sema.h"
#include "filskalang/AST/AST.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/Location.h"
#include "filskalang/Basic/TokenKinds.h"
#include "llvm/ADT/APFloat.h"

using namespace filskalang;

void Sema::initialize(filskalang::DiagnosticsEngine &) {}

ast::Program *Sema::actOnProgram(Location Loc,
                                 std::vector<ast::Subprogram *> &Subprograms) {
  if (SubprogramNames.find("main") == SubprogramNames.end()) {
    Diags.report(Loc.getLoc(), diag::err_no_main);
  }

  return new ast::Program(Loc, Subprograms);
}

void Sema::actOnSubprogram(Location Loc, llvm::StringRef Name,
                           std::vector<ast::Instruction *> &Instructions,
                           std::vector<ast::Subprogram *> &Subprograms) {
  if (SubprogramNames.find(Name.str()) != SubprogramNames.end()) {
    Diags.report(Loc.getLoc(), diag::err_duplicated_subprogram, Name.str());
  }

  ast::Subprogram *Sub = new ast::Subprogram(Loc, Name, Instructions);
  Subprograms.push_back(Sub);
  SubprogramNames.insert(Name.str());
}

void Sema::actOnNullaryInstruction(
    Location Loc, tok::TokenKind OperatorKind,
    std::vector<ast::Instruction *> &Instructions) {
  // TODO: validate

  ast::NullaryInstruction::NullaryOperator Op;
  switch (OperatorKind) {
  case tok::kw_hlt:
    Op = ast::NullaryInstruction::OP_HLT;
    break;
  case tok::kw_neg:
    Op = ast::NullaryInstruction::OP_NEG;
    break;
  case tok::kw_prt:
    Op = ast::NullaryInstruction::OP_PRT;
    break;
  // TODO: handle other ops
  default:
    Diags.report(Loc.getLoc(), diag::err_unexpected);
    return;
  }

  ast::NullaryInstruction *NI = new ast::NullaryInstruction(Loc, Op);
  Instructions.push_back(NI);
}

void Sema::actOnUnaryInstruction(
    Location Loc, tok::TokenKind OperatorKind, ast::NumberLiteral *&Operand,
    std::vector<ast::Instruction *> &Instructions) {
  // TODO: validate
  ast::UnaryInstruction::UnaryOperator Op;
  switch (OperatorKind) {
  case tok::kw_set:
    Op = ast::UnaryInstruction::OP_SET;
    break;
  // TODO: handle other ops
  default:
    Diags.report(Loc.getLoc(), diag::err_unexpected);
    return;
  }

  ast::UnaryInstruction *UI = new ast::UnaryInstruction(Loc, Op, Operand);
  Instructions.push_back(UI);
}

ast::NumberLiteral *Sema::actOnNumberLiteral(Location Loc,
                                             llvm::StringRef Data) {
  // TODO: parse exponent
  // NOTE: use the same precision as the original implementation
  double Value = std::stod(Data.str());
  return new ast::NumberLiteral(Loc, Value);
}
