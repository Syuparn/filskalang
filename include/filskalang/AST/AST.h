/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_AST_AST_H
#define FILSKALANG_AST_AST_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <vector>

namespace filskalang {
namespace ast {
class NumberLiteral {
  mlir::SMLoc Loc;
  llvm::APFloat Value;

public:
  NumberLiteral(mlir::SMLoc Loc, const llvm::APFloat &Value)
      : Loc(Loc), Value(Value) {}
  llvm::APFloat &getValue() { return Value; }
};

class Instruction {
public:
  enum InstructionKind {
    IK_Nullary,
    IK_Unary,
  };

protected:
  mlir::SMLoc Loc;

private:
  const InstructionKind Kind;

protected:
  Instruction(mlir::SMLoc Loc, InstructionKind Kind) : Loc(Loc), Kind(Kind) {}

public:
  InstructionKind getKind() const { return Kind; }
  mlir::SMLoc getLocation() { return Loc; }
};

class NullaryInstruction : public Instruction {
public:
  enum NullaryOperator {
    OP_PRT,
  };

private:
  NullaryOperator Operator;

public:
  NullaryInstruction(mlir::SMLoc Loc, NullaryOperator Operator)
      : Instruction(Loc, IK_Nullary), Operator(Operator) {}

  static bool classof(const Instruction *I) {
    return I->getKind() == IK_Nullary;
  }
  NullaryOperator getOperator() { return Operator; }
};

class UnaryInstruction : public Instruction {
public:
  enum UnaryOperator {
    OP_SET,
  };

private:
  UnaryOperator Operator;
  NumberLiteral *&Operand;

public:
  UnaryInstruction(mlir::SMLoc Loc, UnaryOperator Operator,
                   NumberLiteral *&Operand)
      : Instruction(Loc, IK_Unary), Operator(Operator), Operand(Operand) {}

  static bool classof(const Instruction *I) { return I->getKind() == IK_Unary; }
  UnaryOperator getOperator() { return Operator; }
  NumberLiteral *&getOperand() { return Operand; }
};

class Subprogram {
private:
  mlir::SMLoc Loc;
  llvm::StringRef Name;
  std::vector<Instruction *> Instructions;

public:
  Subprogram(mlir::SMLoc Loc, llvm::StringRef Name,
             std::vector<Instruction *> Instructions)
      : Loc(Loc), Name(Name), Instructions(Instructions) {}

  const std::vector<Instruction *> &getInstructions() { return Instructions; }
  mlir::SMLoc getLocation() { return Loc; }
  llvm::StringRef getName() { return Name; }
};

class Program {
private:
  mlir::SMLoc Loc;
  std::vector<Subprogram *> Subprograms;

  // TODO: separate main from the others

public:
  Program(mlir::SMLoc Loc, std::vector<Subprogram *> Subprograms)
      : Loc(Loc), Subprograms(Subprograms) {}

  const std::vector<Subprogram *> &getSubprograms() { return Subprograms; }
  mlir::SMLoc getLocation() { return Loc; }
};
} // namespace ast
} // namespace filskalang

#endif
