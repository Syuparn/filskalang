/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_AST_AST_H
#define FILSKALANG_AST_AST_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <vector>

class NumberLiteral {
  llvm::SMLoc Loc;
  llvm::APFloat Value;

public:
  NumberLiteral(llvm::SMLoc Loc, const llvm::APFloat &Value)
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
  llvm::SMLoc Loc;

private:
  const InstructionKind Kind;

protected:
  Instruction(llvm::SMLoc Loc, InstructionKind Kind) : Loc(Loc), Kind(Kind) {}

public:
  InstructionKind getKind() const { return Kind; }
  llvm::SMLoc getLocation() { return Loc; }
};

class NullaryInstruction : public Instruction {
public:
  enum NullaryOperator {
    OP_PRT,
  };

private:
  NullaryOperator Operator;

public:
  NullaryInstruction(llvm::SMLoc Loc, NullaryOperator Operator)
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
  UnaryInstruction(llvm::SMLoc Loc, UnaryOperator Operator,
                   NumberLiteral *&Operand)
      : Instruction(Loc, IK_Unary), Operator(Operator), Operand(Operand) {}

  static bool classof(const Instruction *I) { return I->getKind() == IK_Unary; }
  UnaryOperator getOperator() { return Operator; }
  NumberLiteral *&getOperand() { return Operand; }
};

class Subprogram {
private:
  llvm::SMLoc Loc;
  llvm::StringRef Name;
  std::vector<Instruction *> Instructions;

public:
  Subprogram(llvm::SMLoc Loc, llvm::StringRef Name,
             std::vector<Instruction *> Instructions)
      : Loc(Loc), Name(Name), Instructions(Instructions) {}

  const std::vector<Instruction *> &getInstructions() { return Instructions; }
  llvm::SMLoc getLocation() { return Loc; }
  llvm::StringRef getName() { return Name; }
};

class Program {
private:
  llvm::SMLoc Loc;
  std::vector<Subprogram *> Subprograms;

  // TODO: separate main from the others

public:
  Program(llvm::SMLoc Loc, std::vector<Subprogram *> Subprograms)
      : Loc(Loc), Subprograms(Subprograms) {}

  const std::vector<Subprogram *> &getSubprograms() { return Subprograms; }
  llvm::SMLoc getLocation() { return Loc; }
};

#endif
