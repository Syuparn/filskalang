/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_AST_AST_H
#define FILSKALANG_AST_AST_H

#include "deps/magic_enum.hpp"
#include "filskalang/Basic/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace filskalang {
namespace ast {
class NumberLiteral {
  Location Loc;
  double Value;

public:
  NumberLiteral(Location Loc, double Value) : Loc(Loc), Value(Value) {}
  double getValue() { return Value; }

  std::string str() {
    char s[10];
    std::sprintf(s, "%f", Value);
    return s;
  }
};

class Instruction {
public:
  enum InstructionKind {
    IK_Nullary,
    IK_Unary,
  };

protected:
  Location Loc;

private:
  const InstructionKind Kind;

protected:
  Instruction(Location Loc, InstructionKind Kind) : Loc(Loc), Kind(Kind) {}
  virtual ~Instruction() {}

public:
  InstructionKind getKind() const { return Kind; }
  Location getLocation() { return Loc; }

  virtual std::string str() = 0;
};

class NullaryInstruction : public Instruction {
public:
  enum NullaryOperator {
    OP_PRT,
  };

private:
  NullaryOperator Operator;

public:
  NullaryInstruction(Location Loc, NullaryOperator Operator)
      : Instruction(Loc, IK_Nullary), Operator(Operator) {}
  virtual ~NullaryInstruction() {}

  static bool classof(const Instruction *I) {
    return I->getKind() == IK_Nullary;
  }
  NullaryOperator getOperator() { return Operator; }

  std::string str() override {
    std::stringstream ss;
    ss << "(" << magic_enum::enum_name(Operator) << ")";
    return ss.str();
  }
};

class UnaryInstruction : public Instruction {
public:
  enum UnaryOperator {
    OP_SET,
  };

private:
  UnaryOperator Operator;
  NumberLiteral *Operand;

public:
  UnaryInstruction(Location Loc, UnaryOperator Operator,
                   NumberLiteral *&Operand)
      : Instruction(Loc, IK_Unary), Operator(Operator), Operand(Operand) {}
  virtual ~UnaryInstruction() {}

  static bool classof(const Instruction *I) { return I->getKind() == IK_Unary; }
  UnaryOperator getOperator() { return Operator; }
  NumberLiteral *&getOperand() { return Operand; }

  std::string str() override {
    std::stringstream ss;
    ss << "(" << magic_enum::enum_name(Operator) << " " << Operand->str()
       << ")";
    return ss.str();
  }
};

class Subprogram {
private:
  Location Loc;
  llvm::StringRef Name;
  std::vector<Instruction *> Instructions;

public:
  Subprogram(Location Loc, llvm::StringRef Name,
             std::vector<Instruction *> Instructions)
      : Loc(Loc), Name(Name), Instructions(Instructions) {}

  const std::vector<Instruction *> &getInstructions() { return Instructions; }
  Location getLocation() { return Loc; }
  llvm::StringRef getName() { return Name; }

  std::string str() {
    std::stringstream ss;
    ss << "(" << Name.str() << " ";
    for (auto Instruction : Instructions) {
      ss << Instruction->str() << " ";
    }
    ss << ") ";

    return ss.str();
  }
};

class Program {
private:
  Location Loc;
  std::vector<Subprogram *> Subprograms;

  // TODO: separate main from the others

public:
  Program(Location Loc, std::vector<Subprogram *> Subprograms)
      : Loc(Loc), Subprograms(Subprograms) {}

  const std::vector<Subprogram *> &getSubprograms() { return Subprograms; }
  Location getLocation() { return Loc; }

  std::string str() {
    std::stringstream ss;
    for (auto Subprogram : Subprograms) {
      ss << Subprogram->str();
    }
    return ss.str();
  }
};
} // namespace ast
} // namespace filskalang

#endif
