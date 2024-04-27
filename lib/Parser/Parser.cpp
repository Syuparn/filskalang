/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Parser/Parser.h"
#include "filskalang/AST/AST.h"
#include "filskalang/Basic/TokenKinds.h"

using namespace filskalang;

Parser::Parser(Lexer &Lex, Sema &Sem) : Lex(Lex), Sem(Sem) {}

ast::Program *Parser::parse() {
  ast::Program *Program;
  parseProgram(Program);
  return Program;
}

bool Parser::parseProgram(ast::Program *&Program) {
  auto _errorhandler = [this] { return skipUntil(); };
  std::vector<ast::Subprogram *> Subprograms;

  advance();
  while (Tok.is(tok::l_brace)) {
    if (!parseSubprogram(Subprograms)) {
      return _errorhandler();
    }
  }
  // TODO: error handling if !Tok.is(tok::l_brace)

  Program = Sem.actOnProgram(Tok.getLocation(), Subprograms);

  return false;
}

bool Parser::parseSubprogram(std::vector<ast::Subprogram *> &Subprograms) {
  auto _errorhandler = [this] { return skipUntil(tok::r_brace); };

  if (consume(tok::l_brace)) {
    return _errorhandler();
  }

  if (expect(tok::identifier)) {
    return _errorhandler();
  }
  mlir::SMLoc Loc = Tok.getLocation();
  llvm::StringRef Name = Tok.getIdentifier();

  std::vector<ast::Instruction *> Instructions;
  while (Tok.is(tok::l_brace)) {
    if (parseInstruction(Instructions)) {
      return _errorhandler();
    }
  }

  if (consume(tok::r_brace)) {
    return _errorhandler();
  }

  Sem.actOnSubprogram(Loc, Name, Instructions, Subprograms);

  advance();

  return false;
}

bool Parser::parseInstruction(std::vector<ast::Instruction *> &Instructions) {
  // TODO: skip until all operators
  auto _errorhandler = [this] { return skipUntil(tok::kw_prt, tok::kw_set); };

  advance();
  if (Tok.isOneOf(tok::kw_prt)) {
    if (parseNullaryInstruction(Instructions)) {
      return _errorhandler();
    };
  }

  if (Tok.isOneOf(tok::kw_set)) {
    if (parseUnaryInstruction(Instructions)) {
      return _errorhandler();
    };
  }

  return false;
}

bool Parser::parseNullaryInstruction(
    std::vector<ast::Instruction *> &Instructions) {
  mlir::SMLoc Loc = Tok.getLocation();
  tok::TokenKind OperatorKind = Tok.getKind();

  Sem.actOnNullaryInstruction(Loc, OperatorKind, Instructions);

  return false;
}

bool Parser::parseUnaryInstruction(
    std::vector<ast::Instruction *> &Instructions) {
  mlir::SMLoc Loc = Tok.getLocation();
  tok::TokenKind OperatorKind = Tok.getKind();

  if (consume(tok::comma)) {
    // errHandler is called upstream
    return true;
  }

  ast::NumberLiteral *Number;
  if (parseNumberLiteral(Number)) {
    // errHandler is called upstream
    return true;
  }

  Sem.actOnUnaryInstruction(Loc, OperatorKind, Number, Instructions);

  return false;
}

bool Parser::parseNumberLiteral(ast::NumberLiteral *&Number) {
  if (expect(tok::number_literal)) {
    // errHandler is called upstream
    return true;
  }
  mlir::SMLoc Loc = Tok.getLocation();
  llvm::StringRef Data = Tok.getLiteralData();

  Number = Sem.actOnNumberLiteral(Loc, Data);

  return false;
}
