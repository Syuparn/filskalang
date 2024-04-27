/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_PARSER_PARSER_H
#define FILSKALANG_PARSER_PARSER_H

#include "filskalang/AST/AST.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Sema/Sema.h"
#include "llvm/ADT/StringRef.h"

namespace filskalang {

class Parser {
  Lexer &Lex;
  Token Tok;
  Sema &Sem;

  DiagnosticsEngine &getDiagnostics() const { return Lex.getDiagnostics(); }

  void advance() { Lex.next(Tok); }

  bool expect(tok::TokenKind ExpectedTok) {
    if (Tok.is(ExpectedTok)) {
      return false;
    }
    // There must be a better way!
    const char *Expected = tok::getPunctuatorSpelling(ExpectedTok);
    if (!Expected)
      Expected = tok::getKeywordSpelling(ExpectedTok);
    llvm::StringRef Actual(Tok.getLocation().getPointer(), Tok.getLength());
    getDiagnostics().report(Tok.getLocation(), diag::err_expected, Expected,
                            Actual);
    return true;
  }

  bool consume(tok::TokenKind ExpectedTok) {
    if (Tok.is(ExpectedTok)) {
      advance();
      return false;
    }
    return true;
  }

  template <typename... Tokens> bool skipUntil(Tokens &&...Toks) {
    while (true) {
      if ((... || Tok.is(Toks)))
        return false;

      if (Tok.is(tok::eof))
        return true;
      advance();
    }
  }

  bool parseProgram(ast::Program *&Program);
  bool parseSubprogram(std::vector<ast::Subprogram *> &Subprograms);
  bool parseInstruction(std::vector<ast::Instruction *> &Instructions);
  bool parseNullaryInstruction(std::vector<ast::Instruction *> &Instructions);
  bool parseUnaryInstruction(std::vector<ast::Instruction *> &Instructions);
  bool parseNumberLiteral(ast::NumberLiteral *&NumberLiteral);

public:
  Parser(Lexer &Lex, Sema &Sem);

  ast::Program *parse();
};
} // namespace filskalang

#endif
