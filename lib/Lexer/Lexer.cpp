/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Basic/Diagnostic.h"
#include <cstdio>

using namespace filskalang;

void KeywordFilter::addKeyword(llvm::StringRef Keyword,
                               tok::TokenKind TokenCode) {
  HashTable.insert(std::make_pair(Keyword, TokenCode));
}

void KeywordFilter::addKeywords() {
#define KEYWORD(NAME, FLAGS) addKeyword(llvm::StringRef(#NAME), tok::kw_##NAME);
#include "filskalang/Basic/TokenKinds.def"
}

namespace charinfo {
LLVM_READNONE inline bool isASCII(char Ch) {
  return static_cast<unsigned char>(Ch) <= 127;
}

LLVM_READNONE inline bool isVerticalWhitespace(char Ch) {
  return isASCII(Ch) && (Ch == '\r' || Ch == '\n');
}

LLVM_READNONE inline bool isHorizontalWhitespace(char Ch) {
  return isASCII(Ch) && (Ch == ' ' || Ch == '\t' || Ch == '\f' || Ch == '\v');
}

LLVM_READNONE inline bool isWhitespace(char Ch) {
  return isHorizontalWhitespace(Ch) || isVerticalWhitespace(Ch);
}

LLVM_READNONE inline bool isDigit(char Ch) {
  return isASCII(Ch) && Ch >= '0' && Ch <= '9';
}

LLVM_READNONE inline bool isIdentifierHead(char Ch) {
  return isASCII(Ch) &&
         (Ch == '_' || (Ch >= 'A' && Ch <= 'Z') || (Ch >= 'a' && Ch <= 'z'));
}

LLVM_READNONE inline bool isIdentifierBody(char Ch) {
  return isIdentifierHead(Ch) || isDigit(Ch);
}
} // namespace charinfo

void Lexer::next(Token &Result) {
  while (*CurPtr && charinfo::isWhitespace(*CurPtr)) {
    ++CurPtr;
  }
  if (!*CurPtr) {
    Result.setKind(tok::eof);
    return;
  }
  if (charinfo::isIdentifierHead(*CurPtr)) {
    identifier(Result);
    return;
  }
  if (charinfo::isDigit(*CurPtr) || *CurPtr == '-') {
    number(Result);
    return;
  }
  switch (*CurPtr) {
#define CASE(ch, tok)                                                          \
  case ch:                                                                     \
    formToken(Result, CurPtr + 1, tok);                                        \
    break
    CASE('=', tok::equal);
    CASE(',', tok::comma);
    CASE('{', tok::l_brace);
    CASE('}', tok::r_brace);
#undef CASE
  case '"':
    comment();
    next(Result);
    break;
  default:
    Diags.report(getLoc(), diag::err_unknown_token);
    formToken(Result, CurPtr + 1, tok::unknown);
  }
  return;
}

void Lexer::identifier(Token &Result) {
  const char *Start = CurPtr;
  const char *End = CurPtr + 1;
  while (charinfo::isIdentifierBody(*End))
    ++End;
  llvm::StringRef Name(Start, End - Start);
  formToken(Result, End, Keywords.getKeyword(Name, tok::identifier));
}

void Lexer::number(Token &Result) {
  const char *End = CurPtr + 1;
  bool IsFloat = false;
  bool IsExponent = false;

  // peek following characters after the first one
  // NOTE: we don't have to check signs because it only appears at first
  for (; *End; End++) {
    if (charinfo::isDigit(*End)) {
      continue;
    }

    if (*End == '.') {
      if (!IsFloat && !IsExponent) {
        IsFloat = true;
        continue;
      }
      Diags.report(getLoc(), diag::err_invalid_number_token);
    }

    if (*End == 'e') {
      if (!IsExponent) {
        IsExponent = true;
        End++;
        if (*End == '+' || *End == '-') {
          continue;
        }
      }
      Diags.report(getLoc(), diag::err_invalid_number_token);
    }

    // number ends
    break;
  }
  formToken(Result, End, tok::number_literal);
}

void Lexer::comment() {
  const char *End = CurPtr + 1;
  while (*End) {
    if (charinfo::isVerticalWhitespace(*End)) {
      break;
    }
    End++;
  }
  CurPtr = End;
}

void Lexer::formToken(Token &Result, const char *TokEnd, tok::TokenKind Kind) {
  size_t TokLen = TokEnd - CurPtr;
  Result.Ptr = CurPtr;
  ;
  Result.Length = TokLen;
  Result.Kind = Kind;
  CurPtr = TokEnd;
}
