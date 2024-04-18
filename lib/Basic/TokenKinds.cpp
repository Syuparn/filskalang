/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Basic/TokenKinds.h"
#include "llvm/Support/ErrorHandling.h"

using namespace filskalang;

static const char *const TokNames[] = {
#define TOK(ID) #ID,
#define KEYWORD(ID, FLAG) #ID,
#include "filskalang/Basic/TokenKinds.def"
    nullptr};

const char *tok::getTokenName(TokenKind Kind) {
  if (Kind < tok::NUM_TOKENS) {
    return TokNames[Kind];
  }

  llvm_unreachable("unknown TokenKind");
  return nullptr;
}

const char *tok::getPunctuatorSpelling(TokenKind Kind) {
  switch (Kind) {
#define PUNCTUATOR(ID, SP)                                                     \
  case ID:                                                                     \
    return SP;
#include "filskalang/Basic/TokenKinds.def"
  default:
    break;
  }
  // never reached
  return nullptr;
}

const char *tok::getKeywordSpelling(TokenKind Kind) {
  switch (Kind) {
#define KEYWORD(ID, FLAG)                                                      \
  case kw_##ID:                                                                \
    return #ID;
#include "filskalang/Basic/TokenKinds.def"
  default:
    break;
  }
  // never reached
  return nullptr;
}
