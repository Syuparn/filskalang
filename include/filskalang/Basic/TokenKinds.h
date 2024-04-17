/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_BASIC_TOKENKINDS_H
#define FILSKALANG_BASIC_TOKENKINDS_H

namespace filskalang {
namespace tok {
enum TokenKind : unsigned short {
#define TOK(ID) ID,
#include "./TokenKinds.def"
  NUM_TOKENS
};

const char *getTokenName(TokenKind Kind);
const char *getPunctuatorSpelling(TokenKind Kind);
const char *getKeywordSpelling(TokenKind Kind);
} // namespace tok
} // namespace filskalang

#endif
