#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/TokenKinds.h"
#include "filskalang/Lexer/Token.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <vector>

filskalang::Lexer NewLexer(const char *Src) {
  llvm::SourceMgr SrcMgr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
      llvm::MemoryBuffer::getMemBuffer(Src, "DummyBuffer");
  // NOTE: skip error check because it must not be occurred
  SrcMgr.AddNewSourceBuffer(std::move(*BufferOrErr), llvm::SMLoc());

  filskalang::DiagnosticsEngine Diags(SrcMgr);

  return filskalang::Lexer(SrcMgr, Diags);
}

template <unsigned long N>
void RunTest(const char *Src,
             std::array<filskalang::tok::TokenKind, N> ExpectedKinds) {
  auto Lex = NewLexer(Src);
  filskalang::Token Actual;

  for (auto Expected : ExpectedKinds) {
    Lex.next(Actual);
    EXPECT_EQ(Actual.getKind(), Expected);
  }
}

TEST(LexerTest, BasicAssertions) {
  auto Src = R"(
    { main }
    )";

  std::array<filskalang::tok::TokenKind, 4> Expected = {
      filskalang::tok::l_brace,
      filskalang::tok::identifier,
      filskalang::tok::r_brace,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}
