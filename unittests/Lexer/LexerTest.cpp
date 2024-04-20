#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/TokenKinds.h"
#include "filskalang/Lexer/Token.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-port.h"

filskalang::Lexer NewLexer(const char *Src) {
  llvm::SourceMgr SrcMgr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
      llvm::MemoryBuffer::getMemBuffer(Src, "DummyBuffer");
  // NOTE: skip error check because it must not be occurred
  SrcMgr.AddNewSourceBuffer(std::move(*BufferOrErr), llvm::SMLoc());
  std::printf("%d\n", SrcMgr.getNumBuffers());

  filskalang::DiagnosticsEngine Diags(SrcMgr);

  return filskalang::Lexer(SrcMgr, Diags);
}

template <unsigned long N>
void RunTest(const char *Src,
             std::array<filskalang::tok::TokenKind, N> ExpectedKinds,
             const char *ExpectedErr = "") {
  // initialize Lexer
  // (cannot separate to function because segmentation fault occurred in
  // buffers...)
  llvm::SourceMgr SrcMgr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
      llvm::MemoryBuffer::getMemBuffer(Src, "DummyBuffer");
  // NOTE: skip error check because it must not be occurred
  SrcMgr.AddNewSourceBuffer(std::move(*BufferOrErr), llvm::SMLoc());
  filskalang::DiagnosticsEngine Diags(SrcMgr);

  auto Lex = filskalang::Lexer(SrcMgr, Diags);
  filskalang::Token Actual;

  testing::internal::CaptureStderr();
  for (auto Expected : ExpectedKinds) {
    Lex.next(Actual);
    EXPECT_EQ(Actual.getKind(), Expected);
  }

  // check error message
  ASSERT_STREQ(ExpectedErr, testing::internal::GetCapturedStderr().c_str());
}

TEST(LexerTest, Simple) {
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

TEST(LexerTest, Comment) {
  auto Src = R"(
    { main
      ipt
      prt "print it
      tmx
    }
    )";

  std::array<filskalang::tok::TokenKind, 7> Expected = {
      filskalang::tok::l_brace, filskalang::tok::identifier,

      filskalang::tok::kw_ipt,  filskalang::tok::kw_prt,
      filskalang::tok::kw_tmx,

      filskalang::tok::r_brace, filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, UnaryInstruction) {
  auto Src = R"(
    { main
      set,10
    }
    )";

  std::array<filskalang::tok::TokenKind, 7> Expected = {
      filskalang::tok::l_brace,        filskalang::tok::identifier,

      filskalang::tok::kw_set,         filskalang::tok::comma,
      filskalang::tok::number_literal,

      filskalang::tok::r_brace,        filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, JumpInstruction) {
  auto Src = R"(
    { main
      jmp,another
    }
    )";

  std::array<filskalang::tok::TokenKind, 7> Expected = {
      filskalang::tok::l_brace,    filskalang::tok::identifier,

      filskalang::tok::kw_jmp,     filskalang::tok::comma,
      filskalang::tok::identifier,

      filskalang::tok::r_brace,    filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, SwapInstruction) {
  auto Src = R"(
    { main
      swp,xy
    }
    )";

  std::array<filskalang::tok::TokenKind, 7> Expected = {
      filskalang::tok::l_brace,
      filskalang::tok::identifier,

      // NOTE: xy is treated as one identifier and separated by the parser
      filskalang::tok::kw_swp,
      filskalang::tok::comma,
      filskalang::tok::identifier,

      filskalang::tok::r_brace,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, AssignmentInstruction) {
  auto Src = R"(
    { main
      add,m=xy
    }
    )";

  std::array<filskalang::tok::TokenKind, 9> Expected = {
      filskalang::tok::l_brace,
      filskalang::tok::identifier,

      // NOTE: xy is treated as one identifier and separated by the parser
      filskalang::tok::kw_add,
      filskalang::tok::comma,
      filskalang::tok::identifier,
      filskalang::tok::equal,
      filskalang::tok::identifier,

      filskalang::tok::r_brace,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, TestInstruction) {
  auto Src = R"(
    { main
      tst,z,1
    }
    )";

  std::array<filskalang::tok::TokenKind, 9> Expected = {
      filskalang::tok::l_brace,        filskalang::tok::identifier,

      filskalang::tok::kw_tst,         filskalang::tok::comma,
      filskalang::tok::identifier,     filskalang::tok::comma,
      filskalang::tok::number_literal,

      filskalang::tok::r_brace,        filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, MultipleInstructions) {
  auto Src = R"(
    { main
      swp,xy
      tst,z,1
    }
    )";

  std::array<filskalang::tok::TokenKind, 12> Expected = {
      filskalang::tok::l_brace,
      filskalang::tok::identifier,

      // NOTE: xy is treated as one identifier and separated by the parser
      filskalang::tok::kw_swp,
      filskalang::tok::comma,
      filskalang::tok::identifier,

      filskalang::tok::kw_tst,
      filskalang::tok::comma,
      filskalang::tok::identifier,
      filskalang::tok::comma,
      filskalang::tok::number_literal,

      filskalang::tok::r_brace,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

// TODO: number(int, float)

TEST(LexerTest, IntNumberLiteral) {
  auto Src = "1";

  std::array<filskalang::tok::TokenKind, 2> Expected = {
      filskalang::tok::number_literal,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, FloatNumberLiteral) {
  auto Src = "1.23";

  std::array<filskalang::tok::TokenKind, 2> Expected = {
      filskalang::tok::number_literal,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, IntWithExponentNumberLiteral) {
  auto Src = R"(
    1e+04
    1e-04
  )";

  std::array<filskalang::tok::TokenKind, 3> Expected = {
      filskalang::tok::number_literal,
      filskalang::tok::number_literal,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, FloatWithExponentNumberLiteral) {
  auto Src = R"(
    1.23e+04
    1.23e-04
  )";

  std::array<filskalang::tok::TokenKind, 3> Expected = {
      filskalang::tok::number_literal,
      filskalang::tok::number_literal,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, MinusNumberLiteral) {
  auto Src = R"(
    -1
    -1.23
    -1.23e+04
    -1.23e-04
  )";

  std::array<filskalang::tok::TokenKind, 5> Expected = {
      filskalang::tok::number_literal,
      filskalang::tok::number_literal,
      filskalang::tok::number_literal,
      filskalang::tok::number_literal,
      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, InvalidNumberLiteralWithDots) {
  auto Src = "1.2.3";

  std::array<filskalang::tok::TokenKind, 1> Expected = {
      filskalang::tok::number_literal};

  auto Err = "DummyBuffer:1:1: error: number literal must be int, float, or "
             "float with exponent\n1.2.3\n^\n";

  RunTest(Src, Expected, Err);
}

TEST(LexerTest, InvalidNumberLiteralWithEs) {
  auto Src = "1.2ee+01";

  std::array<filskalang::tok::TokenKind, 1> Expected = {
      filskalang::tok::number_literal};

  auto Err = "DummyBuffer:1:1: error: number literal must be int, float, or "
             "float with exponent\n1.2ee+01\n^\n";

  RunTest(Src, Expected, Err);
}

TEST(LexerTest, InvalidNumberLiteralWithoutSign) {
  auto Src = "1.2e01";

  std::array<filskalang::tok::TokenKind, 1> Expected = {
      filskalang::tok::number_literal};

  auto Err = "DummyBuffer:1:1: error: number literal must be int, float, or "
             "float with exponent\n1.2e01\n^\n";

  RunTest(Src, Expected, Err);
}

TEST(LexerTest, InvalidNumberLiteralWithDotInExponent) {
  auto Src = "1.2e+01.2";

  std::array<filskalang::tok::TokenKind, 1> Expected = {
      filskalang::tok::number_literal};

  auto Err = "DummyBuffer:1:1: error: number literal must be int, float, or "
             "float with exponent\n1.2e+01.2\n^\n";

  RunTest(Src, Expected, Err);
}

TEST(LexerTest, InvalidNumberLiteralWithDotInExponentAfterInt) {
  auto Src = "1e+01.2";

  std::array<filskalang::tok::TokenKind, 1> Expected = {
      filskalang::tok::number_literal};

  auto Err = "DummyBuffer:1:1: error: number literal must be int, float, or "
             "float with exponent\n1e+01.2\n^\n";

  RunTest(Src, Expected, Err);
}

TEST(LexerTest, Keywords) {
  // NOTE: for ease of testing, invalid syntax is used here
  auto Src = R"(
    { main
      inc
      dec
      sin
      cos
      tan
      log
      exp
      flr
      cel
      rnd
      neg
      chr
      prt
      ipt
      sqr
      asn
      acs
      atn
      tmx
      tmy
      tmz
      txm
      tym
      tzm
      hlt
      set
      gto
      cmp
      add
      sub
      mul
      div
      mod
      pow
      swp
      jmp
      jpr
    }
    )";

  std::array<filskalang::tok::TokenKind, 41> Expected = {
      filskalang::tok::l_brace, filskalang::tok::identifier,

      filskalang::tok::kw_inc,  filskalang::tok::kw_dec,
      filskalang::tok::kw_sin,  filskalang::tok::kw_cos,
      filskalang::tok::kw_tan,  filskalang::tok::kw_log,
      filskalang::tok::kw_exp,  filskalang::tok::kw_flr,
      filskalang::tok::kw_cel,  filskalang::tok::kw_rnd,
      filskalang::tok::kw_neg,  filskalang::tok::kw_chr,
      filskalang::tok::kw_prt,  filskalang::tok::kw_ipt,
      filskalang::tok::kw_sqr,  filskalang::tok::kw_asn,
      filskalang::tok::kw_acs,  filskalang::tok::kw_atn,
      filskalang::tok::kw_tmx,  filskalang::tok::kw_tmy,
      filskalang::tok::kw_tmz,  filskalang::tok::kw_txm,
      filskalang::tok::kw_tym,  filskalang::tok::kw_tzm,
      filskalang::tok::kw_hlt,  filskalang::tok::kw_set,
      filskalang::tok::kw_gto,  filskalang::tok::kw_cmp,
      filskalang::tok::kw_add,  filskalang::tok::kw_sub,
      filskalang::tok::kw_mul,  filskalang::tok::kw_div,
      filskalang::tok::kw_mod,  filskalang::tok::kw_pow,
      filskalang::tok::kw_swp,  filskalang::tok::kw_jmp,
      filskalang::tok::kw_jpr,

      filskalang::tok::r_brace, filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

TEST(LexerTest, MultipleSubprograms) {
  auto Src = R"(
    { main
      jmp,another
    }

    { another
      hlt
    }
    )";

  std::array<filskalang::tok::TokenKind, 11> Expected = {
      filskalang::tok::l_brace,    filskalang::tok::identifier,

      filskalang::tok::kw_jmp,     filskalang::tok::comma,
      filskalang::tok::identifier,

      filskalang::tok::r_brace,

      filskalang::tok::l_brace,    filskalang::tok::identifier,

      filskalang::tok::kw_hlt,

      filskalang::tok::r_brace,

      filskalang::tok::eof,
  };

  RunTest(Src, Expected);
}

// TODO: unknown
