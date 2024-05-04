#include "filskalang/Parser/Parser.h"
#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Lexer/Token.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-port.h"

void RunTest(const char *Src, const char *Expected,
             const char *ExpectedErr = "") {
  llvm::SourceMgr SrcMgr;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrErr =
      llvm::MemoryBuffer::getMemBuffer(Src, "DummyBuffer");
  // NOTE: skip error check because it must not be occurred
  SrcMgr.AddNewSourceBuffer(std::move(*BufferOrErr), mlir::SMLoc());

  filskalang::DiagnosticsEngine Diags(SrcMgr);

  auto Lex = filskalang::Lexer(SrcMgr, Diags, "dummy.filska");

  auto Sem = filskalang::Sema(Diags);
  auto Parser = filskalang::Parser(Lex, Sem);

  testing::internal::CaptureStderr();
  auto AST = Parser.parse();
  // check error message
  ASSERT_STREQ(ExpectedErr, testing::internal::GetCapturedStderr().c_str());

  // if error occurred, skip the latter comparison
  if (strcmp(ExpectedErr, "") != 0) {
    return;
  }

  auto Actual = AST->str();
  ASSERT_STREQ(Actual.c_str(), Expected);
}

TEST(ParserTest, Simple) {
  auto Src = R"(
    { main }
    )";

  auto Expected = "(main ) ";

  RunTest(Src, Expected);
}

// nullary instructions

TEST(ParserTest, Prt) {
  auto Src = R"(
    { main
      prt
    }
    )";

  auto Expected = "(main (OP_PRT) ) ";

  RunTest(Src, Expected);
}

TEST(ParserTest, Hlt) {
  auto Src = R"(
    { main
      hlt
    }
    )";

  auto Expected = "(main (OP_HLT) ) ";

  RunTest(Src, Expected);
}

// unary instructions

TEST(ParserTest, Set) {
  auto Src = R"(
    { main
      set,10
    }
    )";

  auto Expected = "(main (OP_SET 10.000000) ) ";

  RunTest(Src, Expected);
}

// multiple subprograms

TEST(ParserTest, MultipleSubprograms) {
  // TODO: use jmp
  auto Src = R"(
    { main
      set,10
    }
    { another
      prt
    }
    )";

  auto Expected = "(main (OP_SET 10.000000) ) (another (OP_PRT) ) ";

  RunTest(Src, Expected);
}
