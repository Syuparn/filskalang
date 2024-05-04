#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Lexer/Token.h"
#include "filskalang/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-port.h"

void RunTest(const char *Src, const char *ExpectedErr) {
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
  Parser.parse();
  // check error message
  ASSERT_STREQ(ExpectedErr, testing::internal::GetCapturedStderr().c_str());
}

TEST(SemaTest, NoMain) {
  // `sub` is instruction keyword
  auto Src = R"(
    { another }
    )";

  auto ExpectedErr =
      "DummyBuffer:3:5: error: subprogram main does not exist\n    \n    ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, SubprogramNameIsDuplicated) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main }
    { sub1 }
    { sub1 }
    )";

  auto ExpectedErr = "DummyBuffer:4:11: error: subprogram sub1 already "
                     "exists\n    { sub1 }\n          ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, SubprogramNameIsKeyword) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main }
    { sub }
    )";

  auto ExpectedErr = "DummyBuffer:3:7: error: expected  but found sub\n    { "
                     "sub }\n      ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, PrtArgumentArityIsMore) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main
      prt,1
    }
    )";

  auto ExpectedErr = "DummyBuffer:3:10: error: instruction kw_prt requires 0 "
                     "operands\n      prt,1\n         ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, SetArgumentArityIsLess) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main
      set
    }
    )";

  auto ExpectedErr = "DummyBuffer:4:5: error: instruction kw_set requires 1 "
                     "operands\n    }\n    ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, SetArgumentArityIsMore) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main
      set,1,2
    }
    )";

  auto ExpectedErr = "DummyBuffer:3:12: error: instruction kw_set requires 1 "
                     "operands\n      set,1,2\n           ^\n";

  RunTest(Src, ExpectedErr);
}

TEST(SemaTest, SetArgumentIsNotNumber) {
  // `sub` is instruction keyword
  auto Src = R"(
    { main
      set,foo
    }
    )";

  auto ExpectedErr = "DummyBuffer:3:11: error: argument must be a number\n     "
                     " set,foo\n          ^\n";

  RunTest(Src, ExpectedErr);
}
