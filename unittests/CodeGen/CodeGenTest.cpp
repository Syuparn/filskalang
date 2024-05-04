#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/CodeGen/Dialect.h"
#include "filskalang/CodeGen/MLIRGen.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Lexer/Token.h"
#include "filskalang/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-port.h"

void RunTest(const char *Src, const char *Expected) {
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

  mlir::MLIRContext Context;
  // Load our Dialect in this MLIR Context.
  Context.getOrLoadDialect<mlir::filskalang::FilskalangDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> Module = filskalang::mlirGen(Context, *AST);

  Module->dump();

  auto Actual = testing::internal::GetCapturedStderr();
  ASSERT_STREQ(Actual.c_str(), Expected);
}

// subprogram

TEST(CodeGenTest, Simple) {
  auto Src = R"(
    { main }
    )";

  auto Expected = R"("builtin.module"() ({
  "filskalang.subprogram"() <{function_type = () -> (), sym_name = "main"}> ({
  ^bb0:
  }) : () -> ()
}) : () -> ()
)";

  RunTest(Src, Expected);
}

// TODO: multiple subprograms

// nullary instructions

TEST(CodeGenTest, Hlt) {
  auto Src = R"(
    { main
      hlt
    }
    )";

  auto Expected = R"("builtin.module"() ({
  "filskalang.subprogram"() <{function_type = () -> (), sym_name = "main"}> ({
    "filskalang.hlt"() : () -> ()
  }) : () -> ()
}) : () -> ()
)";

  RunTest(Src, Expected);
}

TEST(CodeGenTest, Prt) {
  auto Src = R"(
    { main
      prt
    }
    )";

  auto Expected = R"("builtin.module"() ({
  "filskalang.subprogram"() <{function_type = () -> (), sym_name = "main"}> ({
    "filskalang.prt"() <{subprogramName = "main"}> : () -> ()
  }) : () -> ()
}) : () -> ()
)";

  RunTest(Src, Expected);
}

// unary instructions

TEST(CodeGenTest, Set) {
  auto Src = R"(
    { main
      set,10
    }
    )";

  auto Expected = R"("builtin.module"() ({
  "filskalang.subprogram"() <{function_type = () -> (), sym_name = "main"}> ({
    "filskalang.set"() <{subprogramName = "main", value = 1.000000e+01 : f64}> : () -> ()
  }) : () -> ()
}) : () -> ()
)";

  RunTest(Src, Expected);
}

// multiple instructions

TEST(CodeGenTest, MultipleInstructions) {
  auto Src = R"(
    { main
      set,10
      prt
    }
    )";

  auto Expected =
      R"("builtin.module"() ({
  "filskalang.subprogram"() <{function_type = () -> (), sym_name = "main"}> ({
    "filskalang.set"() <{subprogramName = "main", value = 1.000000e+01 : f64}> : () -> ()
    "filskalang.prt"() <{subprogramName = "main"}> : () -> ()
  }) : () -> ()
}) : () -> ()
)";

  RunTest(Src, Expected);
}
