/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/Version.h"
#include "filskalang/CodeGen/Dialect.h"
#include "filskalang/CodeGen/MLIRGen.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Parser/Parser.h"
#include "filskalang/Sema/Sema.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace filskalang;

namespace {
enum Action { None, EmitMLIR };
static llvm::cl::opt<enum Action> EmitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(EmitMLIR, "mlir", "output the MLIR")));
} // namespace

static llvm::cl::opt<std::string> InputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input-files>"),
                                            llvm::cl::init("-"));

void printVersion(llvm::raw_ostream &OS) {
  OS << "Filskalang " << filskalang::getFilskalangVersion() << "\n";
}

static const char *Head = "filskalang - Filska compiler";

int main(int Argc, const char **Argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(Argc, Argv, Head);

  // read input soruce code
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(InputFile);
  if (std::error_code BufferError = FileOrErr.getError()) {
    llvm::WithColor::error(llvm::errs(), Argv[0])
        << "Error reading " << InputFile << ": " << BufferError.message()
        << "\n";
    return 1;
  }

  llvm::SourceMgr SrcMgr;
  DiagnosticsEngine Diags(SrcMgr);
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), llvm::SMLoc());

  // parse input and generate AST
  auto Lex = Lexer(SrcMgr, Diags);
  auto Sem = Sema(Diags);
  auto P = Parser(Lex, Sem);
  auto *ProgramAST = P.parse();

  if (!ProgramAST || Diags.numErrors()) {
    return 1;
  }

  // generate MLIR
  mlir::MLIRContext Context;
  // Load our Dialect in this MLIR Context.
  Context.getOrLoadDialect<FilskalangDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(Context, *ProgramAST);
  if (!module) {
    return 1;
  }

  // only emit MLIR
  if (EmitAction == EmitMLIR) {
    module->dump();
    return 0;
  }

  // TODO: lower to llvm
}
