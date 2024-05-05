/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Basic/Diagnostic.h"
#include "filskalang/Basic/Version.h"
#include "filskalang/CodeGen/Dialect.h"
#include "filskalang/CodeGen/LowerToLLVM.h"
#include "filskalang/CodeGen/MLIRGen.h"
#include "filskalang/Lexer/Lexer.h"
#include "filskalang/Parser/Parser.h"
#include "filskalang/Sema/Sema.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace filskalang;

namespace {
enum Action { None, EmitMLIR, EmitLLVMIR };
static llvm::cl::opt<enum Action> EmitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(EmitMLIR, "mlir", "output the MLIR")),
    llvm::cl::values(clEnumValN(EmitLLVMIR, "llvm", "output the LLVM IR")));

static llvm::cl::opt<bool> EnableOpt("opt",
                                     llvm::cl::desc("Enable optimizations"));
} // namespace

static llvm::cl::opt<std::string> InputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input-files>"),
                                            llvm::cl::init("-"));

void printVersion(llvm::raw_ostream &OS) {
  OS << "Filskalang " << filskalang::getFilskalangVersion() << "\n";
}

static const char *Head = "filskalang - Filska compiler";

int initLowering(mlir::OwningOpRef<mlir::ModuleOp> &Module) {
  mlir::PassManager PM(Module.get()->getName());

  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(PM))) {
    llvm::WithColor(llvm::errs())
        << "Failed apply pass manager options from command line options\n";
    return 4;
  }

  // Finish lowering the toy IR to the LLVM dialect.
  PM.addPass(mlir::filskalang::createLowerToLLVMPass());

  // This is necessary to have line tables emitted and basic
  // debugger working. In the future we will add proper debug information
  // emission directly from our frontend.
  PM.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

  if (mlir::failed(PM.run(*Module))) {
    llvm::WithColor(llvm::errs()) << "Failed run pass manager\n";
    return 4;
  }
  return 0;
}

int lowerToLLVM(mlir::OwningOpRef<mlir::ModuleOp> &Module) {
  if (int error = initLowering(Module)) {
    llvm::WithColor(llvm::errs()) << "Failed to initialize lowering\n";
    return error;
  }

  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*Module->getContext());
  mlir::registerLLVMDialectTranslation(*Module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext LLVMContext;
  auto LLVMModule = mlir::translateModuleToLLVMIR(*Module, LLVMContext);
  if (!LLVMModule) {
    llvm::WithColor(llvm::errs()) << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto TMBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!TMBuilderOrError) {
    llvm::WithColor(llvm::errs())
        << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto TMOrError = TMBuilderOrError->createTargetMachine();
  if (!TMOrError) {
    llvm::WithColor(llvm::errs()) << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(LLVMModule.get(),
                                                        TMOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(LLVMModule.get())) {
    llvm::WithColor(llvm::errs())
        << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *LLVMModule << "\n";
  return 0;
}

int main(int Argc, const char **Argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(Argc, Argv, Head);

  // read input soruce code
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(InputFile);
  if (std::error_code BufferError = FileOrErr.getError()) {
    llvm::WithColor::error(llvm::errs())
        << "Error reading " << InputFile << ": " << BufferError.message()
        << "\n";
    return 1;
  }

  llvm::SourceMgr SrcMgr;
  DiagnosticsEngine Diags(SrcMgr);
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), mlir::SMLoc());

  // parse input and generate AST
  auto Lex = Lexer(SrcMgr, Diags, InputFile.c_str());
  auto Sem = Sema(Diags);
  auto P = Parser(Lex, Sem);
  auto *ProgramAST = P.parse();

  if (!ProgramAST || Diags.numErrors()) {
    return 1;
  }

  // generate MLIR
  mlir::MLIRContext Context;
  // Load our Dialect in this MLIR Context.
  Context.getOrLoadDialect<mlir::filskalang::FilskalangDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> Module = mlirGen(Context, *ProgramAST);
  if (!Module) {
    return 1;
  }

  // only emit MLIR
  if (EmitAction == EmitMLIR) {
    Module->dump();
    return 0;
  }

  // lower to llvm
  if (EmitAction == EmitLLVMIR) {
    auto result = lowerToLLVM(Module);
    return result;
  }

  // TODO: compile to the target
}
