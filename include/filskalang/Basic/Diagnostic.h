/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#ifndef FILSKALANG_BASIC_DIAGNOSTIC_H
#define FILSKALANG_BASIC_DIAGNOSTIC_H

#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Support/raw_ostream.h>

namespace filskalang {
namespace diag {
enum {
#define DIAG(ID, Level, Msg) ID,
#include "./Diagnostic.def"
};
} // namespace diag

class DiagnosticsEngine {
  static const char *getDiagnosticText(unsigned DiagID);
  static llvm::SourceMgr::DiagKind getDiagnosticKind(unsigned DiagID);
  llvm::SourceMgr &SrcMgr;
  unsigned NumErrors;

public:
  DiagnosticsEngine(llvm::SourceMgr &SrcMgr) : SrcMgr(SrcMgr), NumErrors(0) {}

  unsigned numErrors() { return NumErrors; }

  template <typename... Args>
  void report(mlir::SMLoc Loc, unsigned DiagID, Args &&...Arguments) {
    std::string Msg = llvm::formatv(getDiagnosticText(DiagID),
                                    std::forward<Args>(Arguments)...)
                          .str();
    llvm::SourceMgr::DiagKind Kind = getDiagnosticKind(DiagID);
    SrcMgr.PrintMessage(Loc, Kind, Msg);
    NumErrors += (Kind == llvm::SourceMgr::DK_Error);
  }
};
} // namespace filskalang

#endif
