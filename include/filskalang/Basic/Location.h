#ifndef FILSKALANG_BASIC_LOCATION_H
#define FILSKALANG_BASIC_LOCATION_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/SMLoc.h"

namespace filskalang {
class Location {
  mlir::SMLoc Loc;
  int Line;
  int Column;
  const char *InputFileName;

public:
  Location(mlir::SMLoc Loc, int Line, int Column, const char *InputFileName)
      : Loc(Loc), Line(Line), Column(Column), InputFileName(InputFileName) {}

  // NOTE: SMLoc is used for Diagnostics
  mlir::SMLoc getLoc() { return Loc; }

  // NOTE: Location is used for MLIRGen
  mlir::Location getLocation(mlir::OpBuilder Builder) {
    return mlir::FileLineColLoc::get(Builder.getStringAttr(InputFileName), Line,
                                     Column);
  }
};
}; // namespace filskalang

#endif
