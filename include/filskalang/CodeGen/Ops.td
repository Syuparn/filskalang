#ifndef FILSKALANG_OPS
#define FILSKALANG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Filskalang_Dialect : Dialect {
  let name = "filskalang";

  let summary = "A high-level dialect for analyzing and optimizing the "
                "Filska language";

  let description = [{
    filskalang is a MLIR implementation of the Filska language, an esolang
    in the book "Strange Code: Esoteric Languages That Make Programming Fun Again".
    The original interpreter is
    https://github.com/rkneusel9/StrangeCodeBook/blob/master/chapter_12/filska.py
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "filskalang";
}

// Base class for filskalang dialect operations.
class Filskalang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Filskalang_Dialect, mnemonic, traits>;

def PrtOp : Filskalang_Op<"prt"> {
  let summary = "print operation";
  let description = [{
    The "prt" builtin operation corresponds to "prt" instruction
  }];

  let arguments = (ins F64:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif
