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
  let cppNamespace = "mlir::filskalang";
}

// Base class for filskalang dialect operations.
class Filskalang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Filskalang_Dialect, mnemonic, traits>;

// Nullary Operations

def PrtOp : Filskalang_Op<"prt"> {
  let summary = "print operation";
  let description = [{
    The "filskalang.prt" builtin operation represents the "prt" instruction
  }];

  let arguments = (ins SymbolNameAttr:$subprogramName);
}

// Unary Operations

def SetOp : Filskalang_Op<"set"> {
  let summary = "set operation";
  let description = [{
    The "filskalang.set" builtin operation represents the "set" instruction
  }];

  let arguments = (ins F64Attr:$value, SymbolNameAttr:$subprogramName);
}

def SubprogramOp : Filskalang_Op<"subprogram", [
    FunctionOpInterface,
  ]> {
  let summary = "subprogram operation";
  let description = [{
    The "filskalang.subprogram" operation represents a subprogram.
    This must have no parameters and return values.
  }];

  // NOTE: dummy parameters and return values are required for FunctionOpInterface
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SubprogramOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let skipDefaultBuilders = 1;
}

#endif
