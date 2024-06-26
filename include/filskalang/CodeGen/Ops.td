#ifndef FILSKALANG_OPS
#define FILSKALANG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
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

// Register Operations
def RegisterOp : Filskalang_Op<"register"> {
  let summary = "register operation";
  let description = [{
    The "filskalang.register" builtin operation represents the value in the register `x`, `y`, `z`, or `m`.
  }];

  let arguments = (ins SymbolNameAttr:$name);
  let results = (outs F64);
}

// Nullary Operations

def HltOp : Filskalang_Op<"hlt"> {
  let summary = "halt operation";
  let description = [{
    The "filskalang.hlt" builtin operation represents the "hlt" instruction
  }];
}

def NegOp : Filskalang_Op<"neg"> {
  let summary = "negate operation";
  let description = [{
    The "filskalang.neg" builtin operation represents the "neg" instruction
  }];

  let arguments = (ins F64:$arg);
  let results = (outs F64);
}

def PrtOp : Filskalang_Op<"prt"> {
  let summary = "print operation";
  let description = [{
    The "filskalang.prt" builtin operation represents the "prt" instruction
  }];

  let arguments = (ins F64:$arg);
}

// Unary Operations

def SetOp : Filskalang_Op<"set"> {
  let summary = "set operation";
  let description = [{
    The "filskalang.set" builtin operation represents the "set" instruction
  }];

  let arguments = (ins F64Attr:$value, SymbolNameAttr:$subprogramName);
}

def ProgramOp : Filskalang_Op<"program", [
    FunctionOpInterface,
  ]> {
  let summary = "program operation";
  let description = [{
    The "filskalang.program" operation represents a program.
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
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // ProgramOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let skipDefaultBuilders = 1;
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

// meta operator

def MetaSetOp : Filskalang_Op<"metaset"> {
  let summary = "meta set operation";
  let description = [{
    The "filskalang.metaset" operation is used to push back the caluculation result to register `m`.
  }];

  let arguments = (ins F64:$value, SymbolNameAttr:$subprogramName);
}

// dummy operator

def DummyTerminatorOp : Filskalang_Op<"dummyterminator", [Terminator]> {
  let summary = "dummy terminator operation";
  let description = [{
    The "filskalang.dummyterminator" operation is a placeholder operator at the end of each subprogram.
    It is only used to satisfy the terminator constraint in an MLIR block;
    the block must have one and only one terminator operator at the end.
  }];
}

#endif
