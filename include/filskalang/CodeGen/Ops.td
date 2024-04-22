#ifndef FILSKALANG_OPS
#define FILSKALANG_OPS

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

#endif
