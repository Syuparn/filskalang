/*
  Source code in this file is inherited and modified from tinylang
  https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter05/tinylang
  MIT License | Copyright (c) 2023 Packt
  see https://opensource.org/licenses/MIT
*/

#include "filskalang/Basic/Version.h"

std::string filskalang::getFilskalangVersion() {
  return FILSKALANG_VERSION_STRING;
}
