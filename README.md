# filskalang
a compiler for Filska language in "[Strange Code](https://github.com/rkneusel9/StrangeCodeBook/blob/master/chapter_12/filska.py)" powered by MLIR

# prepare

```bash
# add repository
# NOTE: deb url depends on OS Version! https://apt.llvm.org/
$ wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
$ apt-add-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main"

# install llvm 18 and mlir 18
$ apt update
$ apt install -y libllvm-18-ocaml-dev libllvm18 llvm-18 llvm-18-dev llvm-18-doc llvm-18-examples llvm-18-runtime libmlir-18-dev libmlir-18 mlir-18-tools
```

# build

```bash
$ cmake -G Ninja . -DLLVM_DIR=/lib/llvm-18/lib/cmake/llvm -DMLIR_DIR=/lib/llvm-18/lib/cmake/mlir -DCMAKE_C_COMPILER=clang 
$ cmake --build .
```

# run

```bash
$ ./bin/filskalang --version
Filskalang 0.1
```
