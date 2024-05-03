# filskalang
a compiler for Filska language in "[Strange Code: Esoteric Languages That Make Programming Fun Again](https://github.com/rkneusel9/StrangeCodeBook/blob/master/chapter_12/filska.py)" powered by MLIR

NOTE: This is under construction!

# progress

- [x] init project
- [x] tokens
- [x] lexer
- [x] prototyping
- [ ] parser
- [ ] evaluator (generate mlir)
- [ ] filskalang dialect
- [ ] lower to llir
- [ ] connect to llvm backend
- [ ] handle each operators

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
$ cmake -G Ninja .
$ cmake --build .
```

# run

```bash
$ ./bin/filskalang --version
Filskalang 0.1
$ ./bin/filskalang example/simple.filska -emit llvm 2>&1 | lli
10.000000
```

# development

## test

```bash
# set CMAKE_BUILD_TYPE=DEBUG for core dump
$ cmake -G Ninja . -DCMAKE_BUILD_TYPE=DEBUG
$ cmake --build .
$ ./run_tests.sh
```

## debug print

```bash
# show mlir in each pass
$ ./bin/filskalang --mlir-print-ir-after-all example/simple.filska -emit llvm
```

## troubleshooting a segmentation fault

```bash
$ lldb ./bin/filskalang
(lldb) run example/simple.filska
```
