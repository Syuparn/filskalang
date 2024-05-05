#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-assert/load'
}

@test "filskalang: show version" {
  run ./bin/filskalang --version
  assert_output --regexp "Filskalang [0-9]+\.[0-9]+\.[0-9]+"
}

@test "filskalang: emit mlir" {
  run ./bin/filskalang tests/src/prt.filska --emit mlir
  # only check if it is filskalang dialect
  assert_output --partial "filskalang.subprogram"
}
