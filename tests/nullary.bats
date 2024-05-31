#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-assert/load'
}

@test "neg instruction" {
  run bash -c './bin/filskalang tests/src/neg.filska -emit llvm | lli'
  assert_output "-10.000000"
}

@test "prt instruction" {
  run bash -c './bin/filskalang tests/src/prt.filska -emit llvm | lli'
  assert_output "0.000000"
}

@test "prt twice" {
  run bash -c './bin/filskalang tests/src/prt_twice.filska -emit llvm | lli'
  # TODO: add breakline
  assert_output "0.000000""0.000000"
}
