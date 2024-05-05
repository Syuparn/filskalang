#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-assert/load'
}

@test "set instruction" {
  run bash -c './bin/filskalang tests/src/set.filska -emit llvm | lli'
  assert_output "10.000000"
}

# TODO: insert space inside
@test "set changes m" {
  run bash -c './bin/filskalang tests/src/set_memory.filska -emit llvm | lli'
  assert_output "0.00000010.000000"
}
