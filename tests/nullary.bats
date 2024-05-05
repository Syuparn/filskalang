#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-assert/load'
}

@test "prt instruction" {
  run bash -c './bin/filskalang tests/src/prt.filska -emit llvm | lli'
  assert_output "0.000000"
}
