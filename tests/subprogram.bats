#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-assert/load'
}

@test "multiple subprograms" {
  run bash -c './bin/filskalang tests/src/subprogram.filska -emit llvm | lli'
  assert_output "10.000000"
}

# TODO: add testcase for subprograms with jmp/jpr
