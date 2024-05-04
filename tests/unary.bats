#!/usr/bin/env bats

# TODO: replace lli with compile options

@test "set instruction" {
  result="$(./bin/filskalang tests/src/set.filska -emit llvm | lli)"
  [ "$result" = "10.000000" ]
}
