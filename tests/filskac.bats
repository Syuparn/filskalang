#!/usr/bin/env bats

setup() {
  load 'test_helper/bats-support/load'
  load 'test_helper/bats-file/load'
  load 'test_helper/bats-assert/load'
}

@test "filskac: show help" {
  run ./filskac -h
  assert_output "Filska compiler"$'\n'"filskac [-o sample] sample.filska"
}

@test "filskac: compile" {
  run ./filskac tests/src/compile/sample.filska
  assert_success
  assert_file_exist tests/src/compile/sample.filska.llir
  assert_file_exist tests/src/compile/sample.filska.llir.s
  assert_file_exist tests/src/compile/sample
}

@test "filskac: compiled file works" {
  run ./filskac tests/src/compile/sample.filska
  assert_success
  run ./tests/src/compile/sample
  assert_output "0.000000"
}

@test "filskac: compile with -o" {
  run ./filskac -o tests/src/compile/anothername tests/src/compile/sample.filska
  assert_success
  assert_file_exist tests/src/compile/sample.filska.llir
  assert_file_exist tests/src/compile/sample.filska.llir.s
  assert_file_exist tests/src/compile/anothername
}

@test "filskac: error: src is not specified" {
  run ./filskac
  assert_failure
  assert_output "error: source file must be specified"$'\n'"Filska compiler"$'\n'"filskac [-o sample] sample.filska"
}

@test "filskac: error: output is same as input" {
  run ./filskac -o tests/src/compile/dummy tests/src/compile/dummy
  assert_failure
  assert_output "error: stop compiling because the output file name 'tests/src/compile/dummy' is same as the src file name"$'\n'"make sure that src file name is Filska source code (*.filska)"
}

remove_file() {
  if [ -f "$1" ]; then
    rm $1
  fi
}

teardown() {
  remove_file tests/src/compile/sample.filska.llir
  remove_file tests/src/compile/sample.filska.llir.s
  remove_file tests/src/compile/sample
  remove_file tests/src/compile/anothername
}
