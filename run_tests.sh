#!/bin/bash

echo "unit tests"
for f in $(ls ./bin/*_test); do
    $f
done

echo "integration tests"
bats tests
