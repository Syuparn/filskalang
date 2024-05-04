#!/bin/bash

for f in $(ls ./bin/*_test); do
    $f
done
