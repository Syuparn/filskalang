#!/bin/bash

for f in $(find . -name 'Test'); do
    $f
done
