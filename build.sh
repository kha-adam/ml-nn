#!/bin/sh

set -xe

# clang -Wall -Wextra -std=c11 -O2 -o intro intro.c -lm
# ./intro

clang -Wall -Wextra -std=c11 -O2 -o abst abst.c -lm
./abst

