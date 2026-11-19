#!/bin/sh

set -xe

clang -Wall -Wextra -std=c11 -O2 -o intro intro.c -lm
./intro

