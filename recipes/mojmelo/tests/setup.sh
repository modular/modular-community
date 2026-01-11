#!/bin/bash

path="./.pixi/envs/default/etc/conda/test-files/mojmelo/0/tests"
curr=$(pwd)
cd $path

mojo ./setup.mojo
mojo ./setup.mojo 1
mojo ./setup.mojo 2
mojo ./setup.mojo 3
mojo ./setup.mojo 4
mojo ./setup.mojo 5
mojo ./setup.mojo 6
mojo ./setup.mojo 7
mojo ./setup.mojo 8
mojo ./setup.mojo 9

cd $curr

mojo package $path/mojmelo/utils/mojmelo_matmul -o ./.pixi/envs/default/lib/mojo/mojmelo_matmul.mojopkg
