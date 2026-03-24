#!/bin/bash

path="./.pixi/envs/default/etc/conda/test-files/mojmelo/0/tests"
curr=$(pwd)
cd $path

mojo build setup.mojo -o setup
./setup
./setup 1
./setup 2
./setup 3
./setup 4
./setup 5
./setup 6
./setup 7
./setup 8
./setup 9
rm -f ./setup

cd $curr

mojo package $path/mojmelo/utils/mojmelo_matmul -o ./.pixi/envs/default/lib/mojo/mojmelo_matmul.mojopkg
