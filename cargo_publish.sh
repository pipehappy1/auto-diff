#!/bin/bash


cd ./tensorboard-rs
cargo publish

cd ../tensor-rs
cargo publish

sleep 2

cd ../macros
cargo publish

cd ../auto-diff
cargo publish

sleep 2

cd ../data-pipe
cargo publish

sleep 2

cd ../ann
cargo publish


