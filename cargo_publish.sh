#!/bin/bash

cd ../tensorboard-proto
cargo publish

cd ../tensorboard-rs
cargo publish

cd ../auto-diff/tensor-rs
cargo publish

sleep 2

cd ../auto-diff
cargo publish




