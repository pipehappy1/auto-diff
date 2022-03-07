#!/bin/bash

cd ../tensorboard-proto
cargo publish

cd ../auto-diff/tensor-rs
cargo publish

cd ../auto-diff
cargo publish

cd ../tensorboard-rs
cargo publish


