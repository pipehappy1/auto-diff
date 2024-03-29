#![allow(dead_code)]
#![allow(unused_variables)]
//#![no_std]
//! An auto-difference library
//! =============================================================
//!
//!
//! Introduction
//! ------------
//! This is yet another auto-difference library for deep neural network.
//! The focus is easy on use and dynamic computation graph building.
//!
//! Install
//! ------------
//! Add auto-diff = "0.5" to the \[dependencies\] section of your project Cargo.toml file.
//!
//! Features
//! ------------
//! The forward operators support a commonly used set, including:
//!
//! 1. getter/setter,
//! 2. index and slicing,
//! 3. +, -, *, / and matmul,
//! 4. speciall functions,
//! 5. statistics,
//! 6. linear algebra,
//! 7. random number generator.
//!
//! The corresponding gradient is work-in-progress.
//!
//! One feature of auto-diff is the auto-difference is in background
//! and don't get in your way if only forward calculation is needed.
//! Thus it can be used without syntax like variable place holder.
//!
//! Example
//! ------------
//!

pub mod err;
pub mod op;
pub mod optim;
pub mod var;

pub use err::AutoDiffError;
pub use var::Var;

pub mod collection;
pub mod compute_graph;
#[cfg(feature = "use-serde")]
pub mod serde;
pub mod var_inner;
