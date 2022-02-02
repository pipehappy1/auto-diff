
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
//! Add auto-diff = "0.5" to the [dependencies] section of your project Cargo.toml file.
//!
//! Example
//! ------------
//!


extern crate ndarray;
extern crate ndarray_linalg;

pub mod var;
pub mod op;
pub mod optim;
pub mod err;

pub use var::{Var};

pub(crate) mod compute_graph;
pub(crate) mod collection;
pub(crate) mod var_inner;

