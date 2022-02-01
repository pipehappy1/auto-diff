
//! A machine learning library with a 1st order learning approach
//! =============================================================
//!
//!
//! Introduction
//! ------------
//!
//! Install
//! ------------
//!
//! Example
//! ------------
//!
//! Licese
//! ------------

extern crate ndarray;
extern crate ndarray_linalg;

pub mod var;
pub mod op;
pub mod optim;
pub mod err;

pub use var::{Var};

pub(crate) mod compute_graph;
pub(crate) mod collection;


