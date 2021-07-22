//! A simple tensor implementation
//! =============================================================
//!
//!
//! Introduction
//! ------------
//! Tensor-rs is a simple tensor implementation for auto difference implementation.
//!
//! Install
//! ------------
//! The package is on the Cargo.rs and to use it. Put the following line in your Cargo.toml file.
//!
//!     [dev-dependencies]
//!     tensor-rs = "0.3"
//!
//! Example
//! ------------
//! The following example shows a dip to using the package.
//!
//!     use tensor_rs::tensor::gen_tensor::*;
//!     let m1 = GenTensor::<f64>::new_raw(&vec![0.; 3*5*2], &vec![3,5,2]);
//!     assert_eq!(m1.stride(), vec![10,2,1]);
//!
//! Licese
//! ------------


//extern crate ndarray;
//extern crate ndarray_linalg;

pub mod tensor;
