//! A simple tensor implementation
//! =============================================================
//!
//!
//! Introduction
//! ------------
//! This is a type less tensor library with the option to use
//! built-in operators or third-party acceleration library.
//! Some API for tensor to implement are listed in [tensor_trait].
//! [typed_tensor] is an enum to cover the tensor type information for [tensor].
//!
//! Currently, there are over 80 methods for [tensor].
//!
//! Install
//! ------------
//! cargo install tensor-rs
//!
//! Example
//! ------------
//! The following example shows a dip to using the package.
//!
//!     use tensor_rs::tensor_impl::gen_tensor::*;
//!     let m1 = GenTensor::<f64>::new_raw(&vec![0.; 3*5*2], &vec![3,5,2]);
//!     assert_eq!(m1.stride(), vec![10,2,1]);
//!
//! Licese
//! ------------


pub mod tensor;
pub mod quaternion;
pub mod typed_tensor;
pub mod tensor_trait;
pub mod tensor_impl;
#[cfg(feature = "use-serde")]
pub mod serde;
