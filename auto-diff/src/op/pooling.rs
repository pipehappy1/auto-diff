use super::{OpHandle, OpTrait};
use tensor_rs::tensor::Tensor;

#[cfg(feature = "use-serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

// MaxPool1d
// Maxpool2d
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct MaxPool2d {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: Tensor,
    dilation: (usize, usize),
    return_indices: bool,
    ceil_mode: bool,
}
impl MaxPool2d {
    pub fn new(
        kernel_size: Option<(usize, usize)>,
        stride: Option<(usize, usize)>,
        padding: Option<Tensor>,
        dilation: Option<(usize, usize)>,
        return_indices: Option<bool>,
        ceil_mode: Option<bool>,
    ) -> MaxPool2d {
        let kernel_size = if let Some(v) = kernel_size { v } else { (2, 2) };
        let stride = if let Some(v) = stride { v } else { (2, 2) };
        let padding = if let Some(v) = padding {
            v
        } else {
            Tensor::zeros(&[1])
        };
        let dilation = if let Some(v) = dilation { v } else { (2, 2) };
        let return_indices = if let Some(v) = return_indices {
            v
        } else {
            false
        };
        let ceil_mode = if let Some(v) = ceil_mode { v } else { false };
        MaxPool2d {
            handle: OpHandle::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
            ceil_mode,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpTrait for MaxPool2d {
    fn get_name(&self) -> &'static str {
        "MaxPool2d"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        unimplemented!();
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// MaxPool3d
// MaxUnpool1d
// MaxUnpool2d
// MaxUnpool3d
// AvgPool1d
// AvgPool2d
// AvgPool3d
// FractionalMaxPool2d
// LPPool1d
// LPPool2d
// AdaptiveMaxPool1d
// AdaptiveMaxPool2d
// AdaptiveMaxPool3d
// AdaptiveAvgPool1d
// AdaptiveAvgPool2d
// AdaptiveAvgPool3d
//
