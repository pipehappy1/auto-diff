use crate::tensor::Tensor;
use super::OpTrait;

pub struct Conv1d {
    alpha: f32,
}
impl Conv1d {
    pub fn new(alpha: f32) -> Conv1d {
        Conv1d {
            alpha: alpha,
        }
    }
}
impl OpTrait for Conv1d {
    fn get_name(&self) -> String {
        "Conv1d".to_string()
    }

    /// Forward pass
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        umimplemented!();
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        umimplemented!();
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

// Conv2d
pub enum PaddingMode{
    Zeros,
    Reflect,
    Replicate,
    Circular,
}

pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    bias_option: bool,
    padding_mode: PaddingMode,
    
    weight: Tensor,
    bias: Tensor,
    weight_grad: Tensor,
    bias_grad: Tensor,
}
impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize,
               kernel_size: (usize, usize),
               stride: (usize, usize),
               padding: (usize, usize),
               dilation: (usize, usize),
               bias: bool,
               padding_mode: PaddingMode
    ) -> Conv2d {
        Conv2d {
            in_channels: in_channels,
            out_channels: out_channels,
            kernel_size: kernel_size,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: 1,
            bias_option: bias,
            padding_mode: padding_mode,
            
            weight: Tensor::empty(&vec![out_channels, in_channels, kernel_size[0], kernel_size[1]]),
            bias: Tensor::empty(&vec![out_channels, ]),
            weight_grad: Tensor::empty(&vec![out_channels, in_channels, kernel_size[0], kernel_size[1]]),
            bias_grad: Tensor::empty(&vec![out_channels, ]),
        }
    }
}
impl OpTrait for Conv2d {
    fn get_name(&self) -> String {
        "Conv2d".to_string()
    }

    /// Forward pass
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        let output = input.conv2d(self.weight, stide, padding, dilation, 0);
        output[0].swap(output);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        umimplemented!();
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}
// Conv3d
// ConvTranspose1d
// ConvTranspose2d
// ConvTranspose3d
// Unfold
// Fold
