use tensor_rs::tensor::{Tensor, PaddingMode};
use std::cell::{RefCell, Ref};
use std::rc::Rc;
use super::{OpTrait, OpCall, Op, OpHandle};
use crate::var::Var;
use crate::err::AutoDiffError;

pub struct Conv1d {
    alpha: f32,
    handle: OpHandle,
}
impl Conv1d {
    pub fn new(alpha: f32) -> Conv1d {
        Conv1d {
            alpha: alpha,
            handle: OpHandle::new(),
        }
    }
    handle_method!();
}
impl OpTrait for Conv1d {
    fn get_name(&self) -> String {
        "Conv1d".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        unimplemented!();
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        unimplemented!();
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

    handle: OpHandle,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups: 1,
            bias_option: bias,
            padding_mode,
            
            weight: Tensor::empty(&[out_channels, in_channels, kernel_size.0, kernel_size.1]),
            bias: Tensor::empty(&[out_channels, ]),
            weight_grad: Tensor::empty(&[out_channels, in_channels, kernel_size.0, kernel_size.1]),
            bias_grad: Tensor::empty(&[out_channels, ]),

            handle: OpHandle::new(),
        }
    }

    handle_method!();
}

impl OpCall for Conv2d {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Conv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size.clone(),
            stride: self.stride.clone(),
            padding: self.padding.clone(),
            dilation: self.dilation.clone(),
            groups: self.groups,
            bias_option: self.bias_option,
            padding_mode: self.padding_mode,
            
            weight: self.weight.ref_copy(),
            bias: self.bias.ref_copy(),
            weight_grad: self.weight_grad.ref_copy(),
            bias_grad: self.bias_grad.ref_copy(),

            handle: OpHandle::new(),
        };
        
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        
        Ok(inputs[0].called_with(op, &inputs[1..inputs.len()])?)
    }
}

impl OpTrait for Conv2d {
    fn get_name(&self) -> String {
        "Conv2d".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        if self.groups > 1 {
            unimplemented!();
        }
        if self.weight.size()[2] != self.kernel_size.0 || self.weight.size()[3] != self.kernel_size.1 {
            panic!("this is conv2d");
        }
        let input_size = input[0].size();
        if input_size[1] != self.in_channels {
            panic!("conv2d expect the same input channel: input: {:?}, config: {:?}", input_size[1], self.in_channels);
        }
        let conv_output = input[0].conv2d(&self.weight, self.stride, self.padding, self.dilation, self.padding_mode);
        //println!("conv_output: {:?}, {:?}, {:?}, {:?}, {:?}, {:?}", self.weight.size(), self.stride, self.padding, self.dilation, conv_output.size(), input[0].size());
        if conv_output.size()[1] != self.out_channels {
            panic!("conv2d expect the same input channel {:?}, {:?}", input_size[1], self.in_channels);
        }

        if self.bias_option {
            //println!("{:?}, {:?}", self.weight.size(), self.bias.size());
            let expanded_bias = self.bias
                .unsqueeze(1)
                .unsqueeze(2)
                .repeat(&[1, conv_output.size()[2], conv_output.size()[3]]);
            //println!("conv_output: {:?}, expanded_bias.size() {:?}", conv_output.size(), expanded_bias.size());
            let ret = conv_output.add(&expanded_bias);
            output[0].swap(&ret);
        } else {
            output[0].swap(&conv_output);            
        }
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        let (w_grad, d_grad) = input[0].conv2d_grad(&self.weight,
                                                    self.stride,
                                                    self.padding,
                                                    self.dilation,
                                                    self.padding_mode,
                                                    &output_grad[0]);
        self.weight_grad.swap(&w_grad);
        input_grad[0].swap(&d_grad);

        if self.bias_option {
            self.bias_grad.swap(&output_grad[0].mean(Some(&[0, 2, 3]), false));
        }
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
    fn set_values(&self, v: &[Tensor]) {
        self.weight.data_copy(&v[0]);
        self.bias.data_copy(&v[1]);
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        vec![&self.weight_grad, &self.bias_grad]
    }
}
// Conv3d
// ConvTranspose1d
// ConvTranspose2d
// ConvTranspose3d
// Unfold
// Fold
