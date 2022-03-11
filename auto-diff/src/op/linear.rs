use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};

use std::cell::{RefCell};
use std::rc::Rc;

use crate::var::{Var};
use crate::err::AutoDiffError;

#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Linear {
    in_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
    weight_grad: Tensor,
    bias_grad: Tensor,
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl Linear {
    pub fn new(in_features: Option<usize>,
               out_features: Option<usize>,
               bias: bool) -> Linear {
        Linear {
            in_fea: in_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
            weight_grad: Tensor::new(),
            bias_grad: Tensor::new(),
            handle: OpHandle::new(),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn set_weight(&self, var: Var) {
        self.weight.swap(&var.val());
    }
    
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
    
    pub fn set_bias(&self, var: Var) {
        self.bias.swap(&var.val());
    }

    handle_method!();
}

impl OpCall for Linear {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Linear {
            in_fea: self.in_fea,
            out_fea: self.out_fea,
            bias_option: self.bias_option,
            weight: self.weight.ref_copy(),
            bias: self.bias.ref_copy(),
            weight_grad: self.weight_grad.ref_copy(),
            bias_grad: self.bias_grad.ref_copy(),
            handle: OpHandle::new(), // TODO; change this to None, this shold never be used.
        };
        
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        
        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }


}

impl OpTrait for Linear {

    

    fn get_name(&self) -> &'static str {
        "Linear"
    }

    fn get_input_size(&self) -> usize {
        1
    }

    fn get_output_size(&self) -> usize {
        1
    }

    fn apply(&self, inputs: &[Tensor],
             outputs: &[Tensor]) {
        // TODO go through condition where dimension is missing somewhere.
        //println!("left sie: {:?}, right size: {:?}", inputs[0], self.weight);
        let ret = inputs[0].matmul(&self.weight);
        outputs[0].swap(&ret);
        //println!("matmut done");
        if self.bias_option {
            let ret = outputs[0].add(&self.bias);
            outputs[0].swap(&ret);
        }
    }

    fn grad(&self, inputs: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        if inputs.is_empty() {
            panic!("Expect one input tensor");
        }
        if inputs[0].size()[1] != self.weight.size()[0] {
            panic!("Expect input dimension matches weight dimension {:?}, {:?}",
                   inputs[0].size(), self.weight.size());
        }
        if inputs[0].size()[0] != output_grad[0].size()[0] {
            panic!("Expect input population matches output gradient population {:?}, {:?}",
                   inputs[0].size(), output_grad[0].size());
        }
        if output_grad[0].size()[1] != self.weight.size()[1] {
            panic!("Expect output gradient dimension matches weight dimension {:?}, {:?}",
                   output_grad[0].size(), self.weight.size());
        }

        input_grad[0].swap(&output_grad[0].matmul(&self.weight.permute(&[1,0])));
        self.weight_grad.swap(&inputs[0].outer(&output_grad[0], Some(true)));
        if self.bias_option {
            self.bias_grad.swap(&output_grad[0].mean(Some(&[0]), false));
        }
    }

    fn get_values(&self) -> Vec<Tensor> {
        let mut ret = vec![self.weight.clone()];
        if self.bias_option {
            ret.push(self.bias.clone());
        }
        ret
    }
    fn set_values(&self, v: &[Tensor]) {
        self.weight.swap(&v[0].clone());
        if self.bias_option {
            self.bias.swap(&v[1].clone());
        }
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        let mut ret = vec![self.weight_grad.clone()];
        if self.bias_option {
            ret.push(self.bias_grad.clone());
        }
        ret
    }

    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
	self
    }
    
}

// Bilinear
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct BiLinear {
    in1_fea: Option<usize>,
    in2_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
    weight_grad: Tensor,
    bias_grad: Tensor,
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl BiLinear {
    pub fn new(in1_features: Option<usize>,
               in2_features: Option<usize>,
               out_features: Option<usize>,
               bias: bool) -> BiLinear {
        BiLinear {
            in1_fea: in1_features,
            in2_fea: in2_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
            weight_grad: Tensor::new(),
            bias_grad: Tensor::new(),
            handle: OpHandle::new(),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn set_weight(&self, var: Var) {
        self.weight.swap(&var.val());
    }
    
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
    
    pub fn set_bias(&self, var: Var) {
        self.bias.swap(&var.val());
    }

    handle_method!();
}

impl OpCall for BiLinear {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = BiLinear {
            in1_fea: self.in1_fea,
            in2_fea: self.in2_fea,
            out_fea: self.out_fea,
            bias_option: self.bias_option,
            weight: self.weight.ref_copy(),
            bias: self.bias.ref_copy(),
            weight_grad: self.weight_grad.ref_copy(),
            bias_grad: self.bias_grad.ref_copy(),
            handle: OpHandle::new(), // TODO; change this to None, this shold never be used.
        };
        
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        
        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }


}

impl OpTrait for BiLinear {
    fn get_name(&self) -> &'static str {
        "BiLinear"
    }

    fn get_input_size(&self) -> usize {
        2
    }

    fn get_output_size(&self) -> usize {
        1
    }

    fn apply(&self, inputs: &[Tensor],
             outputs: &[Tensor]) {
        
        unimplemented!();
        
    }

    fn grad(&self, inputs: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        if inputs.is_empty() {
            panic!("Expect one input tensor");
        }
        if inputs[0].size()[1] != self.weight.size()[0] {
            panic!("Expect input1 dimension matches weight dimension {:?}, {:?}",
                   inputs[0].size(), self.weight.size());
        }
        if self.weight.size()[1] != inputs[1].size()[0] {
            panic!("Expect weight dimension matches input2 dimension {:?}, {:?}",
                   self.weight.size(), inputs[1].size());
        }

        unimplemented!();
    }

    fn get_values(&self) -> Vec<Tensor> {
        let mut ret = vec![self.weight.clone()];
        if self.bias_option {
            ret.push(self.bias.clone());
        }
        ret
    }
    fn set_values(&self, v: &[Tensor]) {
        self.weight.swap(&v[0].clone());
        if self.bias_option {
            self.bias.swap(&v[1].clone());
        }
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        let mut ret = vec![self.weight_grad.clone()];
        if self.bias_option {
            ret.push(self.bias_grad.clone());
        }
        ret
    }

    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
	self
    }
    
}
