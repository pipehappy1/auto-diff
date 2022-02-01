use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};

use std::cell::{RefCell, Ref};
use std::rc::Rc;

use crate::var::{VarInner, Var};
use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};
use crate::err::AutoDiffError;


pub struct Linear {
    in_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
    weight_grad: Tensor,
    bias_grad: Tensor,
    
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
        self.weight.data_copy(&var.val());
    }
    
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
    
    pub fn set_bias(&self, var: Var) {
        self.bias.data_copy(&var.val());
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
        
        Ok(inputs[0].called_with(op, &inputs[1..inputs.len()])?)
    }


}

impl OpTrait for Linear {

    

    fn get_name(&self) -> String {
        "Linear".to_string()
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
        outputs[0].data_copy(&ret);
        //println!("matmut done");
        if self.bias_option {
            let ret = outputs[0].add(&self.bias);
            outputs[0].data_copy(&ret);
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
        // TODO
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
        unimplemented!()
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        // TODO
        Vec::new()
    }
    
}

//impl OpTrait for Linear2 {
//    /// A conventional name for the op
//    fn get_name(&self) -> String {
//        "ab".to_string()
//    }
//
//    /// The number of input needs by this op.
//    fn get_input_size(&self) -> usize {
//        2
//    }
//
//    /// The number of output produced by this op.
//    fn get_output_size(&self) -> usize {
//        1
//    }
//
//    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
//        
//    }
//
//    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
//        
//    }
//
//    /// access weight values
//    fn get_values(&self) -> Vec<&Tensor> {
//        Vec::new()
//    }
//    fn set_values(&self, v: &[Tensor]) {
//    }
//    /// access gradient values
//    fn get_grads(&self) -> Vec<&Tensor> {
//        Vec::new()
//    }
//}
//
//pub struct Linear {
//    in_fea: Option<usize>,
//    out_fea: Option<usize>,
//    bias_option: bool,
//    weight: Tensor,
//    bias: Tensor,
//    weight_grad: Tensor,
//    bias_grad: Tensor,
//}
//impl Linear {
//    pub fn new(in_features: Option<usize>, out_features: Option<usize>, bias: bool) -> Linear{
//        let mut ret = Linear {
//            in_fea: in_features,
//            out_fea: out_features,
//            bias_option: bias,
//            weight: Tensor::new(),
//            bias: Tensor::new(),
//            weight_grad: Tensor::new(),
//            bias_grad: Tensor::new(),
//        };
//        if ret.in_fea != Option::None && ret.out_fea != Option::None {
//            ret._new();
//        }
//        ret
//    }
//    fn _new(&mut self) {
//        self.weight = Tensor::fill(&[self.in_fea.unwrap(), self.out_fea.unwrap()], 0.);
//        self.bias = Tensor::fill(&[self.out_fea.unwrap(),], 0.);
//    }
//
//    pub fn weight(&self) -> &Tensor {
//        &self.weight
//    }
//
//    pub fn set_weight(&self, var: Var) {
//        self.weight.swap(var.val());
//    }
//    
//    pub fn bias(&self) -> &Tensor {
//        &self.bias
//    }
//    
//    pub fn set_bias(&self, var: Var) {
//        self.bias.swap(var.val());
//    }
//
//}
//impl OpTrait for Linear {
//    fn get_name(&self) -> String {
//        "Linear".to_string()
//    }
//    fn get_input_size(&self) -> usize {
//        1
//    }
//    fn get_output_size(&self) -> usize {
//        1
//    }
//    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
//        if self.in_fea == None || self.out_fea == None {
//            if self.in_fea == None {
//                let in_size = input[0].size();
//                self.in_fea = Some(in_size[in_size.len()-1]);
//            }
//            if self.out_fea == None {
//                let out_size = output[0].size();
//                self.out_fea = Some(out_size[0]);
//            }
//            self._new();
//        }
//
//        //println!("left sie: {:?}, right size: {:?}", input[0].size(), self.weight.size());
//        let ret = input[0].matmul(&self.weight);
//        output[0].swap(ret);
//        //println!("matmut done");
//        if self.bias_option {
//            let ret = output[0].add(&self.bias);
//            output[0].swap(ret);
//        }
//    }
//    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
//        if input.is_empty() {
//            panic!("Expect one input tensor");
//        }
//        if input[0].size()[1] != self.weight.size()[0] {
//            panic!("Expect input dimension matches weight dimension {:?}, {:?}",
//                   input[0].size(), self.weight.size());
//        }
//        if input[0].size()[0] != output_grad[0].size()[0] {
//            panic!("Expect input population matches output gradient population {:?}, {:?}",
//                   input[0].size(), output_grad[0].size());
//        }
//        if output_grad[0].size()[1] != self.weight.size()[1] {
//            panic!("Expect output gradient dimension matches weight dimension {:?}, {:?}",
//                   output_grad[0].size(), self.weight.size());
//        }
//
//        input_grad[0].swap(output_grad[0].matmul(&self.weight.permute(&[1,0])));
//        self.weight_grad.swap(input[0].outer(output_grad[0], Some(true)));
//        if self.bias_option {
//            self.bias_grad.swap(output_grad[0].mean(Some(&[0]), false));
//        }
//    }
//
//    fn get_values(&self) -> Vec<&Tensor> {
//        let mut ret = vec![&self.weight];
//        if self.bias_option {
//            ret.push(&self.bias);
//        }
//        ret
//    }
//    fn set_values(&self, v: &[Tensor]) {
//        self.weight.swap(v[0].clone());
//        if self.bias_option {
//            self.bias.swap(v[1].clone());
//        }
//    }
//    fn get_grads(&self) -> Vec<&Tensor> {
//        let mut ret = vec![&self.weight_grad];
//        if self.bias_option {
//            ret.push(&self.bias_grad);
//        }
//        ret
//    }
//}

// Bilinear
