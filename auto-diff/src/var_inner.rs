use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::collections::BTreeMap;
use ::rand::prelude::StdRng;

use tensor_rs::tensor::{Tensor};
use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};
use crate::op::{Op, OpTrait,
                View,
                Add, Sub, Mul, Div, Matmul,
                ELU, ReLU, Sigmoid,
                MSELoss, BCEWithLogitsLoss, CrossEntropyLoss,
                Abs, Acos, Asin, Atan, Ceil, Cos, Cosh, Exp, Expm1, Floor, Frac, Log, Log10, Log1p, Log1pexp, Log2, Neg, Reciprocal, Round, Rsqrt, Sign, Sin, Sinh, Sqrt, Tan, Tanh, Trunc,
                MaxPair, MinPair, ArgSort, EqElem, Equal, Ge, Gt, Le, Lt, Ne,
                Cat, Chunk, Gather, IndexSelect, IndexExclude, Reshape, Split, Squeeze, Stack, T, Take, Permute, Unsqueeze, ConditionalSelect, Repeat,
                Det, Inv, NormalizeUnit,
                Argmax, Argmin, Logsumexp, Mean, Prod, Std, Sum, Variance, Max, Min,
                GetPatch,
};
use crate::err::AutoDiffError;
use crate::optim::Optimizer;


//macro_rules! var_inner_1_to_1_with_args {
//    ($a:ident, $b:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
//        pub fn $a(&self, $( $arg_name : $ArgTy ),*) -> Result<VarInner, AutoDiffError> {
//            if self.need_grad {
//                let ret = VarInner::new_net_tensor(self.net.clone(), Tensor::new());
//                let op = $b::new($( $arg_name ),*);
//                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy()],
//                         &[self.net.borrow().get_tensor(ret.id)?.ref_copy()]);
//                let op = Op::new(Rc::new(RefCell::new(Box::new(op))));
//                let opid = self.net.borrow_mut().add_op(op);
//                
//                self.net.borrow_mut().connect(&[self.id],
//                                              opid, &[ret.id]);
//                
//                Ok(ret)
//            } else {
//                let ret = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())), Tensor::new());
//                let op = $b::new($( $arg_name ),*);
//                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy()],
//                         &[ret.net.borrow().get_tensor(ret.id)?.ref_copy()]);
//                Ok(ret)
//            }
//        }
//    }
//}

/// For elementwise ops
/// var_inner_1_to_1!(abs, Abs);
macro_rules! var_inner_1_to_1 {
    ($a:ident, $b:ident) => {
        pub fn $a(&self) -> Result<VarInner, AutoDiffError> {
            if self.need_grad {
                
                let ret = VarInner::new_net_tensor(self.net.clone(),
                                                   self.need_grad,
                                                   Tensor::new());
                
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy()],
                         &[self.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                let op = Op::new(Rc::new(RefCell::new(Box::new(op))));
                let opid = self.net.borrow_mut().add_op(op);
                
                self.net.borrow_mut().connect(&[self.id],
                                              opid, &[ret.id]);
                
                Ok(ret)
            } else {
                let ret = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())),
                                                   self.need_grad,
                                                   Tensor::new());
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy()],
                         &[ret.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                Ok(ret)
            }
            
        }
    }
}


macro_rules! var_inner_2_to_1 {
    ($a:ident, $b:ident) => {
        pub fn $a(&self, other: &mut VarInner) -> Result<VarInner, AutoDiffError> {
            if self.need_grad {
                if !Rc::ptr_eq(&self.net, &other.net) {
                    let other_key = self.net.borrow_mut().append(
                        &mut other.net.borrow_mut(), &[other.id])?[0];
                
                    other.net = self.net.clone();
                    other.id = other_key;
                }
                
                let ret = VarInner::new_net_tensor(self.net.clone(),
                                                   self.need_grad,
                                                   Tensor::new());
                
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy(),
                           self.net.borrow().get_tensor(other.id)?.ref_copy()],
                         &[self.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                let op = Op::new(Rc::new(RefCell::new(Box::new(op))));
                let opid = self.net.borrow_mut().add_op(op);
                
                self.net.borrow_mut().connect(&[self.id, other.id],
                                              opid, &[ret.id]);
                
                Ok(ret)
            } else {
                let ret = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())),
                                                   self.need_grad,
                                                   Tensor::new());
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy(),
                           other.net.borrow().get_tensor(other.id)?.ref_copy()],
                         &[ret.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                Ok(ret)
            }
            
        }
    }
}

/// Multiple tensor in, 1 out and with parameters
macro_rules! var_inner_more_to_1_with_para {
    ($a:ident, $b:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a(&self, inputs: &[Rc<RefCell<VarInner>>],
        $( $arg_name : $ArgTy ),*) -> Result<VarInner, AutoDiffError> {
            let new_one = $b::new($( $arg_name ),*);
            let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
            let mut result = self.called_with(op, inputs)?;
            Ok(result.remove(0))            
        }
    }
}

macro_rules! var_inner_1_to_1_with_para {
    ($a:ident, $b:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a(&self, $( $arg_name : $ArgTy ),*) -> Result<VarInner, AutoDiffError> {
            let new_one = $b::new($( $arg_name ),*);
            let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
            let mut result = self.called_with(op, &[])?;
            Ok(result.remove(0))            
        }
    }
}



// Macro for creation associated function.
// Not for method.
macro_rules! delegate_new_inner_op {
    ($a:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a($( $arg_name : $ArgTy ),*) -> VarInner {
            let mut net = Net::new();
            let tensor = Tensor::$a($( $arg_name ),*);
            let id = net.add_tensor(tensor);
            VarInner {
                id,
                need_grad: true,
                net: Rc::new(RefCell::new(net)),
            }
        }
    }
}

pub(crate) struct VarInner {
    id: GenKey,
    need_grad: bool,
    net: Rc<RefCell<Net>>,
}

impl VarInner {

    // create functions.
    #[cfg(feature = "use-f64")]
    pub fn new(input: &[f64], dim: &[usize]) -> VarInner {
        let mut net = Net::new();
        
        let tensor = Tensor::from_vec_f64(input, dim);
        
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    #[cfg(feature = "use-f32")]
    pub fn new(input: &[f32], dim: &[usize]) -> VarInner {
        let mut net = Net::new();
        
        let tensor = Tensor::from_vec_f32(input, dim);
        
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    pub fn new_f64(input: &[f64], dim: &[usize]) -> VarInner {
        let mut net = Net::new();
        
        let tensor = Tensor::from_vec_f64(input, dim);
        
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    pub fn new_f32(input: &[f32], dim: &[usize]) -> VarInner {
        let mut net = Net::new();
        
        let tensor = Tensor::from_vec_f32(input, dim);
        
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }

    /// Create a new var with an existing net and value.
    pub(crate) fn new_net_tensor(net: Rc<RefCell<Net>>,
                                 need_grad: bool,
                                 tensor: Tensor) -> VarInner {
        let id = net.borrow_mut().add_tensor(tensor);
        VarInner {
            id,
            need_grad,
            net
        }
    }

    pub(crate) fn new_tensor(tensor: Tensor) -> VarInner {
        let mut net = Net::new();
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }


    pub fn size(&self) -> Vec<usize> {
        self.net.borrow().get_tensor(self.id).expect("").size()
    }
    pub fn numel(&self) -> usize {
        self.net.borrow().get_tensor(self.id).expect("").numel()
    }
    pub fn get_f32(&self, o: &[usize]) -> f32 {
        self.net.borrow().get_tensor(self.id).expect("").get_f32(o)
    }
    pub fn set_f32(&mut self, o: &[usize], v: f32) {
        self.net.borrow().get_tensor(self.id).expect("").set_f32(o, v);
    }
    pub fn get_f64(&self, o: &[usize]) -> f64 {
        self.net.borrow().get_tensor(self.id).expect("").get_f64(o)
    }
    pub fn set_f64(&mut self, o: &[usize], v: f64) {
        self.net.borrow().get_tensor(self.id).expect("").set_f64(o, v);
    }

    pub fn fill(size: &[usize], fill_value: Rc<RefCell<VarInner>>) -> VarInner {
        let mut net = Net::new();
        let tensor = Tensor::fill(size, &fill_value.borrow().val());
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    pub fn fill_f32(size: &[usize], fill_value: f32) -> VarInner {
        let mut net = Net::new();
        let tensor = Tensor::fill_f32(size, fill_value);
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    pub fn fill_f64(size: &[usize], fill_value: f64) -> VarInner {
        let mut net = Net::new();
        let tensor = Tensor::fill_f64(size, fill_value);
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
            net: Rc::new(RefCell::new(net)),
        }
    }
    delegate_new_inner_op!(zeros, dim: &[usize]);
    delegate_new_inner_op!(ones, dim: &[usize]);
    //delegate_new_inner_op!(arange, end: usize);
    //delegate_new_inner_op!(range, start: f32, end: f32, step: Option<f32>);
    //delegate_new_inner_op!(linspace, start: f32, end: f32, steps: usize);
    //delegate_new_inner_op!(logspace, start: f32, end: f32, steps: usize, base: f32);
    delegate_new_inner_op!(eye, n: usize, m: usize);
    delegate_new_inner_op!(empty, dim: &[usize]);

    pub fn from_record_f32(&self, row: usize, record: &[f32]) {
        self.val().from_record_f32(row, record).expect("");
    }
    pub fn from_record_f64(&self, row: usize, record: &[f64]) {
        self.val().from_record_f64(row, record).expect("");
    }
    

    // rand
    delegate_new_inner_op!(rand_usize,
                           rng: &mut StdRng,
                           dim: &[usize],
                           left: usize, right: usize);
    delegate_new_inner_op!(normal_f64,
                           rng: &mut StdRng,
                           dim: &[usize],
                           mean: f64, std: f64);
    delegate_new_inner_op!(normal_f32,
                           rng: &mut StdRng,
                           dim: &[usize],
                           mean: f32, std: f32);
    delegate_new_inner_op!(uniform_f64,
                           rng: &mut StdRng,
                           dim: &[usize],
                           from: f64, to: f64);
    delegate_new_inner_op!(uniform_f32,
                           rng: &mut StdRng,
                           dim: &[usize],
                           from: f32, to: f32);
    

    // get and set.
    /// This is a ref. Clone it to cut the connection.
    pub(crate) fn val(&self) -> Tensor {
        self.net.borrow().get_tensor(self.id).unwrap()
    }
    pub(crate) fn set_val(&mut self, val: Tensor) {
        self.net.borrow_mut().set_tensor(self.id, val).expect("");
    }
    pub fn set(&mut self, o: &VarInner) {
        self.set_val(o.val())
    }

    pub fn grad(&self) -> Result<VarInner, AutoDiffError> {
        Ok(VarInner::new_tensor(self.net.borrow().get_grad(self.id)?))
    }

    /// backward pass.
    pub fn bp(&self) -> Result<(), AutoDiffError> {
        let mut job = BTreeMap::new();
        job.insert(self.id, Tensor::ones_like(&self.val()));
        self.net.borrow_mut().bptt(&job);
        
        Ok(())
    }

    /// Update,
    pub fn step(&self, opt: &mut dyn Optimizer) -> Result<(), AutoDiffError> {
        opt.step(self.net.clone());
        Ok(())
    }

    pub fn rerun(&self) -> Result<(), AutoDiffError> {
        let mut all_input = Vec::new();
        for i in &self.net.borrow().get_input_edge_data() {
            all_input.push(*i);
        }
        self.net.borrow_mut().eval(&all_input).expect("");
        Ok(())
    }

    pub(crate) fn set_grad(&mut self, use_gradient: bool) {
        self.need_grad = use_gradient;
    }

    pub(crate) fn reset_net(&mut self) {
        let value = self.val();
        let mut net = Net::new();
        let id = net.add_tensor(value);
        self.id = id;
        self.net = Rc::new(RefCell::new(net));
    }

    /// used in OpCall trait implementation.
    pub(crate) fn called_with(&self, op: Op,
                              others: &[Rc<RefCell<VarInner>>])
                              -> Result<Vec<VarInner>, AutoDiffError> {
        if self.need_grad {
            // TODO there may the same net among others.
            for item in others.iter().map(|x| x.clone()) {
                if !Rc::ptr_eq(&self.net, &item.borrow().net) {
                    let other_key = self.net.borrow_mut().append(
                        &mut item.borrow().net.borrow_mut(), &[item.borrow().id])?[0];
            
                    item.borrow_mut().net = self.net.clone();
                    item.borrow_mut().id = other_key;
                }
            }
            
            let mut input_id = vec![self.id];
            let mut inputs = vec![self.net.borrow().get_tensor(self.id)?];
            for i in others {
                input_id.push(i.borrow().id);
                inputs.push(self.net.borrow().get_tensor(i.borrow().id)?);
            }
            
            let mut output_id = vec![];
            let mut outputs = Vec::new();
            let mut ret = Vec::new();
            for _ in 0..op.get_output_size() {
                let new_output = VarInner::new_net_tensor(self.net.clone(),
                                                          self.need_grad,
                                                          Tensor::new());
                output_id.push(new_output.id);
                outputs.push(self.net.borrow().get_tensor(new_output.id)?);
                ret.push(new_output);
            }
            
            op.apply(&inputs, &outputs);
            let opid = self.net.borrow_mut().add_op(op);
            
            self.net.borrow_mut().connect(&input_id,
                                          opid,
                                          &output_id);
            
            Ok(ret)    
        } else {
            let mut inputs = vec![self.net.borrow().get_tensor(self.id)?];
            for i in others {
                inputs.push(i.borrow().net.borrow().get_tensor(i.borrow().id)?);
            }
            
            let mut ret = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..op.get_output_size() {
                let new_output = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())),
                                                          self.need_grad,
                                                          Tensor::new());
                outputs.push(new_output.net.borrow().get_tensor(new_output.id)?);
                ret.push(new_output);
            }
            
            op.apply(&inputs, &outputs);

            Ok(ret)
        }
    }

    // 2-in-1 ops
    var_inner_2_to_1!(add, Add);
    var_inner_2_to_1!(sub, Sub);
    var_inner_2_to_1!(mul, Mul);
    var_inner_2_to_1!(div, Div);
    var_inner_2_to_1!(matmul, Matmul);

    // nonlinear
    pub fn elu(&self, alpha: VarInner) -> Result<VarInner, AutoDiffError> {
        let new_one = ELU::new(alpha.val());
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))            
    }
    var_inner_1_to_1!(relu, ReLU);
    var_inner_1_to_1!(sigmoid, Sigmoid);
    
    // loss
    var_inner_2_to_1!(mse_loss, MSELoss);
    var_inner_2_to_1!(bce_with_logits_loss, BCEWithLogitsLoss);
    var_inner_2_to_1!(cross_entropy_loss, CrossEntropyLoss);

    // element ops
    var_inner_1_to_1!(abs, Abs);
    var_inner_1_to_1!(acos, Acos);
    var_inner_1_to_1!(asin, Asin);
    var_inner_1_to_1!(atan, Atan);
    var_inner_1_to_1!(ceil, Ceil);
    var_inner_1_to_1!(cos, Cos);
    var_inner_1_to_1!(cosh, Cosh);
    var_inner_1_to_1!(exp, Exp);
    var_inner_1_to_1!(expm1, Expm1);
    var_inner_1_to_1!(floor, Floor);
    var_inner_1_to_1!(frac, Frac);
    var_inner_1_to_1!(log, Log);
    var_inner_1_to_1!(log10, Log10);
    var_inner_1_to_1!(log1p, Log1p);
    var_inner_1_to_1!(log1pexp, Log1pexp);
    var_inner_1_to_1!(log2, Log2);
    var_inner_1_to_1!(neg, Neg);
    var_inner_1_to_1!(reciprocal, Reciprocal);
    var_inner_1_to_1!(round, Round);
    var_inner_1_to_1!(rsqrt, Rsqrt);
    var_inner_1_to_1!(sign, Sign);
    var_inner_1_to_1!(sin, Sin);
    var_inner_1_to_1!(sinh, Sinh);
    var_inner_1_to_1!(sqrt, Sqrt);
    var_inner_1_to_1!(tan, Tan);
    var_inner_1_to_1!(tanh, Tanh);
    var_inner_1_to_1!(trunc, Trunc);

    // comparison
    var_inner_2_to_1!(max_pair, MaxPair);
    var_inner_2_to_1!(min_pair, MinPair);
    var_inner_1_to_1_with_para!(arg_sort, ArgSort,
                                dim: usize, descending: bool);
    var_inner_2_to_1!(eq_elem, EqElem);
    var_inner_2_to_1!(equal, Equal);
    var_inner_2_to_1!(ge, Ge);
    var_inner_2_to_1!(gt, Gt);
    var_inner_2_to_1!(le, Le);
    var_inner_2_to_1!(lt, Lt);
    var_inner_2_to_1!(ne, Ne);

    // index and slicing
    var_inner_more_to_1_with_para!(cat, Cat, dim: usize);
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<VarInner>, AutoDiffError> {
        let new_one = Chunk::new(chunks, dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let result = self.called_with(op, &Vec::new())?;
        Ok(result)
    }
    pub fn conditional_select(&self, x: Rc<RefCell<VarInner>>, y: Rc<RefCell<VarInner>>) -> Result<VarInner, AutoDiffError> {
        let new_one = ConditionalSelect::new();
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut inputs = Vec::new();
        inputs.push(x.clone());
        inputs.push(y.clone());
        let mut result = self.called_with(op, &inputs)?;
        Ok(result.remove(0))
    }
    pub fn gather(&self, dim: usize, index: Rc<RefCell<VarInner>>) -> Result<VarInner, AutoDiffError> {
        let new_one = Gather::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut inputs = Vec::new();
        inputs.push(index.clone());
        let mut result = self.called_with(op, &inputs)?;
        Ok(result.remove(0))
    }
    pub fn index_select(&self, dim: usize, index: Rc<RefCell<VarInner>>) -> Result<VarInner, AutoDiffError> {
        let new_one = IndexSelect::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut inputs = Vec::new();
        inputs.push(index.clone());
        let mut result = self.called_with(op, &inputs)?;
        Ok(result.remove(0))
    }
    pub fn index_exclude(&self, dim: usize, index: Rc<RefCell<VarInner>>) -> Result<VarInner, AutoDiffError> {
        let new_one = IndexExclude::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut inputs = Vec::new();
        inputs.push(index.clone());
        let mut result = self.called_with(op, &inputs)?;
        Ok(result.remove(0))
    }
    pub fn permute(&self, dim: &[usize]) -> Result<VarInner, AutoDiffError> {
        let new_one = Permute::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    pub fn repeat(&self, dim: &[usize]) -> Result<VarInner, AutoDiffError> {
        let new_one = Repeat::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    pub fn reshape(&self, new_shape: &[usize]) -> Result<VarInner, AutoDiffError> {
        let new_one = Reshape::new(new_shape);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    pub fn split(&self, sections: &[usize], dim: usize) -> Result<Vec<VarInner>, AutoDiffError> {
        let new_one = Split::new(sections, dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let result = self.called_with(op, &Vec::new())?;
        Ok(result)
    }
    pub fn squeeze(&self, dim: Option<usize>) -> Result<VarInner, AutoDiffError> {
        let new_one = Squeeze::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    var_inner_1_to_1!(t, T);
    pub fn take(&self, index: &[usize]) -> Result<VarInner, AutoDiffError> {
        let new_one = Take::new(index);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    pub fn unsqueeze(&self, dim: usize) -> Result<VarInner, AutoDiffError> {
        let new_one = Unsqueeze::new(dim);
        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));
        let mut result = self.called_with(op, &[])?;
        Ok(result.remove(0))
    }
    var_inner_more_to_1_with_para!(stack, Stack, dim: usize);

    // linalg
    var_inner_1_to_1!(det, Det);
    var_inner_1_to_1!(inv, Inv);
    var_inner_1_to_1!(normalize_unit, NormalizeUnit);

    // reduction
    var_inner_1_to_1_with_para!(argmax, Argmax, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(argmin, Argmin, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(logsumexp, Logsumexp, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(mean, Mean, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(prod, Prod, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(std, Std, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(sum, Sum, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(var, Variance, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(max, Max, dim: Option<&[usize]>, keepdim: bool);
    var_inner_1_to_1_with_para!(min, Min, dim: Option<&[usize]>, keepdim: bool);

    // images
    var_inner_1_to_1_with_para!(get_patch, GetPatch, range: &[(usize, usize)], step: Option<&[usize]>);
    var_inner_1_to_1_with_para!(view, View, new_shape: &[usize]);

    pub fn dump_net(&self) -> Rc<RefCell<Net>> {
        self.net.clone()
    }
}

impl PartialEq for VarInner {
    fn eq(&self, other: &Self) -> bool {
        self.val().eq(&other.val())
    }
}

impl Eq for VarInner {}

impl fmt::Display for VarInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.val())
    }
}

impl fmt::Debug for VarInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.val())
    }
}

impl Clone for VarInner {
    fn clone(&self) -> Self {
        let val = self.val().clone();
        let mut ret = VarInner::new(&[], &[]);
        ret.set_val(val);
        ret.need_grad = self.need_grad;
        ret
    }
}


