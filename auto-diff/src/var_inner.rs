use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::collections::BTreeMap;
use ::rand::prelude::StdRng;

use tensor_rs::tensor::{Tensor};
use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};
use crate::op::{Op, OpTrait,
                Add, Sub, Mul, Div, Matmul,
                MSELoss,
                Abs, Acos, Asin, Atan, Ceil, Cos, Cosh, Exp, Expm1, Floor, Frac, Log, Log10, Log1p, Log1pexp, Log2, Neg, Reciprocal, Round, Rsqrt, Sign, Sin, Sinh, Sqrt, Tan, Tanh, Trunc,
                Cat,
                Det,
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
                
                let ret = VarInner::new_net_tensor(self.net.clone(), Tensor::new());
                
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy()],
                         &[self.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                let op = Op::new(Rc::new(RefCell::new(Box::new(op))));
                let opid = self.net.borrow_mut().add_op(op);
                
                self.net.borrow_mut().connect(&[self.id],
                                              opid, &[ret.id]);
                
                Ok(ret)
            } else {
                let ret = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())), Tensor::new());
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
                
                let ret = VarInner::new_net_tensor(self.net.clone(), Tensor::new());
                
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
                let ret = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())), Tensor::new());
                let op = $b::new();
                op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy(),
                           other.net.borrow().get_tensor(other.id)?.ref_copy()],
                         &[ret.net.borrow().get_tensor(ret.id)?.ref_copy()]);
                Ok(ret)
            }
            
        }
    }
}

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

pub struct VarInner {
    id: GenKey,
    need_grad: bool,
    net: Rc<RefCell<Net>>,
}

impl VarInner {

    // create functions.
    pub fn new(input: &[f64], dim: &[usize]) -> VarInner {
        let mut net = Net::new();
        
        #[cfg(feature = "use-f64")]
        let tensor = Tensor::from_vec_f64(input, dim);
        #[cfg(feature = "use-f32")]
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
                                 tensor: Tensor) -> VarInner {
        let id = net.borrow_mut().add_tensor(tensor);
        VarInner {
            id,
            need_grad: true,
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

    //delegate_new_inner_op!(fill, dim: &[usize], fill_value: &);
    delegate_new_inner_op!(zeros, dim: &[usize]);
    delegate_new_inner_op!(ones, dim: &[usize]);
    //delegate_new_inner_op!(arange, end: usize);
    //delegate_new_inner_op!(range, start: f32, end: f32, step: Option<f32>);
    //delegate_new_inner_op!(linspace, start: f32, end: f32, steps: usize);
    //delegate_new_inner_op!(logspace, start: f32, end: f32, steps: usize, base: f32);
    delegate_new_inner_op!(eye, n: usize, m: usize);
    delegate_new_inner_op!(empty, dim: &[usize]);

    


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
        self.net.borrow_mut().eval().expect("");
        Ok(())
    }

    pub(crate) fn set_grad(&mut self, use_gradient: bool) {
        self.need_grad = use_gradient;
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
                let new_output = VarInner::new_net_tensor(self.net.clone(), Tensor::new());
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
                let new_output = VarInner::new_net_tensor(Rc::new(RefCell::new(Net::new())), Tensor::new());
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
    
    var_inner_2_to_1!(mse_loss, MSELoss);

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

    // index and slicing
    var_inner_more_to_1_with_para!(cat, Cat, dim: usize);

    // linalg
    var_inner_1_to_1!(det, Det);
    
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


