use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::ops;
use std::collections::BTreeMap;
use ::rand::prelude::StdRng;

use tensor_rs::tensor::{Tensor, PaddingMode};
use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};
use crate::op::{Op, OpTrait,
                Add, Sub, Mul, Div,
                Linear,
                MSELoss,
};
use crate::err::AutoDiffError;
use crate::optim::Optimizer;


macro_rules! var_2_to_1 {
    ($a:ident) => {
        pub fn $a(&self, other: &Var) -> Result<Var, AutoDiffError> {
            Ok(Var {
                var: Rc::new(RefCell::new(self.var.borrow().$a(&mut other.var.borrow_mut())?))})
        }
    }
}

macro_rules! delegate_new_op {
    ($a:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a($( $arg_name : $ArgTy ),*) -> Var {
            Var {
                var: Rc::new(RefCell::new(VarInner::$a($( $arg_name ),*)))
            }
        }
    }
}

pub struct Var {
    var: Rc<RefCell<VarInner>>
}
impl Var {
    #[cfg(feature = "use-f64")]
    pub fn new(input: &[f64], dim: &[usize]) -> Var {
        Var {
            var: Rc::new(RefCell::new(VarInner::new(input, dim)))
        }
    }
    #[cfg(feature = "use-f32")]
    pub fn new(input: &[f32], dim: &[usize]) -> Var {
        Var {
            var: Rc::new(RefCell::new(VarInner::new(input, dim)))
        }
    }

    delegate_new_op!(ones, dim: &[usize]);
    delegate_new_op!(eye, n: usize, m: usize);

    // rand
    delegate_new_op!(rand_usize,
                     rng: &mut StdRng,
                     dim: &[usize],
                     left: usize, right: usize);
    
    delegate_new_op!(normal_f64,
                     rng: &mut StdRng,
                     dim: &[usize],
                     mean: f64, std: f64);
    delegate_new_op!(normal_f32,
                     rng: &mut StdRng,
                     dim: &[usize],
                     mean: f32, std: f32);
    #[cfg(feature = "use-f32")]
    pub fn normal(rng: &mut StdRng,
                  dim: &[usize],
                  mean: f32, std: f32) -> Var {
        Self::normal_f32(rng, dim, mean, std)
    }
    #[cfg(feature = "use-f64")]
    pub fn normal(rng: &mut StdRng,
                  dim: &[usize],
                  mean: f64, std: f64) -> Var {
        Self::normal_f64(rng, dim, mean, std)
    }
    
    delegate_new_op!(uniform_f64,
                     rng: &mut StdRng,
                     dim: &[usize],
                     from: f64, to: f64);
    delegate_new_op!(uniform_f32,
                     rng: &mut StdRng,
                     dim: &[usize],
                     from: f32, to: f32);
    #[cfg(feature = "use-f32")]
    pub fn uniform(rng: &mut StdRng,
                   dim: &[usize],
                   from: f32, to: f32) -> Var {
        Self::uniform_f32(rng, dim, from, to)
    }
    #[cfg(feature = "use-f64")]
    pub fn uniform(rng: &mut StdRng,
                   dim: &[usize],
                   from: f64, to: f64) -> Var {
        Self::uniform_f64(rng, dim, from, to)
    }


    pub fn grad(&self) -> Result<Var, AutoDiffError> {
        Ok(Var {
            var: Rc::new(RefCell::new(self.var.borrow().grad()?))
        })
    }

    pub fn bp(&self) -> Result<(), AutoDiffError> {
        self.var.borrow().bp()?;

        Ok(())
    }

    var_2_to_1!(add);
    var_2_to_1!(sub);
    var_2_to_1!(mul);
    var_2_to_1!(div);
    
    var_2_to_1!(mse_loss);



    // innternal use
    pub(crate) fn val(&self) -> Tensor {
        self.var.borrow().val()
    }

    pub(crate) fn called_with(&self, op: Op,
                              others: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let mut refs: Vec<Rc<RefCell<VarInner>>> = others.iter().map(|x| x.var.clone()).collect();
        let mut var_inners = self.var.borrow().called_with(op, &mut refs)?;
        let ret: Vec<Var> = var_inners.drain(..).map(|x| Var {
            var: Rc::new(RefCell::new(x))
        }).collect();
        Ok(ret)
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.var.borrow().val().eq(&other.var.borrow().val())
    }
}

impl Eq for Var {}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.var.borrow().val())
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.var.borrow().val())
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        Var {
            var: Rc::new(RefCell::new(self.var.borrow().clone()))
        }
    }
}



macro_rules! var_inner_2_to_1 {
    ($a:ident, $b:ident) => {
        pub fn $a(&self, other: &mut VarInner) -> Result<VarInner, AutoDiffError> {
            if !Rc::ptr_eq(&self.net, &other.net) {
                let other_key = self.net.borrow_mut().append(
                    &mut other.net.borrow_mut(), &[other.id])?[0];

                other.net = self.net.clone();
                other.id = other_key;
            }

            let ret = VarInner::new_net_tensor(self.net.clone(), Tensor::new());

            let mut op = $b::new();
            op.apply(&[self.net.borrow().get_tensor(self.id)?.ref_copy(),
                       self.net.borrow().get_tensor(other.id)?.ref_copy()],
                     &[self.net.borrow().get_tensor(ret.id)?.ref_copy()]);
            let op = Op::new(Rc::new(RefCell::new(Box::new(op))));
            let opid = self.net.borrow_mut().add_op(op);

            self.net.borrow_mut().connect(&[self.id, other.id],
                                          opid, &[ret.id]);

            Ok(ret)
        }
    }
}

macro_rules! delegate_new_inner_op {
    ($a:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a($( $arg_name : $ArgTy ),*) -> VarInner {
            let mut net = Net::new();
            let tensor = Tensor::$a($( $arg_name ),*);
            let id = net.add_tensor(tensor);
            VarInner {
                id,
                net: Rc::new(RefCell::new(net)),
            }
        }
    }
}

pub struct VarInner {
    id: GenKey,    
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
            net: Rc::new(RefCell::new(net)),
        }
    }

    /// Create a new var with an existing net and value.
    pub(crate) fn new_net_tensor(net: Rc<RefCell<Net>>,
                                 tensor: Tensor) -> VarInner {
        let id = net.borrow_mut().add_tensor(tensor);
        VarInner {
            id,
            net
        }
    }

    pub(crate) fn new_tensor(tensor: Tensor) -> VarInner {
        let mut net = Net::new();
        let id = net.add_tensor(tensor);
        VarInner {
            id,
            net: Rc::new(RefCell::new(net)),
        }
    }

    delegate_new_inner_op!(ones, dim: &[usize]);
    delegate_new_inner_op!(eye, n: usize, m: usize);

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
    

    pub(crate) fn called_with(&self, op: Op,
                              others: &[Rc<RefCell<VarInner>>]) -> Result<Vec<VarInner>, AutoDiffError> {
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
        for i in 0..op.get_output_size() {
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
    }

    var_inner_2_to_1!(add, Add);
    var_inner_2_to_1!(sub, Sub);
    var_inner_2_to_1!(mul, Mul);
    var_inner_2_to_1!(div, Div);
    
    var_inner_2_to_1!(mse_loss, MSELoss);
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
        ret
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OpCall;

    #[test]
    fn mul() {
        let a = Var::new(&[2., 3., 4., 5.], &[2, 2]);
        let b = Var::new(&[1., 2., 3., 4.], &[2, 2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c, Var::new(&[2., 6., 12., 20.], &[2, 2]));
        c.bp().unwrap();
        assert_eq!(a.grad().unwrap(), Var::new(&[1., 2., 3., 4.], &[2, 2]));
        assert_eq!(b.grad().unwrap(), Var::new(&[2., 3., 4., 5.], &[2, 2]));
    }

    #[test]
    fn test_mul_repeat_vars() {
        let a = Var::new(&[2., 3., 4., 5.], &[2, 2]);
        let b = Var::new(&[1., 2., 3., 4.], &[2, 2]);
        let c = a.mul(&b).unwrap();
        let d = c.mul(&b).unwrap(); // repeat vars
        assert_eq!(d, Var::new(&[2., 12., 36., 80.], &[2, 2]));
    }

    #[test]
    fn test_add_in_fn() {
        let a = Var::new(&[2., 3., 4., 5.], &[2, 2]);
        let b = Var::new(&[1., 2., 3., 4.], &[2, 2]);
    
        fn my_mul(a: &Var, b: &Var) -> Var {
            a.mul(b).unwrap()
        }
        let c = my_mul(&a, &b);
        assert_eq!(c, Var::new(&[2., 6., 12., 20.], &[2, 2]));
    }

    #[test]
    fn test_op_mse() {
        let a = Var::new(&[1., 2., 3., 4., 5., 6.,], &[3, 2]);
        let b = Var::new(&[2., 3., 4., 5., 6., 7.,], &[3, 2]);
        let c = a.mse_loss(&b).unwrap();
        assert_eq!(c , Var::new(&[1., ], &vec![1]));
    }

    #[test]
    fn test_linear() {
        let mut op1 = Linear::new(Some(2), Some(5), true);
        op1.set_weight(Var::new(&[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.], &[2, 5]));
        op1.set_bias(Var::new(&[1.,2.,3.,4.,5.], &[5]));
        let input = Var::ones(&[3,2]);
        let output = op1.call(&[&input]).unwrap().pop().unwrap();
        assert_eq!(output, Var::new(&[8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0],
                                    &vec![3, 5]));
    }
}
