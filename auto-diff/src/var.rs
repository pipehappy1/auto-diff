use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::ops;
use std::collections::BTreeMap;

use tensor_rs::tensor::{Tensor, PaddingMode};
use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};
use crate::op::{Op, OpTrait, Mul};
use crate::err::AutoDiffError;


pub struct Var1 {
    var: Rc<RefCell<Var>>
}
impl Var1 {
    #[cfg(feature = "use-f64")]
    pub fn new(input: &[f64], dim: &[usize]) -> Var1 {
        Var1 {
            var: Rc::new(RefCell::new(Var::new(input, dim)))
        }
    }

    pub fn grad(&self) -> Result<Var1, AutoDiffError> {
        Ok(Var1 {
            var: Rc::new(RefCell::new(self.var.borrow().grad()?))
        })
    }

    pub fn bp(&self) -> Result<(), AutoDiffError> {
        self.var.borrow().bp()?;

        Ok(())
    }

    pub fn mul(&self, other: &Var1) -> Result<Var1, AutoDiffError> {
        Ok(Var1 {
            var: Rc::new(RefCell::new(self.var.borrow().mul(&mut other.var.borrow_mut())?))})
    }
}

impl PartialEq for Var1 {
    fn eq(&self, other: &Self) -> bool {
        self.var.borrow().val().eq(&other.var.borrow().val())
    }
}

impl Eq for Var1 {}

impl fmt::Display for Var1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.var.borrow().val())
    }
}

impl fmt::Debug for Var1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.var.borrow().val())
    }
}

impl Clone for Var1 {
    fn clone(&self) -> Self {
        Var1 {
            var: Rc::new(RefCell::new(self.var.borrow().clone()))
        }
    }
}


pub struct Var {
    id: GenKey,    
    net: Rc<RefCell<Net>>,
}

impl Var {

    // create functions.
    #[cfg(feature = "use-f64")]
    pub fn new(input: &[f64], dim: &[usize]) -> Var {
        let mut net = Net::new();
        let tensor = Tensor::from_vec_f64(input, dim);
        let id = net.add_tensor(tensor);
        Var {
            id,
            net: Rc::new(RefCell::new(net)),
        }
    }

    /// Create a new var with an existing net and value.
    pub(crate) fn new_net_tensor(net: Rc<RefCell<Net>>,
                                 tensor: Tensor) -> Var {
        let id = net.borrow_mut().add_tensor(tensor);
        Var {
            id,
            net
        }
    }

    pub(crate) fn new_tensor(tensor: Tensor) -> Var {
        let mut net = Net::new();
        let id = net.add_tensor(tensor);
        Var {
            id,
            net: Rc::new(RefCell::new(net)),
        }
    }

    pub fn eye(n: usize, m: usize) -> Var {
        let mut net = Net::new();
        let tensor = Tensor::eye(n, m);
        let id = net.add_tensor(tensor);
        Var {
            id,
            net: Rc::new(RefCell::new(net)),
        }
    }

    // get and set.
    /// This is a ref. Clone it to cut the connection.
    pub(crate) fn val(&self) -> Tensor {
        self.net.borrow().get_tensor(self.id).unwrap()
    }
    pub(crate) fn set_val(&mut self, val: Tensor) {
        self.net.borrow_mut().set_tensor(self.id, val).expect("");
    }

    pub fn grad(&self) -> Result<Var, AutoDiffError> {
        Ok(Var::new_tensor(self.net.borrow().get_grad(self.id)?))
    }

    /// backward pass.
    pub fn bp(&self) -> Result<(), AutoDiffError> {
        let mut job = BTreeMap::new();
        job.insert(self.id, Tensor::ones_like(&self.val()));
        self.net.borrow_mut().bptt(&job);
        
        Ok(())
    }

    pub fn mul(&self, other: &mut Var) -> Result<Var, AutoDiffError> {

        let other_key = self.net.borrow_mut().append(
            &mut other.net.borrow_mut(), &[other.id])?[0];

        other.net = self.net.clone();
        other.id = other_key;

        let mut op = Mul::new();
        let result = op.call(&[&self.net.borrow().get_tensor(self.id)?,
                               &self.net.borrow().get_tensor(other_key)?])?[0].clone();
        let op = Op::new(Box::new(op));
        let opid = self.net.borrow_mut().add_op(op);
        
        let ret = Var::new_net_tensor(self.net.clone(), result);

        self.net.borrow_mut().connect(&[self.id, other_key],
                                      opid, &[ret.id]);

        Ok(ret)
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.val().eq(&other.val())
    }
}

impl Eq for Var {}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.val())
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id: {}", self.id)?;
        write!(f, "tensor: {}", self.val())
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        let val = self.val().clone();
        let mut ret = Var::new(&[], &[]);
        ret.set_val(val);
        ret
    }
}

//macro_rules! typed_tensor_method_single_same_return {
//    ($a:ident, $b:ty) => {
//        pub fn $a(&self) -> $b {
//            match &self {
//                TypedTensor::Typef32(v1) => {v1.$a()},
//                TypedTensor::Typef64(v1) => {v1.$a()},
//                #[cfg(feature = "use-cuda")]
//                TypedTensor::Cudaf32(v1) => {v1.$a()},
//                //_ => {panic!("should have same tensor type!");},
//            }
//        }
//    }
//}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul() {
        let a = Var1::new(&[2., 3., 4., 5.], &[2, 2]);
        let mut b = Var1::new(&[1., 2., 3., 4.], &[2, 2]);
        let c = a.mul(&mut b).unwrap();
        assert_eq!(c, Var1::new(&[2., 6., 12., 20.], &[2, 2]));
        c.bp().unwrap();
        assert_eq!(a.grad().unwrap(), Var1::new(&[1., 2., 3., 4.], &[2, 2]));
        assert_eq!(b.grad().unwrap(), Var1::new(&[2., 3., 4., 5.], &[2, 2]));
    }
}
