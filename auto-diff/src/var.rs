use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use ::rand::prelude::StdRng;

use tensor_rs::tensor::{Tensor};
use crate::op::{Op
};
use crate::err::AutoDiffError;
use crate::optim::Optimizer;
use crate::var_inner::VarInner;


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


    var_2_to_1!(add);
    var_2_to_1!(sub);
    var_2_to_1!(mul);
    var_2_to_1!(div);
    
    var_2_to_1!(mse_loss);


    // innternal use
    pub(crate) fn val(&self) -> Tensor {
        self.var.borrow().val()
    }

        /// Use gradient or not, default is to use.
    pub fn set_grad(&self, use_gradient: bool) {
        self.var.borrow_mut().set_grad(use_gradient);
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

    pub fn step(&self, opt: &mut dyn Optimizer) -> Result<(), AutoDiffError> {
        Ok(self.var.borrow().step(opt)?)
    }

    pub fn rerun(&self) -> Result<(), AutoDiffError> {
        Ok(self.var.borrow().rerun()?)
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
