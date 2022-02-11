use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use ::rand::prelude::StdRng;

use tensor_rs::tensor::{Tensor};
use crate::op::{Op};
use crate::err::AutoDiffError;
use crate::optim::Optimizer;
use crate::var_inner::VarInner;
use crate::compute_graph::{Net};


macro_rules! var_1_to_1 {
    ($a:ident) => {
        pub fn $a(&self) -> Result<Var, AutoDiffError> {
            Ok(Var {
                var: Rc::new(RefCell::new(self.var.borrow().$a()?))})
        }
    }
}

macro_rules! var_2_to_1 {
    ($a:ident) => {
        pub fn $a(&self, other: &Var) -> Result<Var, AutoDiffError> {
            Ok(Var {
                var: Rc::new(RefCell::new(self.var.borrow().$a(&mut other.var.borrow_mut())?))})
        }
    }
}

macro_rules! var_more_to_1_with_para {
    ($a:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a(&self, other: &[Var], $( $arg_name : $ArgTy ),*) -> Result<Var, AutoDiffError> {
            let mut other_input = Vec::new();
            for i in other {
                other_input.push(i.var.clone());
            }
            Ok(Var {
                var: Rc::new(RefCell::new(self.var.borrow().$a(&other_input, $( $arg_name ),*)?))})
        }
    }
}

macro_rules! var_1_to_1_with_para {
    ($a:ident, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub fn $a(&self, $( $arg_name : $ArgTy ),*) -> Result<Var, AutoDiffError> {
            Ok(Var {
                var: Rc::new(RefCell::new(self.var.borrow().$a($( $arg_name ),*)?))})
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

/// [Var] can be thought as the value it holds plus a link
/// to the computation graph.
/// Majority of operators are methods on [Var].
pub struct Var {
    var: Rc<RefCell<VarInner>>
}
impl Var {
    #[cfg(feature = "use-f64")]
    pub fn new(input: &[f64], dim: &[usize]) -> Var {
        Var::new_f64(input, dim)
    }
    #[cfg(feature = "use-f32")]
    pub fn new(input: &[f32], dim: &[usize]) -> Var {
        Var::new_f32(input, dim)
    }
    pub fn new_f64(input: &[f64], dim: &[usize]) -> Var {
        Var {
            var: Rc::new(RefCell::new(VarInner::new_f64(input, dim)))
        }
    }
    pub fn new_f32(input: &[f32], dim: &[usize]) -> Var {
        Var {
            var: Rc::new(RefCell::new(VarInner::new_f32(input, dim)))
        }
    }


    pub fn size(&self) -> Vec<usize> {
        self.var.borrow().size()
    }
    pub fn numel(&self) -> usize {
        self.var.borrow().numel()
    }
    pub fn get_f32(&self, o: &[usize]) -> f32 {
        self.var.borrow().get_f32(o)
    }
    pub fn set_f32(&self, o: &[usize], v: f32) {
        self.var.borrow_mut().set_f32(o, v);
    }
    pub fn get_f64(&self, o: &[usize]) -> f64 {
        self.var.borrow().get_f64(o)
    }
    pub fn set_f64(&self, o: &[usize], v: f64) {
        self.var.borrow_mut().set_f64(o, v);
    }

    pub fn set(&self, o: &Var) {
        self.var.borrow_mut().set(&mut o.var.borrow());
    }

    //delegate_new_inner_op!(fill, dim: &[usize], fill_value: &);
    delegate_new_op!(fill_f32, size: &[usize], fill_value: f32);
    delegate_new_op!(fill_f64, size: &[usize], fill_value: f64);
    delegate_new_op!(zeros, dim: &[usize]);
    delegate_new_op!(ones, dim: &[usize]);
    //delegate_new_inner_op!(arange, end: usize);
    //delegate_new_inner_op!(range, start: f32, end: f32, step: Option<f32>);
    //delegate_new_inner_op!(linspace, start: f32, end: f32, steps: usize);
    //delegate_new_inner_op!(logspace, start: f32, end: f32, steps: usize, base: f32);
    delegate_new_op!(eye, n: usize, m: usize);
    delegate_new_op!(empty, dim: &[usize]);
    
    /// Fill row by row.
    pub fn from_record_f32(&self, row: usize, record: &[f32]) {
        self.var.borrow().from_record_f32(row, record)
    }
    pub fn from_record_f64(&self, row: usize, record: &[f64]) {
        self.var.borrow().from_record_f64(row, record)
    }
    
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
    var_2_to_1!(matmul);

    // nonlinear
    var_1_to_1!(relu);
    var_1_to_1!(sigmoid);

    // loss
    var_2_to_1!(mse_loss);
    var_2_to_1!(bce_with_logits_loss);
    var_2_to_1!(cross_entropy_loss);

    //elementwise op
    var_1_to_1!(abs);
    var_1_to_1!(acos);
    var_1_to_1!(asin);
    var_1_to_1!(atan);
    var_1_to_1!(ceil);
    var_1_to_1!(cos);
    var_1_to_1!(cosh);
    var_1_to_1!(exp);
    var_1_to_1!(expm1);
    var_1_to_1!(floor);
    var_1_to_1!(frac);
    var_1_to_1!(log);
    var_1_to_1!(log10);
    var_1_to_1!(log1p);
    var_1_to_1!(log1pexp);
    var_1_to_1!(log2);
    var_1_to_1!(neg);
    var_1_to_1!(reciprocal);
    var_1_to_1!(round);
    var_1_to_1!(rsqrt);
    var_1_to_1!(sign);
    var_1_to_1!(sin);
    var_1_to_1!(sinh);
    var_1_to_1!(sqrt);
    var_1_to_1!(tan);
    var_1_to_1!(tanh);
    var_1_to_1!(trunc);

    // index and slicing
    var_more_to_1_with_para!(cat, dim: usize);
    pub fn chunk(&self, chunks: usize, dim: usize)
                 -> Result<Vec<Var>, AutoDiffError> {
        let mut result = self.var.borrow().chunk(chunks, dim)?;
        let mut ret = Vec::new();
        for i in result.drain(..) {
            ret.push(Var {
                var: Rc::new(RefCell::new(i)),
            });
        }
        Ok(ret)
    }
    pub fn conditional_select(&self, x: &Var, y: &Var)
                              -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().conditional_select(x.var.clone(),
                                                          y.var.clone())?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn gather(&self, dim: usize, index: Var)
                  -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().gather(dim, index.var.clone())?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn index_select(&self, dim: usize,
                        index: Var)
                        -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().index_select(
            dim, index.var.clone())?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn index_exclude(&self, dim: usize,
                        index: Var)
                        -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().index_exclude(
            dim, index.var.clone())?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn permute(&self, dim: &[usize])
                   -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().permute(dim)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn repeat(&self, dim: &[usize])
                  -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().repeat(dim)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn reshape(&self, new_shape: &[usize])
                  -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().reshape(new_shape)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn split(&self, sections: &[usize], dim: usize)
                  -> Result<Vec<Var>, AutoDiffError> {
        let mut result = self.var.borrow().split(sections, dim)?;
        let mut ret = Vec::new();
        for i in result.drain(..) {
            ret.push(Var {
                var: Rc::new(RefCell::new(i)),
            });
        }
        Ok(ret)
    }
    pub fn squeeze(&self, dim: Option<usize>)
                   -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().squeeze(dim)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    var_1_to_1!(t);
    pub fn take(&self, index: &[usize])
                -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().take(index)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    pub fn unsqueeze(&self, dim: usize)
                   -> Result<Var, AutoDiffError> {
        let result = self.var.borrow().unsqueeze(dim)?;
        Ok(Var {
            var: Rc::new(RefCell::new(result)),
        })
    }
    var_more_to_1_with_para!(stack, dim: usize);

    // linalg
    var_1_to_1!(det);
    var_1_to_1!(inv);
    var_1_to_1!(normalize_unit);

    // reduction
    var_1_to_1_with_para!(argmax, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(argmin, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(logsumexp, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(mean, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(prod, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(std, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(sum, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(var, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(max, dim: Option<&[usize]>, keepdim: bool);
    var_1_to_1_with_para!(min, dim: Option<&[usize]>, keepdim: bool);


    // innternal use
    pub(crate) fn val(&self) -> Tensor {
        self.var.borrow().val()
    }

    /// Use gradient or not, default is to use.
    pub fn set_grad(&self, use_gradient: bool) {
        self.var.borrow_mut().set_grad(use_gradient);
    }

    /// Reset net in the background.
    pub fn reset_net(&self) {
        self.var.borrow_mut().reset_net();
    }

    /// The current gradient for the Var.
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

    pub fn dump_net(&self) -> Rc<RefCell<Net>> {
        self.var.borrow().dump_net()
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
        use crate::op::Linear;
        
        let mut op1 = Linear::new(Some(2), Some(5), true);
        op1.set_weight(Var::new(&[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.], &[2, 5]));
        op1.set_bias(Var::new(&[1.,2.,3.,4.,5.], &[5]));
        let input = Var::ones(&[3,2]);
        let output = op1.call(&[&input]).unwrap().pop().unwrap();
        assert_eq!(output, Var::new(&[8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0],
                                    &vec![3, 5]));
    }
}
