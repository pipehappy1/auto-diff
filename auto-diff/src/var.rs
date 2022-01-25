use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::ops;

use tensor_rs::tensor::{Tensor, PaddingMode};

use crate::compute_graph::{Net};
use crate::collection::generational_index::{GenKey};


pub struct Var {
    id: GenKey,    
    net: Rc<RefCell<Net>>,
}

impl Var {

    pub fn grad() -> Var {
        unimplemented!();
    }

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
    
    pub fn eye(n: usize, m: usize) -> Var {
        let mut net = Net::new();
        let tensor = Tensor::eye(n, m);
        let id = net.add_tensor(tensor);
        Var {
            id,
            net: Rc::new(RefCell::new(net)),
        }
    }

    pub fn matmul(&self, other: &Var) -> Var {
        unimplemented!();
    }

    pub fn bp(&self) -> Result<(), &str> {
        unimplemented!();
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
    fn add() {
        let a = Var::eye(2, 2);
        let b = Var::new(&[1., 2., 3., 4.], &[2, 2]);
        let c = a.matmul(&b);
        c.bp();
        assert_eq!(a.grad(), Var::new(&[1., 0., 0., 1.], &[2, 2]));
        assert_eq!(b.grad(), Var::new(&[1., 2., 3., 4.], &[2, 2]));
    }
}
