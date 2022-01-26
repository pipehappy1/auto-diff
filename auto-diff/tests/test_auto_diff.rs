use auto_diff::var::*;
use tensor_rs::tensor::*;
use auto_diff::op::{Op, Linear};
use auto_diff::rand::*;








//
//#[test]
//fn test_op_mse() {
//    let mut m = Module::new();
//    let a = m.var();
//    let b = m.var();
//
//    let c = mseloss(&a, &b);
//    a.set(Tensor::from_vec_f32(&vec![1., 2., 3., 4., 5., 6.,], &vec![3, 2]));
//    b.set(Tensor::from_vec_f32(&vec![2., 3., 4., 5., 6., 7.,], &vec![3, 2]));
//    m.forward();
//    println!("test_op_mse, c: {}", c);
//    
//    assert_eq!(c.get() , Tensor::from_vec_f32(&vec![1., ], &vec![1]));
//    println!("hehe");
//}
