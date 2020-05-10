use auto_diff::var::*;
use auto_diff::tensor::*;
use auto_diff::collection::generational_index::*;
use auto_diff::op::{Op, Linear};
use auto_diff::rand::*;



#[test]
fn test_add_2vars() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();
    assert_eq!(*a._id(), NetIndex::new(0, 0));
    assert_eq!(*b._id(), NetIndex::new(1, 0));
    let c = a.add(&b);
    a.set(Tensor::new());
    b.set(Tensor::new());
    m.eval();
    assert_eq!(NetIndex::new(2, 0), *c._id());
}

#[test]
fn test_add_repeat_vars() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();
    assert_eq!(*a._id(), NetIndex::new(0, 0));
    assert_eq!(*b._id(), NetIndex::new(1, 0));
    let c = a.add(&b);
    let d = c.add(&b); // repeat vars
    a.set(Tensor::new());
    b.set(Tensor::new());
    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    println!("{}", d);
    m.eval();
    println!("{}", d);
}

#[test]
fn test_add_in_fn() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();

    fn my_add(a: &Var, b: &Var) -> Var {
        a.add(b)
    }
    let c = my_add(&a, &b);
    a.set(Tensor::new());
    b.set(Tensor::new());
    m.eval();
}

#[test]
fn test_op_mse() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();

    let c = mseloss(&a, &b);
    a.set(Tensor::from_vec_f32(&vec![1., 2., 3., 4., 5., 6.,], &vec![3, 2]));
    b.set(Tensor::from_vec_f32(&vec![2., 3., 4., 5., 6., 7.,], &vec![3, 2]));
    m.forward();
    println!("test_op_mse, c: {}", c);
    
    assert_eq!(c.get() , Tensor::from_vec_f32(&vec![1., ], &vec![1]));
    println!("hehe");
}
