use auto_diff::var::*;
use auto_diff::tensor::*;
use auto_diff::collection::generational_index::*;
use auto_diff::op::{Op, Linear};
use auto_diff::rand::*;

#[test]
fn test_gen_index() {
    #[derive(Debug, Copy, Clone)]
    struct A {
        v: u32,
    };
    let mut a = GenIndex::<A>::new();

    let index1 = a.insert(A { v: 10 });
    assert_eq!(index1, NetIndex::new(0, 0));
    let index2 = a.insert(A { v: 20 });
    assert_eq!(index2, NetIndex::new(1, 0));

    let tv1 = a.get(&index1).unwrap().v;
    assert_eq!(tv1, 10);
    let tv2 = a.get(&index2).unwrap().v;
    assert_eq!(tv2, 20);
    let tv_none = a.get(&NetIndex::new(0, 1));
    assert_eq!(tv_none.is_none(), true);

    let a2 = a.remove(&index2);
    let tv_none = a.get(&index2);
    assert_eq!(tv_none.is_none(), true);
    assert_eq!(a2, true);

    let index3 = a.insert(A { v: 30 });
    assert_eq!(index3, NetIndex::new(1, 1));
}

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
fn test_bf() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();

    fn my_add(a: &Var, b: &Var) -> Var {
        a.add(b)
    }
    let c = my_add(&a, &b);
    a.set(Tensor::new());
    b.set(Tensor::new());
    m.forward();
    m.backward(&vec![Tensor::new()]);
}


#[test]
fn test_op_mse() {
    let mut m = Module::new();
    let a = m.var();
    let b = m.var();

    let c = MSELoss(&a, &b);
    a.set(Tensor::from_vec_f32(&vec![1., 2., 3., 4., 5., 6.,], &vec![3, 2]));
    b.set(Tensor::from_vec_f32(&vec![2., 3., 4., 5., 6., 7.,], &vec![3, 2]));
    m.forward();
    println!("test_op_mse, c: {}", c);
    
    assert_eq!(c.get() , Tensor::from_vec_f32(&vec![1., ], &vec![1]));
    println!("hehe");
}

#[test]
fn test_linear_regression() {

    println!("The beginning of  linear regression");

    fn func(input: &Tensor) -> Tensor {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]))
    }

    let N = 100;
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);
    let x = rng.normal(&vec![N, 2], 0., 2.);

    //println!("LR: {}, {:?}", x.numel(), x.size());
    // println!("LR x: {}", x);

    let y = func(&x);
    // println!("LR: {}", y);
    let op = Linear::new(Some(2), Some(1), true);

    let input = m.var();
    let output = input.to(&Op::new(Box::new(op)));
    let label = m.var();

    let loss = MSELoss(&output, &label);

    input.set(x);
    label.set(y);
    
    m.forward();
    println!("LR: {:?}", output.size());
    println!("LR: {:?}", loss.size());
    
    m.backward_scale(-1.);

    println!("End of linear regression");
}
