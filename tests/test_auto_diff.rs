use auto_diff::graph::*;
use auto_diff::tensor::*;

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
    assert_eq!(a2.is_some(), true);

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
    println!("{}", c);
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
