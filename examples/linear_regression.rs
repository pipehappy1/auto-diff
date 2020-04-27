use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op};
use auto_diff::var::{Module, mseloss};

fn main() {

    fn func(input: &Tensor) -> Tensor {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]))
    }

    let N = 10;
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);
    let x = rng.normal(&vec![N, 2], 0., 2.);

    //println!("LR: {}, {:?}", x.numel(), x.size());
    // println!("LR x: {}", x);

    let y = func(&x);
    // println!("LR: {}", y);
    let op = Linear::new(Some(2), Some(1), true);
    rng.normal_(op.weight(), 0., 1.);
    rng.normal_(op.bias(), 0., 1.);

    let input = m.var();
    let output = input.to(&Op::new(Box::new(op)));
    let label = m.var();

    let loss = mseloss(&output, &label);

    input.set(x);
    label.set(y);
    
    m.forward();
    println!("LR: {}", output);
    println!("LR: {}", loss);
    
    m.backward(-1.);

}
