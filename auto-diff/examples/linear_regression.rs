use rand::prelude::*;
use auto_diff::var::Var;
use auto_diff::optim::{SGD};
use auto_diff::op::Linear;
use auto_diff::op::OpCall;
use auto_diff::err::AutoDiffError;

fn main() {

    fn func(input: &Var) -> Result<Var, AutoDiffError> {
        let input = input.clone();
        input.set_grad(false);
        let result = input.matmul(&Var::new(&vec![2., 3.], &vec![2, 1]))?.add(&Var::new(&vec![1.], &vec![1]))?;
        result.set_grad(true);
        Ok(result)
    }

    let N = 15;
    let mut rng = StdRng::seed_from_u64(671);
    let mut data = Var::normal(&mut rng, &vec![N, 2], 0., 2.);
    let label = func(&data).unwrap();
    
    let mut op1 = Linear::new(Some(2), Some(1), true);
    op1.set_weight(Var::normal(&mut rng, &[2, 1], 0., 2.));
    op1.set_bias(Var::normal(&mut rng, &[1, ], 0., 2.));

    let output = op1.call(&[&data]).unwrap().pop().unwrap();

    let loss = output.mse_loss(&label).unwrap();
    
    let mut opt = SGD::new(3.);

    for i in 0..200 {
        
        println!("index: {}, loss: {:?}", i, loss);
        loss.rerun().unwrap();
        loss.bp().unwrap();
        loss.step(&mut opt).unwrap();
    }

    let weight = op1.weight();
    let bias = op1.bias();
    println!("{:?}, {:?}", weight, bias);
}
