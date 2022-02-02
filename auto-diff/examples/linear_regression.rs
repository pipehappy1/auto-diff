use tensor_rs::tensor::Tensor;
use rand::prelude::*;
use auto_diff::var::Var;
use auto_diff::optim::{SGD, Optimizer};
use auto_diff::op::Linear;

fn main() {

    fn func(input: &Var) -> Var {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1])).add(&Tensor::from_vec_f32(&vec![1.], &vec![1]));
        
    }

    let N = 100;
    let mut rng = StdRng::seed_from_u64(671);
    let data = Var::normal(&mut rng, &vec![N, 2], 0., 2.);
    let label = func(&data.clone().set_grad(false));

    
    let op1 = Linear::new(Some(2), Some(1), true);
    op1.set_weight(Var::normal(&mut rng, &[2], 0., 2.));
    op1.set_bias(Var::normal(&mut rng, &[1], 0., 2.));


    
    
    let mut opt = SGD::new(3.);

    for i in 0..200 {
        let input = m.var_value(data.clone());
        
        let y = block.call(&[&input]);
        
        let loss = loss_func.call(&[&y, &m.var_value(label.clone())]);
        println!("index: {}, loss: {}", i, loss.get().get_scale_f32());
        
        loss.backward(-1.);
        opt.step2(&block);

    }

    let weights = op1.get_values().expect("");
    println!("{:?}, {:?}", weights[0], weights[1]);
}
