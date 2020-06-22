use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op};
use auto_diff::var::{Module, mseloss};
use auto_diff::optim::{SGD, Optimizer};

fn main() {

    fn func(input: &Tensor) -> Tensor {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]))
    }

    let N = 10000;
    let mut rng = RNG::new();
    rng.set_seed(123);
    let data = rng.normal(&vec![N, 2], 0., 2.);
    let label = func(&data);


    let mut m = Module::new();
    
    let op1 = m.linear(Some(2), Some(1), true);
    let weights = op1.get_values().unwrap();
    rng.normal_(&weights[0], 0., 1.);
    rng.normal_(&weights[1], 0., 1.);
    op1.set_values(&weights);

    let op2 = op1.clone();
    let block = m.func(
        move |x| {
            op2.call(x)
        }
    );
    
    let loss_func = m.mseloss();
    
    let mut opt = SGD::new(0.2);

    for i in 0..10 {
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
