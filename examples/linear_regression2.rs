use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op};
use auto_diff::var::{Module, mseloss};
use auto_diff::optim::{SGD, Optimizer};

fn main() {

    fn func(input: &Tensor) -> Tensor {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]))
    }

    let N = 10;
    let mut rng = RNG::new();
    rng.set_seed(123);
    let data = rng.normal(&vec![N, 2], 0., 2.);
    let label = func(&data);


    let mut m = Module::new();
    
    let op1 = m.linear(Some(2), Some(1), true);
    let block = m.func(
        move |x| {
            op1.call(x)
        }
    );
    
    let loss_func = m.mseloss();
    
    let mut opt = SGD::new(0.2);

    for i in 0..10 {
        println!("index: {:?}", i);
        
        let input = m.var_value(data.clone());
        
        let y = block.call(&[&input]);
        
        let loss = loss_func.call(&[&y, &m.var_value(label.clone())]);

        loss.backward(-1.);
        opt.step2(&block);

    }
    
    
    
    //
    //let y = func(&data);
    //// println!("LR: {}", y);
    //let op = Linear::new(Some(2), Some(1), true);
    //rng.normal_(op.weight(), 0., 1.);
    //rng.normal_(op.bias(), 0., 1.);
    //
    //// Good is good.
    ////let good_weight = Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]);
    ////let good_bias = Tensor::from_vec_f32(&vec![0.], &vec![1]);
    ////op.weight().swap(good_weight);
    ////op.bias().swap(good_bias);
    //
    //let linear = Op::new(Box::new(op));
    //
    //
    //let input = m.var();
    //let output = input.to(&linear);
    //let label = m.var();
    //
    //let loss = mseloss(&output, &label);
    //
    //input.set(x);
    //label.set(y);
    //
    //let mut opt = SGD::new(0.2);
    //
    //for i in 0..100 {
    //    m.forward();
    //    m.backward(-1.);
    //
    //    println!("{}", loss.get().get_scale_f32());
    //
    //    opt.step(&m);
    //
    //    let weights = linear.get_values();
    //    println!("{:?}, {:?}", weights[0], weights[1]);
    //}

}
