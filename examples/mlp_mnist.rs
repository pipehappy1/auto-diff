use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op};
use auto_diff::var::{Module, bcewithlogitsloss};
use auto_diff::optim::{SGD, Optimizer};
use csv;
use std::collections::{BTreeMap, BTreeSet};

use tensorboard_rs::summary_writer::SummaryWriter;

mod mnist;
use mnist::{load_images, load_labels};

fn main() {
    
    let train_img = load_images("examples/data/mnist/train-images-idx3-ubyte");
    let test_img = load_images("examples/data/mnist/t10k-images-idx3-ubyte");
    let train_label = load_images("examples/data/mnist/train-labels-idx1-ubyte");
    let test_label = load_images("examples/data/mnist/t10k-labels-idx3-ubyte");

    let train_size = train_img.size();
    let n = train_size[0];
    let h = train_size[1];
    let w = train_size[2];
    let train_img = train_img.reshape(&vec![n, h*w]);

    let test_size = test_img.size();
    let n = train_size[0];
    let h = train_size[1];
    let w = train_size[2];
    let test_img = test_img.reshape(&vec![n, h*w]);


    // build the model
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);
    
    let op1 = Linear::new(Some(h*w), Some(h*w*2), true);
    rng.normal_(op1.weight(), 0., 1.);
    rng.normal_(op1.bias(), 0., 1.);
    
    let linear1 = Op::new(Box::new(op1));
    
    let op2 = Linear::new(Some(h*w*2), Some(10), true);
    rng.normal_(op2.weight(), 0., 1.);
    rng.normal_(op2.bias(), 0., 1.);
    
    let linear2 = Op::new(Box::new(op2));
    
    let activator = Op::new(Box::new(Sigmoid::new()));
    
    let input = m.var();
    let output = input
        .to(&linear1)
        .to(&activator)
        .to(&linear2);
    let label = m.var();
    
    let loss = bcewithlogitsloss(&output, &label);
    
    //println!("{}, {}", &train_data, &train_label);
    
    
    let mut opt = SGD::new(0.2);
    
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
    
    
    //for i in 0..500 {
    //    m.forward();
    //    m.backward(-1.);
    //
    //    opt.step(&m);
    //
    //    let predict = Tensor::empty(&test_label.size());
    //    linear.apply(&vec![test_data], &vec![&predict]);
    //    let tsum = predict.sigmoid().sub(&test_label).sum();
    //    println!("{}, loss: {}, accuracy: {}", i, loss.get().get_scale_f32(), 1.-tsum.get_scale_f32()/(test_size as f32));
    //
    //}
}
