use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op, Sigmoid, Conv2d, PaddingMode, OpTrait, ReLU, View};
use auto_diff::var::{Module, crossentropyloss};
use auto_diff::optim::{SGD, Optimizer, MiniBatch};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use tensorboard_rs::summary_writer::SummaryWriter;

mod mnist;
use mnist::{load_images, load_labels};

fn main() {
    let train_img = load_images("examples/data/mnist/train-images-idx3-ubyte");
    let test_img = load_images("examples/data/mnist/t10k-images-idx3-ubyte");
    let train_label = load_labels("examples/data/mnist/train-labels-idx1-ubyte");
    let test_label = load_labels("examples/data/mnist/t10k-labels-idx1-ubyte");

    let train_size = train_img.size();
    let n = train_size[0];
    let h = train_size[1];
    let w = train_size[2];
    let train_data = train_img.reshape(&vec![n, 1, h, w]);

    let test_size = test_img.size();
    let n = test_size[0];
    let h = test_size[1];
    let w = test_size[2];
    let test_data = test_img.reshape(&vec![n, 1, h, w]);

    let patch_size = 16;
    let class_size = 10;

    // build the model
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);

    // 28 - (3x3) - 28 - (3x3,2) - 14 - (3x3) - 14 - (3x3,2) - 7 - (3x3) - 7 - (3x3,2,nopadding) - 3 - (3x3,nopadding) - 1 - (1x1) - 1 - view
    
    let op1 = Conv2d::new(1, 16, (3,3), (1,1), (1,1), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op1.get_values()[0], 0., 1.);
    rng.normal_(op1.get_values()[1], 0., 1.);
    let conv1 = Op::new(Box::new(op1));

    let op2 = Conv2d::new(16, 32, (3,3), (2,2), (1,1), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op2.get_values()[0], 0., 1.);
    rng.normal_(op2.get_values()[1], 0., 1.);
    let conv2 = Op::new(Box::new(op2));

    let op3 = Conv2d::new(32,32, (3,3), (1,1), (1,1), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op3.get_values()[0], 0., 1.);
    rng.normal_(op3.get_values()[1], 0., 1.);
    let conv3 = Op::new(Box::new(op3));

    let op4 = Conv2d::new(32, 64, (3,3), (2,2), (1,1), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op4.get_values()[0], 0., 1.);
    rng.normal_(op4.get_values()[1], 0., 1.);
    let conv4 = Op::new(Box::new(op4));

    let op5 = Conv2d::new(64, 64, (3,3), (1,1), (1,1), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op5.get_values()[0], 0., 1.);
    rng.normal_(op5.get_values()[1], 0., 1.);
    let conv5 = Op::new(Box::new(op5));

    let op6 = Conv2d::new(64, 128, (3,3), (2,2), (0,0), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op6.get_values()[0], 0., 1.);
    rng.normal_(op6.get_values()[1], 0., 1.);
    let conv6 = Op::new(Box::new(op6));

    let op7 = Conv2d::new(128, 128, (3,3), (1,1), (0,0), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op7.get_values()[0], 0., 1.);
    rng.normal_(op7.get_values()[1], 0., 1.);
    let conv7 = Op::new(Box::new(op7));

    let op8 = Conv2d::new(128, 10, (1,1), (1,1), (0,0), (1,1), true, PaddingMode::Zeros);
    rng.normal_(op8.get_values()[0], 0., 1.);
    rng.normal_(op8.get_values()[1], 0., 1.);
    let conv8 = Op::new(Box::new(op8));

    let mut acts = Vec::new();
    for i in 0..8 {
        let act1 = Op::new(Box::new(ReLU::new()));
        acts.push(act1);
    }

    let view = Op::new(Box::new(View::new(&[patch_size, class_size])));

    let input = m.var();
    let output = input
        .to(&conv1)
        .to(&acts[0])
        .to(&conv2)
        .to(&acts[1])
        .to(&conv3)
        .to(&acts[2])
        .to(&conv4)
        .to(&acts[3])
        .to(&conv5)
        .to(&acts[4])
        .to(&conv6)
        .to(&acts[5])
        .to(&conv7)
        .to(&acts[6])
        .to(&conv8)
        .to(&acts[7])
        .to(&view)
        ;
    let label = m.var();

    let loss = crossentropyloss(&output, &label);

    let rng = RNG::new();
    let minibatch = MiniBatch::new(rng, patch_size);

    let mut lr = 0.1;
    let mut opt = SGD::new(lr);
    
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
    

    for i in 0..900 {
        println!("index: {}", i);
        let (mdata, mlabel) = minibatch.next(&train_data, &train_label);
        input.set(mdata);
        label.set(mlabel);
        println!("load data done");
        m.forward();
        println!("forward done");
        m.backward(-1.);
        println!("backward done");
        opt.step(&m);
        println!("update done");


        //if i % 10 == 0 {
        //    input.set(test_data.clone());
        //    label.set(test_label.clone());
        //    m.forward();
        //
        //    let loss_value = loss.get().get_scale_f32();
        //
        //    let tsum = output.get().argmax(Some(&[1]), false).eq_t(&test_label).mean(None, false);
        //    let accuracy = tsum.get_scale_f32();
        //    println!("{}, loss: {}, accuracy: {}", i, loss_value, accuracy);
        //
        //    writer.add_scalar(&"cnn/run1/accuracy".to_string(), accuracy, i);
        //    writer.flush();
        //}
        
        println!("{}, loss: {}", i, loss.get().get_scale_f32());
        writer.add_scalar(&"cnn/run1/test_loss".to_string(), loss.get().get_scale_f32(), i);
        writer.flush();

        if i != 0 && i % 300 == 0 {
            lr = lr / 3.;
            opt = SGD::new(lr);
        }
    }
}
