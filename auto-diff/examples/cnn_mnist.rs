use tensor_rs::tensor::{PaddingMode};
use auto_diff::op::{Linear, OpCall, Conv2d};
use auto_diff::optim::{SGD, MiniBatch};
use auto_diff::Var;
use rand::prelude::*;
use ::rand::prelude::StdRng;


//use tensorboard_rs::summary_writer::SummaryWriter;

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
    let train_data = train_img.reshape(&vec![n, 1, h, w]).unwrap();

    let test_size = test_img.size();
    let n = test_size[0];
    let h = test_size[1];
    let w = test_size[2];
    let test_data = test_img.reshape(&vec![n, 1, h, w]).unwrap();

    train_data.reset_net();
    train_label.reset_net();
    test_data.reset_net();
    test_label.reset_net();

    let patch_size = 16;
    let class_size = 10;

    // build the model
//    let mut m = Module::new();
//    let mut rng = RNG::new();
//    rng.set_seed(123);
//
//    // 28 - (3x3) - 28 - (3x3,2) - 14 - (view) - 196 - (linear, 98.0) - 98 - (linear, 10) - 10
//    
//    let op1 = Conv2d::new(1, 32, (3,3), (1,1), (1,1), (1,1), true, PaddingMode::Zeros);
//    rng.normal_(op1.get_values()[0], 0., 1.);
//    rng.normal_(op1.get_values()[1], 0., 1.);
//    let conv1 = Op::new(Box::new(op1));
//
//    let op2 = Conv2d::new(32, 64, (3,3), (2,2), (1,1), (1,1), true, PaddingMode::Zeros);
//    rng.normal_(op2.get_values()[0], 0., 1.);
//    rng.normal_(op2.get_values()[1], 0., 1.);
//    let conv2 = Op::new(Box::new(op2));
//
//    let view = Op::new(Box::new(View::new(&[patch_size, 14*14*64])));
//
//    let op3 = Linear::new(Some(14*14*64), Some(14*14), true);
//    rng.normal_(op3.weight(), 0., 1.);
//    rng.normal_(op3.bias(), 0., 1.);
//    let linear3 = Op::new(Box::new(op3));
//
//    let op4 = Linear::new(Some(14*14), Some(10), true);
//    rng.normal_(op4.weight(), 0., 1.);
//    rng.normal_(op4.bias(), 0., 1.);
//    let linear4 = Op::new(Box::new(op4));
//
//    let mut acts = Vec::new();
//    for i in 0..3 {
//        let act1 = Op::new(Box::new(ReLU::new()));
//        acts.push(act1);
//    }
//
//    let input = m.var();
//    let output = input
//        .to(&conv1)
//        .to(&acts[0])
//        .to(&conv2)
//        .to(&acts[1])
//        .to(&view)
//        .to(&linear3)
//        .to(&acts[2])
//        .to(&linear4)
//        ;
//    let label = m.var();
//
//    let loss = crossentropyloss(&output, &label);
//
//    let rng = RNG::new();
//    let minibatch = MiniBatch::new(rng, patch_size);
//
//    let mut lr = 0.01;
//    let mut opt = SGD::new(lr);
//    
    //    let mut writer = SummaryWriter::new(&("./logdir".to_string()));


    let mut rng = StdRng::seed_from_u64(671);

    let mut op1 = Conv2d::new(1, 32, (3,3), (1,1), (1,1), (1,1), true, PaddingMode::Zeros);
    op1.set_weight(Var::normal(&mut rng, &op1.weight().size(), 0., 1.));
    op1.set_bias(Var::normal(&mut rng, &op1.bias().size(), 0., 1.));

    let mut op2 = Conv2d::new(32, 64, (3,3), (2,2), (1,1), (1,1), true, PaddingMode::Zeros);
    op2.set_weight(Var::normal(&mut rng, &op2.weight().size(), 0., 1.));
    op2.set_bias(Var::normal(&mut rng, &op2.bias().size(), 0., 1.));

    let mut op3 = Linear::new(Some(14*14*64), Some(14*14), true);
    op3.set_weight(Var::normal(&mut rng, &[14*14*64, 14*14], 0., 1.));
    op3.set_bias(Var::normal(&mut rng, &[14*14, ], 0., 1.));

    let mut op4 = Linear::new(Some(14*14), Some(10), true);
    op4.set_weight(Var::normal(&mut rng, &[14*14, 10], 0., 1.));
    op4.set_bias(Var::normal(&mut rng, &[10, ], 0., 1.));

//    //println!("{}, {}", &train_data, &train_label);
    let mut rng = StdRng::seed_from_u64(671);
    let mut minibatch = MiniBatch::new(rng, 16);

    //    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
    let (input, label) = minibatch.next(&train_data, &train_label).unwrap();       println!("here0");

    let output1 = op1.call(&[&input]).unwrap().pop().unwrap(); println!("here");
    let output1_1 = output1.relu().unwrap();  println!("here2");
    let output2 = op2.call(&[&output1_1]).unwrap().pop().unwrap();  println!("here3");
    let output2_1 = output2.relu().unwrap().view(&[patch_size, 14*14*64]).unwrap();  println!("her4");
    let output3 = op3.call(&[&output2_1]).unwrap().pop().unwrap();  println!("here5");
    let output3_1 = output3.relu().unwrap(); println!("her6");
    let output = op4.call(&[&output3_1]).unwrap().pop().unwrap();  println!("here7");

    let loss = output.cross_entropy_loss(&label).unwrap();  println!("here8");
    
    let mut lr = 0.1;
    let mut opt = SGD::new(lr);

    println!("{:?}", loss);
    
//    
//
    for i in 1..900 {
        println!("index: {}", i);

        //let (mdata, mlabel) = minibatch.next(&train_data, &train_label).unwrap();
        let (input_next, label_next) = minibatch.next(&train_data, &train_label).unwrap();        
        input.set(&input_next);
        label.set(&label_next);
        println!("load data done");

        loss.rerun().unwrap(); println!("rerun");
        loss.bp().unwrap();    println!("bp");
        loss.step(&mut opt).unwrap();  println!("step");
        
        if i % 10 == 0 {
        
            let (input_next, label_next) = minibatch.next(&test_data, &test_label).unwrap();        
            input.set(&input_next);
            label.set(&label_next);
            loss.rerun().unwrap();
        
            println!("test loss: {:?}", loss);
        
            //let loss_value = loss.get().get_scale_f32();
        
            let tsum = output.clone().argmax(Some(&[1]), false).unwrap().eq_elem(&test_label).unwrap().mean(None, false);
            //let accuracy = tsum.get_scale_f32();
            //println!("{}, loss: {}, accuracy: {}", i, loss_value, accuracy);
            println!("test error: {:?}", tsum);
        
            //writer.add_scalar(&"cnn/run1/accuracy".to_string(), accuracy, i);
            //writer.flush();
        }
        
        //println!("{}, loss: {}", i, loss.get().get_scale_f32());
        //writer.add_scalar(&"cnn/run1/test_loss".to_string(), loss.get().get_scale_f32(), i);
        //writer.flush();
        //
        //if i != 0 && i % 300 == 0 {
        //    lr = lr / 3.;
        //    opt = SGD::new(lr);
        //}
    }
}
