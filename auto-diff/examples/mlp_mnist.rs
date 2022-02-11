use auto_diff::op::{Linear, OpCall};
use auto_diff::optim::{SGD, MiniBatch};
use auto_diff::Var;
use rand::prelude::*;


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
    let train_data = train_img.reshape(&vec![n, h*w]);

    let test_size = test_img.size();
    let n = test_size[0];
    let h = test_size[1];
    let w = test_size[2];
    let test_data = test_img.reshape(&vec![n, h*w]);

    


    // build the model
//    let mut m = Module::new();
//    let mut rng = RNG::new();
//    rng.set_seed(123);
//    
//    let op1 = Linear::new(Some(h*w), Some(h*w*2), true);
//    rng.normal_(op1.weight(), 0., 1.);
//    rng.normal_(op1.bias(), 0., 1.);
//    
//    let linear1 = Op::new(Box::new(op1));
//    
//    let op2 = Linear::new(Some(h*w*2), Some(10), true);
//    rng.normal_(op2.weight(), 0., 1.);
//    rng.normal_(op2.bias(), 0., 1.);
//    
//    let linear2 = Op::new(Box::new(op2));
//    
//    let activator = Op::new(Box::new(Sigmoid::new()));
//    
//    let input = m.var();
//    let output = input
//        .to(&linear1)
//        .to(&activator)
//        .to(&linear2);
//    let label = m.var();
//    
    //    let loss = crossentropyloss(&output, &label);

    let mut rng = StdRng::seed_from_u64(671);

    let mut op1 = Linear::new(Some(30), Some(10), true);
    op1.set_weight(Var::normal(&mut rng, &[30, 10], 0., 1.));
    op1.set_bias(Var::normal(&mut rng, &[10, ], 0., 1.));

    let mut op2 = Linear::new(Some(10), Some(1), true);
    op2.set_weight(Var::normal(&mut rng, &[10, 1], 0., 1.));
    op2.set_bias(Var::normal(&mut rng, &[1, ], 0., 1.));

    //    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
    let input = train_data.clone();
    let label = train_label.clone();

    let output1 = op1.call(&[&input]).unwrap().pop().unwrap();
    let output2 = output1.sigmoid().unwrap();
    let output = op2.call(&[&output2]).unwrap().pop().unwrap();

    let loss = output.bce_with_logits_loss(&label).unwrap();
    
    
//    
//    //println!("{}, {}", &train_data, &train_label);
//    let rng = RNG::new();
//    let minibatch = MiniBatch::new(rng, 16);
//
    let mut lr = 0.2;
    let mut opt = SGD::new(lr);
//    
//    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
//    
//    
//    for i in 0..900 {
//        println!("index: {}", i);
//        let (mdata, mlabel) = minibatch.next(&train_data, &train_label);
//        input.set(mdata);
//        label.set(mlabel);
//        println!("load data done");
//        m.forward();
//        println!("forward done");
//        m.backward(-1.);
//        println!("backward done");
//        opt.step(&m);
//        println!("update done");
//
//
//        if i % 10 == 0 {
//            input.set(test_data.clone());
//            label.set(test_label.clone());
//            m.forward();
//
//            let loss_value = loss.get().get_scale_f32();
//        
//            let tsum = output.get().argmax(Some(&[1]), false).eq_t(&test_label).mean(None, false);
//            let accuracy = tsum.get_scale_f32();
//            println!("{}, loss: {}, accuracy: {}", i, loss_value, accuracy);
//
//            writer.add_scalar(&"run3/accuracy".to_string(), accuracy, i);
//            writer.flush();
//        }
//        
//        //println!("{}, loss: {}", i, loss.get().get_scale_f32());
//        writer.add_scalar(&"run3/test_loss".to_string(), loss.get().get_scale_f32(), i);
//        writer.flush();
//
//        if i != 0 && i % 300 == 0 {
//            lr = lr / 3.;
//            opt = SGD::new(lr);
//        }
//    }
}
