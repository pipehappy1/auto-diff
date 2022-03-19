use auto_diff::op::{Linear, OpCall};
use auto_diff::optim::{SGD};
use auto_diff_ann::minibatch::MiniBatch;
//use auto_diff::Var;
use auto_diff_ann::init::normal;
use auto_diff_data_pipe::dataloader::{mnist::Mnist, DataSlice};
use tensorboard_rs::summary_writer::SummaryWriter;
use std::path::Path;
use rand::prelude::*;
use ::rand::prelude::StdRng;
use auto_diff_data_pipe::dataloader::DataLoader;
use std::fs;

extern crate openblas_src;


fn main() {

    let mut rng = StdRng::seed_from_u64(671);

    let mnist = Mnist::load(&Path::new("../auto-diff/examples/data/mnist"));
    
    let train_size = mnist.get_size(Some(DataSlice::Train)).unwrap();
    let h = train_size[1];
    let w = train_size[2];

    // init
    let mut op1 = Linear::new(Some(h*w), Some(120), true);
    normal(op1.weight(), None, None, &mut rng).unwrap();
    normal(op1.bias(), None, None, &mut rng).unwrap();

    let mut op2 = Linear::new(Some(120), Some(84), true);
    normal(op2.weight(), None, None, &mut rng).unwrap();
    normal(op2.bias(), None, None, &mut rng).unwrap();

    let mut op3 = Linear::new(Some(84), Some(10), true);
    normal(op3.weight(), None, None, &mut rng).unwrap();
    normal(op3.bias(), None, None, &mut rng).unwrap();


    let mut minibatch = MiniBatch::new(rng, 16);
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    // get data
    let (input, label) = minibatch.next(&mnist, &DataSlice::Train).unwrap();
    let input = input.reshape(&[16, h*w]).unwrap();
    input.reset_net();

    // the network
    let output1 = op1.call(&[&input]).unwrap().pop().unwrap();
    let output2 = output1.relu().unwrap();
    let output3 = op2.call(&[&output2]).unwrap().pop().unwrap();
    let output4 = output3.relu().unwrap();
    let output = op3.call(&[&output4]).unwrap().pop().unwrap();

    // label the predict var.
    output.set_predict().unwrap();

    let loss = output.cross_entropy_loss(&label).unwrap();
    
    let lr = 0.001;
    let mut opt = SGD::new(lr);    
    
    for i in 0..100000 {
        let (input_next, label_next) = minibatch.next(&mnist, &DataSlice::Train).unwrap();
        let input_next = input_next.reshape(&[16, h*w]).unwrap();
        input_next.reset_net();

	// set data and label
        input.set(&input_next);
        label.set(&label_next);

        loss.rerun().unwrap();
        loss.bp().unwrap();
        loss.step(&mut opt).unwrap();
	
	if i % 1000 == 0 {
	    println!("i: {:?}, loss: {:?}", i, loss);
	    writer.add_scalar(&"mlp_mnist/train_loss".to_string(), f64::try_from(loss.clone()).unwrap() as f32, i);
	    
	    let encoded: Vec<u8> = bincode::serialize(&loss).unwrap();
	    fs::write(format!("saved_model/net_{}", i), encoded).expect("Unable to write file");
	}
    }
}
