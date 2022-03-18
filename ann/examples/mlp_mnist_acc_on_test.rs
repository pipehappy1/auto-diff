use auto_diff::op::{Linear, OpCall};
use auto_diff::optim::{SGD};
use auto_diff_ann::minibatch::MiniBatch;
use auto_diff::Var;
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
    
    let mut minibatch = MiniBatch::new(rng, 16);

    let file_name = "./saved_model/net_9900";
    let deserialized = fs::read(file_name).expect("unable to read file");
    let deserialized: Var = bincode::deserialize(&deserialized).unwrap();

    //println!("net: {:?}", deserialized.dump_net().borrow());

    let (inputs, outputs) = deserialized.get_io_var().expect("");
    let predict = deserialized.predict().expect("");

    let mut right = 0.;
    let mut total = 0.;
    for (data, label) in minibatch.iter_block(&mnist, &DataSlice::Test).expect("") {
	//println!("{:?}, {:?}", data.size(), label.size());
	let input = data.reshape(&[minibatch.batch_size(), data.size()[1]*data.size()[2]]).unwrap();
	input.reset_net();
	label.reset_net();
	//println!("{:?}, {:?}", input.size(), label.size());
	inputs[0].set(&input);
	inputs[1].set(&label);
	deserialized.rerun();

	//println!("{:?}, {:?}, {:?}",outputs[0], predict, label);

	let predict_max = predict.clone().argmax(Some(&[1]), false).unwrap();
	//println!("{:?}, {:?}", predict_max, label);
	right += f64::try_from(predict_max.eq_elem(&label).unwrap().sum(None, false).unwrap()).unwrap();
	total += input.size()[0] as f64;

	println!("acc: {}", right/total);
	
	//let tsum = predict.clone().argmax(Some(&[1]), false).unwrap().eq_elem(&label).unwrap().mean(None, false);
        //    //let accuracy = tsum.get_scale_f32();
        //    //println!("{}, loss: {}, accuracy: {}", i, loss_value, accuracy);
        //    println!("test accuracy: {:?}", tsum);
    }
    println!("acc: {}", right/total);
}
