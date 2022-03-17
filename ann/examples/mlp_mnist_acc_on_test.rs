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

    let file_name = "./net_495";
    let deserialized = fs::read(file_name).expect("unable to read file");
    let deserialized: Var = bincode::deserialize(&deserialized).unwrap();

    let (inputs, outputs) = deserialized.get_io_var().expect("");

    for (data, label) in minibatch.iter_block(&mnist, &DataSlice::Test).expect("") {
	println!("{:?}, {:?}", data.size(), label.size());
	let input = data.reshape(&[minibatch.batch_size(), data.size()[1]*data.size()[2]]).unwrap();
	input.reset_net();
	label.reset_net();
	println!("{:?}, {:?}", input.size(), label.size());
	inputs[0].set(&input);
	inputs[1].set(&label);
	deserialized.rerun();

	println!("{:?}",outputs[0]);
    }

}
