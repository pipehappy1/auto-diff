use auto_diff_ann::minibatch::MiniBatch;
use auto_diff::Var;
use auto_diff_data_pipe::dataloader::{mnist::Mnist, DataSlice};
use std::path::Path;
use rand::prelude::*;
use ::rand::prelude::StdRng;
use std::fs;

extern crate openblas_src;

fn main() {

    let rng = StdRng::seed_from_u64(671);
    
    let mnist = Mnist::load(&Path::new("../auto-diff/examples/data/mnist"));
    let minibatch = MiniBatch::new(rng, 16);

    let file_name = "./saved_model/net_9900";
    let deserialized = fs::read(file_name).expect("unable to read file");
    let loss: Var = bincode::deserialize(&deserialized).unwrap();

    let (inputs, _outputs) = loss.get_io_var().expect("");
    let predict = loss.predict().expect("");

    let mut right = 0.;
    let mut total = 0.;
    for (data, label) in minibatch.iter_block(&mnist, &DataSlice::Test).expect("") {

	let input = data.reshape(&[minibatch.batch_size(), data.size()[1]*data.size()[2]]).unwrap();
	input.reset_net();

	inputs[0].set(&input);
	inputs[1].set(&label);
	loss.rerun().expect("");

	let predict_max = predict.clone().argmax(Some(&[1]), false).unwrap();
	right += f64::try_from(predict_max.eq_elem(&label).unwrap().sum(None, false).unwrap()).unwrap();
	total += input.size()[0] as f64;

	println!("acc: {}", right/total);
    }
}
