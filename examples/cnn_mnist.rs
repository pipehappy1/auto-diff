use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op, Sigmoid, Conv2d};
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
    let train_data = train_img.reshape(&vec![n, h*w]);

    let test_size = test_img.size();
    let n = test_size[0];
    let h = test_size[1];
    let w = test_size[2];
    let test_data = test_img.reshape(&vec![n, h*w]);

    // build the model
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);

    // 28 - (3x3) - 28 - (3x3,2) - 14 - (3x3) - 14 - (3x3,2) - 7 - (3x3) - 7 - (3x3,2,nopadding) - 3 - (3x3,nopadding) - 1 - (1x1) - 1 - view
    
    let op1 = Conv2d::new(3, 8, (3,3), (1,1), (1,1), 0, true, PaddingMode::Zeros);
    rng.normal_(op1.get_values()[0], 0., 1.);
    rng.normal_(op1.get_values()[0], 0., 1.);

    let op1 = Conv2d::new(3, 8, (3,3), (1,1), (1,1), 0, true, PaddingMode::Zeros);
    rng.normal_(op1.get_values()[0], 0., 1.);
    rng.normal_(op1.get_values()[0], 0., 1.);
}
