use auto_diff::op::{Linear, OpCall};
use auto_diff::optim::{SGD};
use auto_diff_ann::minibatch::MiniBatch;
use auto_diff::Var;
use auto_diff_ann::init::normal;
use auto_diff_data_pipe::dataloader::{mnist::Mnist, DataSlice};
use tensorboard_rs::summary_writer::SummaryWriter;
use std::path::Path;
use ::rand::prelude::StdRng;

fn main() {
    



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

    let mnist = Mnist::load("../auto-diff/examples/data/mnist/train-images-idx3-ubyte" as Path);
    
    let train_size = mnist.get_size(Some(DataSlice::Train)).unwrap();
    let h = train_size[1];
    let w = train_size[2];

    let mut op1 = Linear::new(Some(h*w), Some(h*w*2), true);
    normal(op1.weight(), None, None, rng);
    normal(op1.bias(), None, None, rng);

    let mut op2 = Linear::new(Some(h*w*2), Some(10), true);
    normal(op2.weight(), None, None, rng);
    normal(op2.bias(), None, None, rng);

//    //println!("{}, {}", &train_data, &train_label);

    let mut minibatch = MiniBatch::new(rng, 16);

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
    let (input, label) = minibatch.next(&mnist, &DataSlice::Train).unwrap();        

    let output1 = op1.call(&[&input]).unwrap().pop().unwrap();
    let output2 = output1.sigmoid().unwrap();
    let output = op2.call(&[&output2]).unwrap().pop().unwrap();

    let loss = output.cross_entropy_loss(&label).unwrap();
    
    let lr = 0.1;
    let mut opt = SGD::new(lr);    

    
    for i in 0..900 {
        println!("index: {}", i);
        let (input_next, label_next) = minibatch.next(&train_data, &train_label).unwrap();        
        input.set(&input_next);
        label.set(&label_next);
        println!("load data done");

        //println!("dump net: {:?}", loss.dump_net().borrow());
        loss.rerun().unwrap();
        loss.bp().unwrap();
        loss.step(&mut opt).unwrap();

        println!("loss: {:?}", loss);
	writer.add_scalar(&"mlp_mnist/train_loss".to_string(), f64::try_from(loss.clone()).unwrap() as f32, i);


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
            println!("test accuracy: {:?}", tsum);

            //writer.add_scalar(&"run3/accuracy".to_string(), accuracy, i);
            //writer.flush();
        }
//        
//        //println!("{}, loss: {}", i, loss.get().get_scale_f32());
//        writer.add_scalar(&"run3/test_loss".to_string(), loss.get().get_scale_f32(), i);
//        writer.flush();
//
//        if i != 0 && i % 300 == 0 {
//            lr = lr / 3.;
//            opt = SGD::new(lr);
//        }
    }
}
