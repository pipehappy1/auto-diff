//! 1 hidden layer MLP example on Breast Cancer Wisconsin (Diagnostic) Data Set
//!
//! The dataset is from http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29


use tensor_rs::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op, Sigmoid};
//use auto_diff::var::{Module, bcewithlogitsloss};
//use auto_diff::optim::{SGD, Optimizer};
use csv;
use std::collections::{BTreeSet};

use tensorboard_rs::summary_writer::SummaryWriter;

fn main() {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("examples/data/wdbc.data")
        .expect("Cannot read wdbc.data");

    let mut id;
    let mut ill;
    let mut ids = BTreeSet::<usize>::new();
    let head = reader.position().clone();

    for record in reader.records() {
        let line = record.expect("");
        id = line[0].trim().parse::<usize>().expect("");
        //ill = line[1].trim().parse::<String>().expect("");
        //println!("{}, {}", id, ill);

        if !ids.contains(&id) {
            ids.insert(id);
        } else {
            println!("duplicate {}", id);
        }
    }
    let size = ids.len();
    println!("total size: {}", size);

    let data = Tensor::empty(&vec![size, 31]);
    //println!("{:?} \n {}", data.size(), data);
    reader.seek(head).expect("");
    for (record, index) in reader.records().zip(0..size) {
        let line = record.expect("");
        let mut tmp = Vec::<f32>::with_capacity(31);
        
        ill = line[1].trim().parse::<String>().expect("");
        if ill == "M" {
            tmp.push(1.);
        } else {
            tmp.push(0.);
        }
        
        for i in 2..32 {
            let value = line[i].trim().parse::<f32>().expect("");
            //println!("{}", value);
            tmp.push(value);
        }
        //println!("{:?}", tmp);
        data.from_record(index, &tmp).expect("");
    }

    
    //println!("{:?} \n {}", data.size(), data);
    let train_size = ((size as f32)*0.7) as usize;
    let test_size = size - train_size;
    //let splited_data = data.split(&vec![train_size, test_size], 0);
    let data_label_split = data.split(&vec![1, 30], 1);
    let label = &data_label_split[0];
    let data = &data_label_split[1];
    let data = data.normalize_unit();
    let label_split = label.split(&vec![train_size, test_size], 0);
    let data_split = data.split(&vec![train_size, test_size], 0);
    
    let train_data = &data_split[0];
    let train_label = &label_split[0];
    let test_data = &data_split[1];
    let test_label = &label_split[1];

    
    // build the model
//    let mut m = Module::new();
//    let mut rng = RNG::new();
//    rng.set_seed(123);
//
//    let op1 = m.linear(Some(30), Some(10), true);
//    let weights = op1.get_values().unwrap();
//    rng.normal_(&weights[0], 0., 1.);
//    rng.normal_(&weights[1], 0., 1.);
//    op1.set_values(&weights);
//
//    let op2 = m.linear(Some(10), Some(1), true);
//    let weights = op2.get_values().unwrap();
//    rng.normal_(&weights[0], 0., 1.);
//    rng.normal_(&weights[1], 0., 1.);
//    op2.set_values(&weights);
//
//
//    let act = m.sigmoid();
//
//    let linear1 = op1.clone();
//    let linear2 = op2.clone();
//    let block = m.func(
//        move |x| {
//            let x1 = act.call(&[&linear1.call(x)]);
//            linear2.call(&[&x1])
//        }
//    );
//
//    let loss = m.bce_with_logits_loss();
//    
//    let mut opt = SGD::new(0.2);
//
//    let mut writer = SummaryWriter::new(&("./logdir".to_string()));
//
//    for i in 0..500 {
//        let input = m.var_value(train_data.clone());
//        
//        let y = block.call(&[&input]);
//        
//        let loss = loss.call(&[&y, &m.var_value(train_label.clone())]);
//        println!("index: {}, loss: {}", i, loss.get().get_scale_f32());
//        
//        loss.backward(-1.);
//        opt.step2(&block);
//
//        let test_input = m.var_value(test_data.clone());
//        let y = block.call(&[&test_input]);
//        let tsum = y.get().sigmoid().sub(&test_label).sum(None, false);
//        writer.add_scalar("run1/loss", loss.get().get_scale_f32(), i);
//        writer.add_scalar("run1/accuracy", 1.-tsum.get_scale_f32()/(test_size as f32), i);
//        writer.flush();
//    }
}
