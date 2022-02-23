//! 1 hidden layer MLP example on Breast Cancer Wisconsin (Diagnostic) Data Set
//!
//! The dataset is from http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29


use auto_diff::var::Var;
use auto_diff::op::{Linear, OpCall};
use auto_diff::optim::{SGD};
use csv;
use std::collections::{BTreeSet};
use rand::prelude::*;

//use tensorboard_rs::summary_writer::SummaryWriter;

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

    let data = Var::empty(&vec![size, 31]);
    //println!("{:?} \n {}", data.size(), data);
    reader.seek(head).expect("");
    for (record, index) in reader.records().zip(0..size) {
        let line = record.expect("");
        let mut tmp = Vec::<f64>::with_capacity(31);
        
        ill = line[1].trim().parse::<String>().expect("");
        if ill == "M" {
            tmp.push(1.);
        } else {
            tmp.push(0.);
        }
        
        for i in 2..32 {
            let value = line[i].trim().parse::<f64>().expect("");
            //println!("{}", value);
            tmp.push(value);
        }
        //println!("{:?}", tmp);
        data.from_record_f64(index, &tmp);
    }

    
    //println!("{:?} \n {}", data.size(), data);
    let train_size = ((size as f32)*0.7) as usize;
    let test_size = size - train_size;
    //let splited_data = data.split(&vec![train_size, test_size], 0);
    let data_label_split = data.split(&vec![1, 30], 1).unwrap();
    let label = &data_label_split[0];
    let data = &data_label_split[1];
    let data = data.normalize_unit().unwrap();
    let label_split = label.split(&vec![train_size, test_size], 0).unwrap();
    let data_split = data.split(&vec![train_size, test_size], 0).unwrap();
    let train_data = &data_split[0];
    let train_label = &label_split[0];
    let test_data = &data_split[1];
    let test_label = &label_split[1];


    train_data.reset_net();
    train_label.reset_net();
    test_data.reset_net();
    test_label.reset_net();

    println!("{:?}", train_data.size());
    println!("{:?}", train_label.size());
    println!("{:?}", test_data.size());
    println!("{:?}", test_label.size());

    
    // build the model
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

    let mut opt = SGD::new(1.);

    for i in 0..500 {

        println!("i: {:?}", i);
        input.set(train_data);
        label.set(train_label);
        loss.rerun().unwrap();
        loss.bp().unwrap();
        loss.step(&mut opt).unwrap();

        input.set(test_data);
        label.set(test_label);
        loss.rerun().unwrap();
        println!("loss: {:?}", loss);

        //writer.add_scalar("run1/loss", loss.get().get_scale_f32(), i);
        //writer.add_scalar("run1/accuracy", 1.-tsum.get_scale_f32()/(test_size as f32), i);
        //writer.flush();

        let output1 = output.clone();
        let err = output1.sigmoid().unwrap().sub(&test_label.clone()).unwrap().abs().unwrap().sum(None, false).unwrap();
        println!("err: {:?}", err);
    }

    //println!("{:?}, {:?}", test_label, output.sigmoid());
}
