//! Logistic regression example on Breast Cancer Wisconsin (Diagnostic) Data Set
//!
//! The dataset is from http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29


use auto_diff::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::op::{Linear, Op};
use auto_diff::var::{Module, bcewithlogitsloss};
use auto_diff::optim::{SGD, Optimizer};
use csv;
use std::collections::{BTreeMap, BTreeSet};

fn main() {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("examples/data/wdbc.data")
        .expect("Cannot read wdbc.data");

    let mut size = 0;
    let mut id;
    let mut ill;
    let mut ids = BTreeSet::<usize>::new();
    let head = reader.position().clone();

    for record in reader.records() {
        let line = record.expect("");
        id = line[0].trim().parse::<usize>().expect("");
        ill = line[1].trim().parse::<String>().expect("");
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
    let label = data_label_split[0].clone();
    let data = data_label_split[1].clone();
    let data = data.normalize_unit();
    let label_split = label.split(&vec![train_size, test_size], 0);
    let data_split = data.split(&vec![train_size, test_size], 0);
    let train_data = data_split[0].clone();
    let train_label = label_split[0].clone();
    let test_data = data_split[1].clone();
    let test_label = label_split[1].clone();
    
    

    // build the model
    let mut m = Module::new();
    let mut rng = RNG::new();
    rng.set_seed(123);

    let op = Linear::new(Some(30), Some(1), true);
    rng.normal_(op.weight(), 0., 1.);
    rng.normal_(op.bias(), 0., 1.);

    let linear = Op::new(Box::new(op));

    let input = m.var();
    let output = input.to(&linear);
    let label = m.var();

    let loss = bcewithlogitsloss(&output, &label);
    
    //println!("{}, {}", &train_data, &train_label);
    input.set(train_data.clone());
    label.set(train_label.clone());

    let mut opt = SGD::new(0.1);


    for i in 0..5000 {
        m.forward();
        m.backward(-1.);

        opt.step(&m);

        let predict = Tensor::empty(&test_label.size());
        linear.apply(&vec![&test_data], &vec![&predict]);
        let tsum = predict.sigmoid().sub(&test_label).sum();
        println!("loss: {}, accuracy: {}", loss.get().get_scale_f32(), tsum.get_scale_f32()/(test_size as f32));

    }
}
