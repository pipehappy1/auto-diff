use tensorboard_rs::summary_writer::SummaryWriter;
use std::collections::HashMap;

pub fn main() {
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let name = "run1";
    let mut scalar = 2.3;
    let mut step = 12;    
    for i in 0..2 {
        println!("{}", i);
        scalar += (i as f32)*0.1;
        step += i;

        writer.add_scalar(name, scalar, step).expect("");
    }
    writer.flush().expect("");

    for n_iter in 0..100 {
        let mut map = HashMap::new();
        map.insert("xsinx".to_string(), (n_iter as f32) * (n_iter as f32).sin());
        map.insert("xcosx".to_string(), (n_iter as f32) * (n_iter as f32).cos());
        map.insert("arctanx".to_string(), (n_iter as f32).atan());
        writer.add_scalars(&"data/scalar_group".to_string(), &map, n_iter);
    }
    writer.flush().expect("");
}
