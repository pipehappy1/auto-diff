use tensorboard_rs::summary_writer::SummaryWriter;

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
}
