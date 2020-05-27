use tensorboard_rs::summary_writer::SummaryWriter;
use image::{open, };

pub fn main() {

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let stop_image = "./examples/stop.jpg";
    let img = open(stop_image).expect("");
    let img = img.into_rgb();
    let (width, height) = img.dimensions();

    
    writer.add_image(&"test_image".to_string(), &img.into_raw()[..], &vec![3, width as usize, height as usize][..], 12).expect("");
    writer.flush().expect("");
}
