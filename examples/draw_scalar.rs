use tensorboard_rs::event_file_writer::EventFileWriter;
use tensorboard_rs::proto::summary::{Summary, Summary_Value};
use tensorboard_rs::proto::event::Event;
use protobuf::RepeatedField;

pub fn main() {
    let mut writer = EventFileWriter::new(&("./logdir".to_string()));

    let name = "run1";
    let mut scalar = 2.3;
    let mut step = 12;    
    for i in 0..2 {
        println!("{}", i);
        scalar += (i as f32)*0.1;
        step += i;

        let mut value = Summary_Value::new();
        value.set_tag(name.to_string());
        value.set_simple_value(scalar);
        
        let values = RepeatedField::from(vec![value]);
        let mut summary = Summary::new();
        summary.set_value(values);
        
        let mut evn = Event::new();
        evn.set_summary(summary);
        evn.set_step(step);
        
        writer.add_event(&evn).expect("");
    }
    writer.flush().expect("");
}
