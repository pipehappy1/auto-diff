//from tensorboard.compat.proto.summary_pb2 import Summary
//from tensorboard.compat.proto import event_pb2

use tensorboard_rs::proto::summary::{Summary, Summary_Value};
use tensorboard_rs::proto::event::Event;
use protobuf::RepeatedField;
//use protobuf::text_format::print_to_string;
use protobuf::Message;


fn main() {
    
    let name = "run1";
    let scalar = 2.3;

    let mut value = Summary_Value::new();
    value.set_tag("run1".to_string());
    value.set_simple_value(2.4);

    let values = RepeatedField::from(vec![value]);
    let mut summary = Summary::new();
    summary.set_value(values);

    let mut evn = Event::new();
    evn.set_summary(summary);
    evn.set_step(12);
    //evn.set_wall_time();

    //let msg = print_to_string(&evn);
    //println!("{:?}", msg);
    let mut dump: Vec<u8> = Vec::new();
    println!("{:?}", evn);
    // the following is good
    evn.write_to_vec(&mut dump);
    //println!("{:x?}", dump);

    //header = struct.pack('<Q', len(data))
    //header_crc = struct.pack('<I', masked_crc32c(header))
    //footer_crc = struct.pack('<I', masked_crc32c(data))
    //self._writer.write(header + header_crc + data + footer_crc)
}

