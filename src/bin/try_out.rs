//from tensorboard.compat.proto.summary_pb2 import Summary
//from tensorboard.compat.proto import event_pb2

use tensorboard_rs::proto::summary::{Summary, Summary_Value};
use tensorboard_rs::proto::event::Event;
use tensorboard_rs::masked_crc32c::masked_crc32c;
use protobuf::RepeatedField;
//use protobuf::text_format::print_to_string;
use protobuf::Message;

use std::fs::File;
use std::io::Write;


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
    evn.write_to_vec(&mut dump).expect("");
    println!("{:x?}", dump);

    println!("{}", masked_crc32c(&dump));
    
    //header = struct.pack('<Q', len(data))
    //header_crc = struct.pack('<I', masked_crc32c(header))
    //footer_crc = struct.pack('<I', masked_crc32c(data))
    //self._writer.write(header + header_crc + data + footer_crc)

    let header = dump.len() as u64;
    let header_crc = (masked_crc32c(&(header.to_le_bytes())) as u32).to_le_bytes();
    let footer_crc = (masked_crc32c(&dump) as u32).to_le_bytes();
    let header = header.to_le_bytes();

    let mut file = File::create("test.log").expect("");
    file.write_all(&header).expect("");
    file.write_all(&header_crc).expect("");
    file.write_all(&dump).expect("");
    file.write_all(&footer_crc).expect("");
    
    
}

