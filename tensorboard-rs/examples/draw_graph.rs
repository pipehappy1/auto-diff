use tensorboard_rs::summary_writer::SummaryWriter;
//use tensorboard_proto::event::{Event, TaggedRunMetadata};
//use tensorboard_proto::summary::{Summary};
//use tensorboard_proto::graph::{GraphDef, };
use tensorboard_proto::node_def::{NodeDef, };
//use tensorboard_proto::versions::{VersionDef, };
use tensorboard_proto::attr_value::{AttrValue, };
//use tensorboard_proto::tensor_shape::{TensorShapeProto, };
//use tensorboard_proto::step_stats::{RunMetadata, };
use protobuf::RepeatedField;
use std::collections::HashMap;

pub fn main() {
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let mut node1 = NodeDef::new();
    node1.set_name("node1".to_string());
    node1.set_op("op1".to_string());
    
    let inputs = RepeatedField::from(vec![]);
    node1.set_input(inputs);
    
    let mut attrs = HashMap::new();
    let mut v1 = AttrValue::new();
    v1.set_i(16);
    attrs.insert("attr1".to_string(), v1);
    node1.set_attr(attrs);

    writer.add_graph(&[node1]);
    
    writer.flush();
}
