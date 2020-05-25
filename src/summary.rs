use crate::proto::summary::{Summary, Summary_Value};
use protobuf::RepeatedField;

pub fn scalar(name: &str, scalar_value: f32) -> Summary {

    let mut value = Summary_Value::new();
    value.set_tag(name.to_string());
    value.set_simple_value(scalar_value);

    let values = RepeatedField::from(vec![value]);
    let mut summary = Summary::new();
    summary.set_value(values);

    summary
}
