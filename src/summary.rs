use crate::proto::summary::{Summary, Summary_Value, SummaryMetadata, SummaryMetadata_PluginData};
use crate::proto::layout::{Layout, Category};
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

pub fn custom_scalars(layout: f32) {
    let mut layout = Layout::new();
    let mut value = Category::new();
    let values = RepeatedField::from(vec![value]);
    layout.set_category(values);

    let mut plugin_data = SummaryMetadata_PluginData::new();
    plugin_data.set_plugin_name("custom_scalars".to_string());
    let mut smd = SummaryMetadata::new();
    smd.set_plugin_data(plugin_data);

    
}
