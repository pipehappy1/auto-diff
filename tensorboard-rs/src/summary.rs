use crate::proto::summary::{Summary, Summary_Value, Summary_Image, SummaryMetadata, SummaryMetadata_PluginData, HistogramProto};
use crate::proto::layout::{Layout, Category};
use protobuf::RepeatedField;

use image::{RgbImage, DynamicImage, ImageOutputFormat};
use std::io::Write;

pub fn scalar(name: &str, scalar_value: f32) -> Summary {

    let mut value = Summary_Value::new();
    value.set_tag(name.to_string());
    value.set_simple_value(scalar_value);

    let values = RepeatedField::from(vec![value]);
    let mut summary = Summary::new();
    summary.set_value(values);

    summary
}

pub fn histogram_raw(name: &str,
                     min: f64, max: f64,
                     num: f64,
                     sum: f64, sum_squares: f64,
                     bucket_limits: &[f64],
                     bucket_counts: &[f64],
) -> Summary {
    let mut hist = HistogramProto::new();
    hist.set_min(min);
    hist.set_max(max);
    hist.set_num(num);
    hist.set_sum(sum);
    hist.set_sum_squares(sum_squares);
    hist.set_bucket_limit(bucket_limits.to_vec());
    hist.set_bucket(bucket_counts.to_vec());
    
    let mut value = Summary_Value::new();
    value.set_tag(name.to_string());
    value.set_histo(hist);

    let values = RepeatedField::from(vec![value]);
    let mut summary = Summary::new();
    summary.set_value(values);

    summary
}

/// dim is in CHW
pub fn image(tag: &str, data: &[u8], dim: &[usize]) -> Summary {
    if dim.len() != 3 {
        panic!("format:CHW");
    }
    if dim[0] != 3 {
        panic!("needs rgb");
    }
    if data.len() != dim[0]*dim[1]*dim[2] {
        panic!("length of data should matches with dim.");
    }
    
    let mut img = RgbImage::new(dim[1] as u32, dim[2] as u32);
    img.clone_from_slice(data);
    let dimg = DynamicImage::ImageRgb8(img);
    let mut output_buf = Vec::<u8>::new();
    dimg.write_to(&mut output_buf, ImageOutputFormat::Png).expect("");

    let mut output_image = Summary_Image::new();
    output_image.set_height(dim[1] as i32);
    output_image.set_width(dim[2] as i32);
    output_image.set_colorspace(3);
    output_image.set_encoded_image_string(output_buf);
    let mut value = Summary_Value::new();
    value.set_tag(tag.to_string());
    value.set_image(output_image);
    let values = RepeatedField::from(vec![value]);
    let mut summary = Summary::new();
    summary.set_value(values);

    summary
}

pub fn custom_scalars(layout: f32) {
    let mut layout = Layout::new();
    let value = Category::new();
    let values = RepeatedField::from(vec![value]);
    layout.set_category(values);

    let mut plugin_data = SummaryMetadata_PluginData::new();
    plugin_data.set_plugin_name("custom_scalars".to_string());
    let mut smd = SummaryMetadata::new();
    smd.set_plugin_data(plugin_data);

    
}
