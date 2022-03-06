use tensorboard_rs::summary_writer::SummaryWriter;
use image::{open, };

pub fn main() {

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let min = 1.001;
    let max = 29.001;
    let num = 435.;
    let sum = 8555.435;
    let sum_squares = 189242.110435;
    let bucket_limits = [3.8009999999999997, 6.600999999999999, 9.400999999999998, 12.200999999999999, 15.001, 17.801, 20.601, 23.401, 26.201, 29.001];
    let bucket_counts = [ 6., 15., 24., 33., 27., 48., 57., 66., 75., 84.];
    
    writer.add_histogram_raw("run1/histo1",
                             min, max,
                             num,
                             sum, sum_squares,
                             &bucket_limits, &bucket_counts,
                             1
    );

    writer.add_histogram_raw("run1/histo1",
                             min, max,
                             num,
                             sum, sum_squares,
                             &bucket_limits, &bucket_counts,
                             2
    );

    writer.add_histogram_raw("run1/histo1",
                             min, max,
                             num,
                             sum, sum_squares,
                             &bucket_limits, &bucket_counts,
                             3
    );
    writer.flush();
}
