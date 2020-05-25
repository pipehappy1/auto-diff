use std::path::{PathBuf, Path};
use std::time::SystemTime;
use crate::event_file_writer::EventFileWriter;
use crate::proto::event::Event;
use crate::proto::summary::{Summary};
use crate::summary::scalar;


pub struct FileWriter {
    writer: EventFileWriter,
}
impl FileWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> FileWriter {
        FileWriter {
            writer: EventFileWriter::new(logdir),
        }
    }
    pub fn get_logdir(&self) -> PathBuf {
        self.writer.get_logdir()
    }
    pub fn add_event(&mut self, event: &Event, step: usize) -> std::io::Result<()> {
        let mut event = event.clone();
        
        let mut time_full = 0.0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time_full = n.as_secs_f64();
        }
        event.set_wall_time(time_full);
        
        event.set_step(step as i64);
        
        self.writer.add_event(&event)
    }
    pub fn add_summary(&mut self, summary: Summary, step: usize) -> std::io::Result<()> {
        let mut evn = Event::new();
        evn.set_summary(summary);
        self.add_event(&evn, step)
    }
    pub fn add_graph(&mut self) {
        unimplemented!();
    }
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

pub struct SummaryWriter {
    writer: FileWriter,
}
impl SummaryWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> SummaryWriter {
        SummaryWriter {
            writer: FileWriter::new(logdir),
        }
    }
    pub fn add_hparams(&mut self) {unimplemented!();}
    pub fn add_scalar(&mut self, tag: &str, scalar_value: f32, step: usize) -> std::io::Result<()> {
        self.writer.add_summary(scalar(tag, scalar_value), step)
    }
    pub fn add_scalars(&mut self) {unimplemented!();}

    pub fn export_scalars_to_json(&self) {unimplemented!();}
    pub fn add_histogram(&mut self) {unimplemented!();}
    pub fn add_histogram_raw(&mut self) {unimplemented!();}
    pub fn add_image(&mut self) {unimplemented!();}
    pub fn add_images(&mut self) {unimplemented!();}
    pub fn add_image_with_boxes(&mut self) {unimplemented!();}
    pub fn add_figure(&mut self) {unimplemented!();}
    pub fn add_video(&mut self) {unimplemented!();}
    pub fn add_audio(&mut self) {unimplemented!();}
    pub fn add_text(&mut self) {unimplemented!();}
    pub fn add_onnx_graph(&mut self) {unimplemented!();}
    pub fn add_openvino_graph(&mut self) {unimplemented!();}
    pub fn add_graph(&mut self) {unimplemented!();}
    pub fn add_embedding(&mut self) {unimplemented!();}
    pub fn add_pr_curve(&mut self) {unimplemented!();}
    pub fn add_pr_curve_raw(&mut self) {unimplemented!();}
    pub fn add_custom_scalars_multilinechart(&mut self) {unimplemented!();}
    pub fn add_custom_scalars_marginchart(&mut self) {unimplemented!();}
    pub fn add_custom_scalars(&mut self) {unimplemented!();}
    pub fn add_mesh(&mut self) {unimplemented!();}

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
