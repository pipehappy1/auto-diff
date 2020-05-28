use std::path::{PathBuf, Path};
use std::time::SystemTime;
use std::collections::HashMap;
use crate::event_file_writer::EventFileWriter;
use crate::proto::event::Event;
use crate::proto::summary::{Summary};
use crate::summary::{scalar, image};


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
    pub fn add_event(&mut self, event: &Event, step: usize) {
        let mut event = event.clone();
        
        let mut time_full = 0.0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time_full = n.as_secs_f64();
        }
        event.set_wall_time(time_full);
        
        event.set_step(step as i64);
        
        self.writer.add_event(&event)
    }
    pub fn add_summary(&mut self, summary: Summary, step: usize) {
        let mut evn = Event::new();
        evn.set_summary(summary);
        self.add_event(&evn, step)
    }
    pub fn add_graph(&mut self) {
        unimplemented!();
    }
    pub fn flush(&mut self) {
        self.writer.flush()
    }
}

pub struct SummaryWriter {
    writer: FileWriter,
    all_writers: HashMap<PathBuf, FileWriter>,
}
impl SummaryWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> SummaryWriter {
        SummaryWriter {
            writer: FileWriter::new(logdir),
            all_writers: HashMap::new(),
        }
    }
    pub fn add_hparams(&mut self) {unimplemented!();}
    pub fn add_scalar(&mut self, tag: &str, scalar_value: f32, step: usize) {
        self.writer.add_summary(scalar(tag, scalar_value), step);
    }
    pub fn add_scalars(&mut self, main_tag: &str, tag_scalar: &HashMap<String, f32>, step: usize) {
        let base_logdir = self.writer.get_logdir();
        for (tag, scalar_value) in tag_scalar.iter() {
            let fw_tag = base_logdir.join(main_tag).join(tag);
            if ! self.all_writers.contains_key(&fw_tag) {
                let new_writer = FileWriter::new(fw_tag.clone());
                self.all_writers.insert(fw_tag.clone(), new_writer);
            }
            let fw = self.all_writers.get_mut(&fw_tag).expect("");
            fw.add_summary(scalar(main_tag, *scalar_value), step);
        }
    }

    pub fn export_scalars_to_json(&self) {unimplemented!();}
    pub fn add_histogram(&mut self) {unimplemented!();}
    pub fn add_histogram_raw(&mut self) {unimplemented!();}
    pub fn add_image(&mut self, tag: &str, data: &[u8], dim: &[usize], step: usize) {
        self.writer.add_summary(image(tag, data, dim), step);
    }
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

    pub fn flush(&mut self) {
        self.writer.flush();
    }
}
