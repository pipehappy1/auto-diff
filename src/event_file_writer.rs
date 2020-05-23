use std::path::{Path, PathBuf};
use std::fs;
use std::time::SystemTime;

use crate::record_writer::RecordWriter;

pub struct EventFileWriter {
    logdir: PathBuf,
}
impl EventFileWriter {
    //pub fn new<P: AsRef<Path>>(logdir: P) -> EventFileWriter {
    pub fn new<P: AsRef<PathBuf>>(logdir: P) -> EventFileWriter {
        let logdir = logdir.as_ref().to_path_buf();

        fs::create_dir_all(&logdir).expect("");

        let mut time = 0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time = n.as_secs();
        }
        
        format!("events.out.tfevents.{:010}.{}.{}.{}", time, 0, 0, 0);
        
        EventFileWriter {
            logdir: logdir,
        }
    }
}
