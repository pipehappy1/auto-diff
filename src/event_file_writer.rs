use std::path::{PathBuf, Path};
use std::fs;
use std::time::SystemTime;
use gethostname::gethostname;
use std::process::id;
use std::fs::File;
use protobuf::Message;

use crate::proto::event::Event;
use crate::record_writer::RecordWriter;

pub struct EventFileWriter {
    logdir: PathBuf,
    writer: RecordWriter<File>,
}
impl EventFileWriter {
    //pub fn new<P: AsRef<Path>>(logdir: P) -> EventFileWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> EventFileWriter {
        let logdir = logdir.as_ref().to_path_buf();

        fs::create_dir_all(&logdir).expect("");

        let mut time = 0;
        let mut time_full = 0.0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time = n.as_secs();
            time_full = n.as_secs_f64();
        }
        let hostname = gethostname().into_string().expect("");
        let pid = id();
        
        let file_name = format!("events.out.tfevents.{:010}.{}.{}.{}", time, hostname, pid, 0);
        let file_writer = File::create(logdir.join(file_name)).expect("");
        let writer = RecordWriter::new(file_writer);
        
        let mut ret = EventFileWriter {
            logdir: logdir,
            writer: writer,
        };

        let mut evn = Event::new();
        evn.set_wall_time(time_full);
        evn.set_file_version("brain.Event:2".to_string());
        ret.add_event(&evn).expect("");
        ret.flush().expect("");

        ret
    }
}

impl EventFileWriter {
    pub fn get_logdir(&self) -> PathBuf {
        self.logdir.to_path_buf()
    }
    
    pub fn add_event(&mut self, event: &Event) -> std::io::Result<()> {
        let mut data: Vec<u8> = Vec::new();
        event.write_to_vec(&mut data).expect("");
        self.writer.write(&data)
    }
    
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
