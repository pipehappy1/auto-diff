use crate::{DataLoader, Slice};
use auto_diff::{Var, AutoDiffError};

use std::path::{Path, PathBuf};
use std::io;
use std::fs::File;
use std::io::Read;

pub struct Mnist {
    path: PathBuf,
    train: Var,
    test: Var,
    train_label: Var,
    test_label: Var,
}
impl Mnist {
    pub fn new() -> Mnist {
        unimplemented!()
    }
    pub fn load(path: &Path) -> Mnist {
	let train_img = Self::load_images("examples/data/mnist/train-images-idx3-ubyte");
	let test_img = Self::load_images("examples/data/mnist/t10k-images-idx3-ubyte");
	let train_label = Self::load_labels("examples/data/mnist/train-labels-idx1-ubyte");
	let test_label = Self::load_labels("examples/data/mnist/t10k-labels-idx1-ubyte");
	
        Mnist {
            path: PathBuf::from(path),
	    train: train_img,
	    test: test_img,
	    train_label: train_label,
	    test_label: test_label,
        }
    }

    fn load_images<P: AsRef<Path>>(path: P) -> Var {
        let ref mut reader = io::BufReader::new(File::open(path).expect(""));
        let magic = Self::read_as_u32(reader);
        if magic != 2051 {
            panic!("Invalid magic number. expected 2051, got {}", magic)
        }
        let num_image = Self::read_as_u32(reader) as usize;
        let rows = Self::read_as_u32(reader) as usize;
        let cols = Self::read_as_u32(reader) as usize;
        assert!(rows == 28 && cols == 28);
    
        // read images
        let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
        let _ = reader.read_exact(buf.as_mut());
        let ret: Vec<f64> = buf.into_iter().map(|x| (x as f64) / 255.).collect();
        let ret = Var::new(&ret[..], &vec![num_image, rows, cols]);
        ret
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> Var {
        let ref mut reader = io::BufReader::new(File::open(path).expect(""));
        let magic = Self::read_as_u32(reader);
        if magic != 2049 {
            panic!("Invalid magic number. Got expect 2049, got {}", magic);
        }
        let num_label = Self::read_as_u32(reader) as usize;
        // read labels
        let mut buf: Vec<u8> = vec![0u8; num_label];
        let _ = reader.read_exact(buf.as_mut());
        let ret: Vec<f64> = buf.into_iter().map(|x| x as f64).collect();
        let ret = Var::new(&ret[..], &vec![num_label]);
        ret
    }

    fn read_as_u32<T: Read>(reader: &mut T) -> u32 {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        let _ = reader.read_exact(&mut buf);
        u32::from_be_bytes(buf)
    }
}
impl DataLoader for Mnist {
    fn get_size(&self, slice: Option<Slice>) -> Option<Vec<usize>> {
        match slice {
	    Some(Slice::Train) => {Some(self.train.size())},
	    Some(Slice::Test) => {Some(self.test.size())},
	    None => {Some(vec![self.train.size()[0] + self.test.size()[0],
			  self.train.size()[1]])}
	    _ => {None}
	}
    }
    fn get_item(&self, index: usize, slice: Option<Slice>) -> Result<(Var, Var), AutoDiffError> {
        match slice {
	    Some(Slice::Train) => {Some(self.train.get_patch((index, index+1),()))},
	    Some(Slice::Test) => {Some(self.test.get_patch((index, index+1),()))},
	    
	}
    }
    fn get_batch(&self, start: usize, end: usize, slice: Option<Slice>) -> Result<(Var, Var), AutoDiffError> {
        unimplemented!()
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn it_works() {
        let mnist = Mnist::load(Path::new("../auto-diff/examples/data/mnist/"));

    }
}

