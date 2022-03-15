use crate::dataloader::{DataLoader, DataSlice};
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
        // TODO download the data if it is not there.
        unimplemented!()
    }
    pub fn load(path: &Path) -> Mnist {
        let train_fn = path.join("train-images-idx3-ubyte");
        let test_fn = path.join("t10k-images-idx3-ubyte");
        let train_label = path.join("train-labels-idx1-ubyte");
        let test_label = path.join("t10k-labels-idx1-ubyte");
        
	let train_img = Self::load_images(train_fn);
	let test_img = Self::load_images(test_fn);
	let train_label = Self::load_labels(train_label);
	let test_label = Self::load_labels(test_label);
	
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
    fn get_size(&self, slice: Option<DataSlice>) -> Result<Vec<usize>, AutoDiffError> {
        match slice {
	    Some(DataSlice::Train) => {Ok(self.train.size())},
	    Some(DataSlice::Test) => {Ok(self.test.size())},
	    None => {
                let n = self.train.size()[0] + self.test.size()[1];
                let mut new_size = self.train.size();
                new_size[0] = n;
                Ok(new_size)
            },
	    _ => {Err(AutoDiffError::new("TODO"))}
	}
    }
    fn get_item(&self, index: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError> {
        match slice {
	    Some(DataSlice::Train) => {
                let dim = self.train.size().len();
                let mut index_block = vec![(index, index+1)];
                index_block.append(
                    &mut vec![0; dim-1].iter().zip(&self.train.size()[1..])
                        .map(|(x,y)| (*x, *y)).collect());
                let data = self.train.get_patch(&index_block, None)?;
                let label = self.train_label.get_patch(&[(index, index+1)], None)?;
                return Ok((data, label));
            },
	    Some(DataSlice::Test) => {
                let dim = self.test.size().len();
                let mut index_block = vec![(index, index+1)];
                index_block.append(
                    &mut vec![0; dim-1].iter().zip(&self.test.size()[1..])
                        .map(|(x,y)| (*x, *y)).collect());
                let data = self.test.get_patch(&index_block, None)?;
                let label = self.test_label.get_patch(&[(index, index+1)], None)?;
                return Ok((data, label));
            },
	    _ => {Err(AutoDiffError::new("only train and test"))}
	}
    }
    fn get_batch(&self, start: usize, end: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError> {
        match slice {
	    Some(DataSlice::Train) => {
                let dim = self.train.size().len();
                let mut index_block = vec![(start, end)];
                index_block.append(
                    &mut vec![0; dim-1].iter().zip(&self.train.size()[1..])
                        .map(|(x,y)| (*x, *y)).collect());
                let data = self.train.get_patch(&index_block, None)?;
                let label = self.train_label.get_patch(&[(start, end)], None)?;
                return Ok((data, label));
            },
	    Some(DataSlice::Test) => {
                let dim = self.test.size().len();
                let mut index_block = vec![(start, end)];
                index_block.append(
                    &mut vec![0; dim-1].iter().zip(&self.test.size()[1..])
                        .map(|(x,y)| (*x, *y)).collect());
                let data = self.test.get_patch(&index_block, None)?;
                let label = self.test_label.get_patch(&[(start, end)], None)?;
                return Ok((data, label));
            },
	    _ => {Err(AutoDiffError::new("only train and test"))}
	}
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

