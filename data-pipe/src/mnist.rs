use crate::DataLoader;
use auto_diff::{Var, AutoDiffError};

use std::path::{Path, PathBuf};

pub struct Mnist {
    path: PathBuf,
}
impl Mnist {
    pub fn new() -> Mnist {
        unimplemented!()
    }
    pub fn load(path: &Path) -> Mnist {
        Mnist {
            path: PathBuf::from(path),
        }
    }
}
impl DataLoader for Mnist {
    fn get_size(&self) -> Vec<usize> {
        unimplemented!()
    }
    fn get_item(&self, index: usize) -> Result<(Var, Var), AutoDiffError> {
        unimplemented!()
    }
    fn get_batch(&self, start: usize, end: usize) -> Result<(Var, Var), AutoDiffError> {
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

