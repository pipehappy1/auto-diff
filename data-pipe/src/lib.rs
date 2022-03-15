//! A data loader for machine learning.
use auto_diff::{Var, AutoDiffError};

pub trait DataLoader {
    fn get_size(&self) -> Vec<usize>;
    fn get_item(&self, index: usize) -> Result<(Var, Var), AutoDiffError>;
    fn get_batch(&self, start: usize, end: usize) -> Result<(Var, Var), AutoDiffError>;
    fn get
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
