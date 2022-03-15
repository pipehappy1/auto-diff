use auto_diff::{Var, AutoDiffError};

pub enum DataSlice {
    Train,
    Test,
    Tune,
    Other,
}

pub trait DataLoader {
    fn get_size(&self, slice: Option<DataSlice>) -> Result<Vec<usize>, AutoDiffError>;
    fn get_item(&self, index: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError>;
    fn get_batch(&self, start: usize, end: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError>;
    fn get_indexed_batch(&self, index: &[usize], slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError>;
}

pub mod mnist;
