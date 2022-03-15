use auto_diff::{Var, AutoDiffError};

pub enum Slice {
    Train,
    Test,
    Tune,
    Other,
}

pub trait DataLoader {
    fn get_size(&self, slice: Option<Slice>) -> Option<Vec<usize>>;
    fn get_item(&self, index: usize, slice: Option<Slice>) -> Result<(Var, Var), AutoDiffError>;
    fn get_batch(&self, start: usize, end: usize, slice: Option<Slice>) -> Result<(Var, Var), AutoDiffError>;
}

pub mod mnist;
