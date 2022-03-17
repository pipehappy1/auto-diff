use auto_diff::{Var, AutoDiffError};

#[derive(Copy, Clone)]
pub enum DataSlice {
    Train,
    Test,
    Tune,
    Other,
}

pub trait DataLoader {
    /// The shape of the data if applicable.
    fn get_size(&self, slice: Option<DataSlice>) -> Result<Vec<usize>, AutoDiffError>;
    /// Return one sample.
    fn get_item(&self, index: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError>;
    /// Return a batch following original order.
    fn get_batch(&self, start: usize, end: usize, slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError>;
    /// Return a batch given the index.
    fn get_indexed_batch(&self, index: &[usize], slice: Option<DataSlice>) -> Result<(Var, Var), AutoDiffError> {
        let mut data: Vec<Var> = vec![];
        let mut label: Vec<Var> = vec![];

        for elem_index in index {
            let (elem_data, elem_label) = self.get_item(*elem_index, slice)?;
            data.push(elem_data);
            label.push(elem_label);
        }
        let d1 = data[0].cat(&data[1..], 0)?;
        let d2 = label[0].cat(&label[1..], 0)?;
        d1.reset_net();
        d2.reset_net();
        Ok((d1, d2))
    }
}

pub mod mnist;
