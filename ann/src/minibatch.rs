use ::rand::prelude::StdRng;
use auto_diff::{Var, AutoDiffError};
use auto_diff_data_pipe::dataloader::{DataLoader, DataSlice};

pub struct MiniBatch {
    rng: StdRng,
    size: usize,
}
impl MiniBatch {
    pub fn new(rng: StdRng, size: usize) -> MiniBatch {
        MiniBatch {
            rng,
            size,
        }
    }

    pub fn batch_size(&self) -> usize {
	self.size
    }

    /// Get a random set of samples from the data loader.
    pub fn next(&mut self, loader: &dyn DataLoader, part: &DataSlice) -> Result<(Var, Var), AutoDiffError> {
        let sample_size = loader.get_size(Some(*part))?[0];
        let index_t = Var::rand_usize(&mut self.rng, &[self.size], 0, sample_size);
        loader.get_indexed_batch(&(Vec::<usize>::try_from(index_t)?), Some(*part))
    }
    /// Get a random set of samples given the data and label.
    pub fn next_data_slice(&mut self, data: &Var, label: &Var) -> Result<(Var, Var), AutoDiffError> {
        let sample_size = data.size()[0];
        let sample_size2 = label.size()[0];

        if sample_size != sample_size2 {
            return Err(AutoDiffError::new(&format!("minibatch needs data and label has the same N {}, {}",
                                                   sample_size, sample_size2)));
        }
        let index_t = Var::rand_usize(&mut self.rng, &[self.size], 0, sample_size);

        let mdata = data.index_select(0, index_t.clone())?;
        let mlabel = label.index_select(0, index_t)?;
        mdata.reset_net();
        mlabel.reset_net();
        Ok((mdata, mlabel))
    }

    pub fn iter_block<'a>(&self, loader: &'a dyn DataLoader, part: & DataSlice) -> Result<BlockIterator<'a>, AutoDiffError> {
	Ok(BlockIterator {
            loader,
	    part: *part,
	    block_size: self.size,
	    block_index: 0,
        })
    }
}

pub struct BlockIterator<'a> {
    loader: &'a dyn DataLoader,
    part: DataSlice,
    block_size: usize,
    block_index: usize,
}
impl<'a> Iterator for BlockIterator<'a> {
    type Item = (Var, Var);
    fn next(&mut self) -> Option<Self::Item> {
	let n = if let Ok(size) = self.loader.get_size(Some(self.part)) {
	    size[0]
	} else {
	    return None;
	};

	if self.block_index >= n {
	    return None;
	}
	let mut end_index = self.block_index + self.block_size;
	if end_index > n {
	    end_index = n;
	}

	let result = self.loader.get_batch(self.block_index,
					   end_index,
					   Some(self.part));
	self.block_index += self.block_size;
	result.ok()
    }
}
