use super::gen_tensor::GenTensor;

pub trait IndexSlicing {
    type TensorType;

    fn squeeze(&self, dim: Option<usize>) -> Self::TensorType;

    fn gather();
    fn index_select(&self, );
    fn unsqueeze(&self, dim: usize) -> Self::TensorType;
}

impl<T> IndexSlicing for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;

    fn squeeze(&self, dim: Option<usize>) -> Self::TensorType {
        GenTensor::<T>::new()
    }

    fn gather() {}
    fn index_select(&self, ) {}
    
    fn unsqueeze(&self, dim: usize) ->  Self::TensorType {
        let mut new_dim = Vec::new();
        for i in 0..self.size().len() {
            if i == dim {
                new_dim.push(1);
            }
            new_dim.push(self.size()[i]);
        }
        GenTensor::new_raw(&self.get_data(), &new_dim)
    }
}
