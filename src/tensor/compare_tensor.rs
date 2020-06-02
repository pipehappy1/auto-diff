use crate::tensor::gen_tensor::GenTensor;

pub trait CompareTensor {
    type TensorType;

    fn min(&self, o: Option<&Self::TensorType>, dim: Option<usize>, keep_dim: Option<bool>) -> Self::TensorType;
    fn max(&self, o: Option<&Self::TensorType>, dim: Option<usize>, keep_dim: Option<bool>) -> Self::TensorType;
}

impl<T> CompareTensor for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;
    
    fn min(&self, o: Option<&Self::TensorType>, dim: Option<usize>, keep_dim: Option<bool>) -> Self::TensorType {
        if o.is_none() && dim.is_none() && keep_dim.is_none() {
            self.min_all()
        } else if o.is_some() && dim.is_none() && keep_dim.is_none() {
            self.min_pair(o.unwrap())
        } else if o.is_none() && dim.is_some() {
            let dim_to_min = dim.unwrap();
            let keep_dim_bool;
            if keep_dim.is_some() {
                keep_dim_bool = keep_dim.unwrap();
            } else {
                keep_dim_bool = true;
            }
            
            self.min_along(dim_to_min, keep_dim_bool)
        } else {
            panic!("min expect either o or dim, not both");
        }
    }
    fn max(&self, o: Option<&Self::TensorType>, dim: Option<usize>, keep_dim: Option<bool>) -> Self::TensorType {
        if o.is_none() && dim.is_none() && keep_dim.is_none() {
            self.max_all()
        } else if o.is_some() && dim.is_none() && keep_dim.is_none() {
            self.max_pair(o.unwrap())
        } else if o.is_none() && dim.is_some() {
            let dim_to_max = dim.unwrap();
            let keep_dim_bool;
            if keep_dim.is_some() {
                keep_dim_bool = keep_dim.unwrap();
            } else {
                keep_dim_bool = true;
            }
            
            self.max_along(dim_to_max, keep_dim_bool)
        } else {
            panic!("max/min expect either o or dim, not both");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use super::*;
    
    #[test]
    fn max() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.max(Some(&b), None, None);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![2., 4., 10., 11.], &vec![2,2]));
    }

    #[test]
    fn min() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.min(Some(&b), None, None);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![1., 3., 5., 6.], &vec![2,2]));
    }
}
