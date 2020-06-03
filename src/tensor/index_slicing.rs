use std::ops::Range;
use super::gen_tensor::GenTensor;

pub trait IndexSlicing {
    type TensorType;

    fn cat(&self, tensors: &[&Self::TensorType], dim: usize) -> Self::TensorType;
    fn gather();
    fn index_select(&self, );
    fn reshape(&self, new_shape: &[usize]) -> Self::TensorType;
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self::TensorType>;
    fn squeeze(&self, dim: Option<usize>) -> Self::TensorType;
    fn unsqueeze(&self, dim: usize) -> Self::TensorType;
}

impl<T> IndexSlicing for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;

    /// Concatenates the given sequence of seq tensors in the given dimension.
    fn cat(&self, tensors: &[&Self::TensorType], dim: usize) -> Self::TensorType {
        let total_dim = self.size();
        for i in tensors {
            if i.size().len() != total_dim.len() {
                panic!("cat needs all have the same number of dimens {}, {}", total_dim.len(), i.size().len());
            }
            for (j, index) in i.size().iter().zip(0..total_dim.len()) {
                if index != dim && total_dim[index] != *j {
                    panic!("cat needs all but the cat dim to have the same dim {:?}, {:?}", total_dim, i.size());
                }
            }
        }

        let mut cap = 1;
        let mut new_dim = Vec::new();
        let mut outer_size = 1;
        let mut inner_size = 1;
        for i in 0..total_dim.len() {
            if i != dim {
                cap *= total_dim[i];
                new_dim.push(total_dim[i]);                
            } else {
                let mut dim_total = total_dim[i];
                for j in tensors {
                    dim_total += j.size()[i];
                }
                cap *= dim_total;
                new_dim.push(dim_total);
            }
            if i < dim {
                outer_size *= total_dim[i];
            }
            if i > dim {
                inner_size *= total_dim[i];
            }
        }
        let mut data = Vec::with_capacity(cap);
        unsafe{ data.set_len(cap); }

        let mut ret_range = Range{start: 0, end: total_dim[dim]*inner_size};
        for i in 0..outer_size {
            data[ret_range.clone()].clone_from_slice(&self.get_data()[i*total_dim[dim]*inner_size..(i+1)*total_dim[dim]*inner_size]);
            
            //println!("outer: {:?}", ret_range);
            
            for j in tensors {
                ret_range = Range{start: ret_range.end, end: ret_range.end + j.size()[dim]*inner_size};
                data[ret_range.clone()].clone_from_slice(&j.get_data()[i*j.size()[dim]*inner_size..(i+1)*j.size()[dim]*inner_size]);
                //println!("inner: {:?}", ret_range);
            }

            ret_range = Range{start: ret_range.end, end: ret_range.end + total_dim[dim]*inner_size};
        }

        GenTensor::new_raw(&data, &new_dim)
    }



    fn gather() {}
    fn index_select(&self, ) {}

    fn reshape(&self, new_shape: &[usize]) -> Self::TensorType {
        if self.size().iter().product::<usize>() != new_shape.iter().product::<usize>() {
            panic!("reshape expects the same number of elements {:?}, {:?}", self.size(), new_shape);
        }
        GenTensor::new_raw(&self.get_data(), new_shape)
    }
    
    /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self::TensorType> {
        let total_dim = self.size();
        if sections.iter().sum::<usize>() != total_dim[dim] {
            panic!("sum of sections should be the size on dim.");
        }

        let mut outer_size = 1;
        let mut inner_size = 1;
        for (i, index) in total_dim.iter().zip(0..total_dim.len()) {
            if index < dim {
                outer_size *= i;
            }
            if index > dim {
                inner_size *= i;
            }
        }
        
        let mut ret = Vec::new();
        for i in sections {
            let mut t_size = Vec::new();
            for (j, index) in total_dim.iter().zip(0..total_dim.len()) {
                if index == dim {
                    t_size.push(*i);
                } else {
                    t_size.push(*j);
                }
            }
            let mut t = GenTensor::empty(&t_size);
            ret.push(t);
        }

        for i in 0..outer_size {
            let mut start = 0;
            for (j, index) in ret.iter_mut().zip(0..sections.len()) {
                j.get_mut_data()[i*inner_size*sections[index]..(i+1)*inner_size*sections[index]].clone_from_slice(
                    &self.get_data()[i*inner_size*total_dim[dim] + start..i*inner_size*total_dim[dim] + start + sections[index]*inner_size]);
                start += sections[index]*inner_size;
            }
        }
        
        ret
    }
    
    fn squeeze(&self, dim: Option<usize>) -> Self::TensorType {
        GenTensor::<T>::new()
    }
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

#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use super::*;
    
    #[test]
    fn cat() {
        let a = GenTensor::<f32>::fill(1., &vec![5, 3, 3, 2]);
        let b = GenTensor::<f32>::fill(2., &vec![5, 3, 3, 2]);
        let c = GenTensor::<f32>::fill(3., &vec![5, 3, 3, 2]);

        let d = a.cat(&vec![&b, &c], 1);
        //println!("{}", d);
        assert_eq!(d, GenTensor::new_raw(&vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], &vec![5, 9, 3, 2]));
    }

    #[test]
    fn split() {
        let a = GenTensor::<f32>::fill(1., &vec![5, 3, 3, 2]);
        let b = GenTensor::<f32>::fill(2., &vec![5, 3, 3, 2]);
        let c = GenTensor::<f32>::fill(3., &vec![5, 3, 3, 2]);
    
        let d = a.cat(&vec![&b, &c], 1);
    
        let secs = vec![3, 3, 3];
        let tensors = d.split(&secs, 1);
        //println!("{}, \n{}, \n{}", tensors[0], tensors[1], tensors[2]);
        assert_eq!(tensors[0], a);
        assert_eq!(tensors[1], b);
        assert_eq!(tensors[2], c);
    }
}
