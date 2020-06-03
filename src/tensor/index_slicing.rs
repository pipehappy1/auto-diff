use std::ops::Range;
use super::gen_tensor::GenTensor;

pub trait IndexSlicing {
    type TensorType;

    fn cat(&self, tensors: &[&Self::TensorType], dim: usize) -> Self::TensorType;
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self::TensorType>;
    fn gather();
    fn index_select(&self, );
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}
    fn reshape(&self, new_shape: &[usize]) -> Self::TensorType;
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self::TensorType>;
    fn squeeze(&self, dim: Option<usize>) -> Self::TensorType;
    fn stack(tensors: &[&Self], dim: usize) -> Self::TensorType;
    //pub fn t() {}
    fn take(&self, index: &[usize]) -> Self::TensorType;
    //pub fn transpose() {}
    //pub fn unbind() {}
    fn permute(&self, dims: &[usize]) -> Self::TensorType;
    fn unsqueeze(&self, dim: usize) -> Self::TensorType;
    //pub fn condition() {} // this is pytorch where
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

    /// Splits a tensor into a specific number of chunks.
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self::TensorType> {
        //let mut ret = Vec::new();
        //let mut chunk_size = self.dim[dim] / chunks;
        //if self.dim[dim] % chunks > 0 {
        //    chunk_size += 1;
        //}
        //let mut start;
        //let mut end;
        //for i in 0..chunks {
        //    start = i*chunk_size;
        //    end = (i+1)*chunk_size;
        //    if end > self.dim[dim] {
        //        end = self.dim[dim];
        //    }
        //    
        //}
        //ret
        unimplemented!();
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

    /// Concatenates sequence of tensors along a new dimension.
    ///
    /// All tensors need to be of the same size.
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// # use crate::auto_diff::tensor::index_slicing::IndexSlicing;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![3,2]);
    /// let result = GenTensor::<f64>::stack(&vec![&m1, &m2], 1);
    /// let raw = result.get_raw();
    /// for i in raw {
    ///     println!("{}", i);
    /// }
    /// assert_eq!(result.size(), vec![3,2,2]);
    /// ```
    fn stack(tensors: &[&Self], dim: usize) -> Self::TensorType {
        
        let cap = tensors.len()*tensors[0].numel();
        let mut odim = Vec::new();
        for i in 0..tensors[0].size().len() {
            if i == dim {
                odim.push(tensors.len());
            }
            odim.push(tensors[0].size()[i]);
        }
        if odim.len() == tensors[0].size().len() {
            odim.push(tensors.len());
        }
        
        
        let mut d = Vec::with_capacity(cap);
        
        let mut outter_loop = 1;
        let mut inner_loop = 1;
        for i in 0..tensors[0].size().len() {
            if i < dim {
                outter_loop *= tensors[0].size()[i];
            } else {
                inner_loop *= tensors[0].size()[i];
            }
        }
        for i in 0..outter_loop {
            for j in 0..tensors.len() {
                for k in 0..inner_loop {
                    d.push(tensors[j].get_data()[k + i*inner_loop]);
                }
            }
        }
        GenTensor::new_raw(&d, &odim)
    }

    /// Returns a new tensor with the elements of input at the given indices. 
    /// The input tensor is treated as if it were viewed as a 1-D tensor.
    /// The result takes the same shape as the indices.
    fn take(&self, index: &[usize]) -> Self::TensorType {
        let mut ret = Vec::<T>::with_capacity(index.len());
        for i in index {
            ret.push(self.get_data()[*i]);
        }
        GenTensor::new_raw(&ret, &vec![index.len()])
    }

    /// Permute the dimensions of this tensor.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// # use crate::auto_diff::tensor::index_slicing::IndexSlicing;
    /// let mut m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
    /// m1.permute(&vec![2, 0, 1]);
    /// ```
    fn permute(&self, dims: &[usize]) -> Self::TensorType {

        let ret_d = self.get_data().to_vec();
        let ret_dim = self.size();

        let dim_len = ret_dim.len();
        let mut target_dim = vec![0; dim_len];
        for i in 0..dim_len {
            target_dim[i] = ret_dim[dims[i]];
        }

        let mut new_d = ret_d.to_vec();
        let mut index = vec![0; dim_len];
        let mut old_index = vec![0; dim_len];
        let old_stride = self.stride();
        let ret_dim = target_dim.to_vec();
        let ret = GenTensor::new_raw(&ret_d, &ret_dim);
        let new_stride = ret.stride();
        for _i in 0..ret_d.len() {
            for j in 0..dim_len {
                old_index[dims[j]] = index[j];
            }

            let mut item_index = 0;
            let mut new_item_index = 0;
            for j in 0..dim_len {
                item_index += old_stride[j]*old_index[j];
                new_item_index += new_stride[j]*index[j];
            }
            new_d[new_item_index] = ret_d[item_index];
            
            index[dim_len-1] += 1;
            let mut next_dim = dim_len-1;
            while index[next_dim] >= target_dim[next_dim] {
                if next_dim <= 0 {
                    break
                } else {
                    index[next_dim] = 0;
                    index[next_dim-1] += 1;
                    next_dim -= 1;                    
                }

            }

        }
        GenTensor::new_raw(&new_d, &ret_dim)
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

    #[test]
    fn stack() {}

    #[test]
    fn permute() {
        let m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
        let m11 = m1.permute(&vec![2, 0, 1]);
        assert_eq!(m11.size(), vec![5, 2, 3]);

        let m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
        let m22 = m2.permute(&vec![1, 0]);
        assert_eq!(m22.get_raw(), vec![1., 3., 2., 4.]);
    }
}
