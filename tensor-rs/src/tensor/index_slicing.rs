use std::ops::Range;
use std::cmp;
use super::gen_tensor::GenTensor;
#[cfg(feature = "use-cuda")]
use super::cuda_tensor::CudaTensor;

pub trait IndexSlicing where Self: std::marker::Sized {

    fn cat(&self, tensors: &[&Self], dim: usize) -> Self;
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self>;
    fn gather(&self, dim: usize, index: &Self) -> Self;
    fn index_select(&self, dim: usize, index: &Self) -> Self;
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}
    fn reshape(&self, new_shape: &[usize]) -> Self;
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self>;
    fn squeeze(&self, dim: Option<usize>) -> Self;
    fn stack(tensors: &[&Self], dim: usize) -> Self;
    fn t(&self) -> Self;
    fn take(&self, index: &[usize]) -> Self;
    //pub fn transpose() {}
    //pub fn unbind() {}
    fn permute(&self, dims: &[usize]) -> Self;
    fn unsqueeze(&self, dim: usize) -> Self;
    //pub fn condition() {} // this is pytorch where
    fn conditional_select(&self, x: &Self, y: &Self) -> Self;

    fn repeat(&self, sizes: &[usize]) -> Self;
}

impl<T> IndexSlicing for GenTensor<T> where T: num_traits::Float {

    /// Concatenates the given sequence of seq tensors in the given dimension.
    fn cat(&self, tensors: &[&Self], dim: usize) -> Self {
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
        for (i, item) in total_dim.iter().enumerate() {
            if i != dim {
                cap *= *item;
                new_dim.push(*item);
            } else {
                let mut dim_total = *item;
                for j in tensors {
                    dim_total += j.size()[i];
                }
                cap *= dim_total;
                new_dim.push(dim_total);
            }
            if i < dim {
                outer_size *= *item;
            }
            if i > dim {
                inner_size *= *item;
            }
        }
        let mut data = vec![T::zero(); cap];

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
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self> {
        let mut ret = Vec::new();
        let mut chunk_size = self.size()[dim] / chunks;
        if self.size()[dim] % chunks > 0 {
            chunk_size += 1;
        }
        
        for i in 0..chunks {
            let start = i*chunk_size;
            let end = cmp::min((i+1)*chunk_size, self.size()[dim]);

            let mut start_index = vec![0; self.size().len()];
            let mut end_index = self.size().to_vec();
            start_index[dim] = start;
            end_index[dim] = end;
            let range: Vec<(usize, usize)> = start_index.iter().zip(end_index.iter()).map(|x| (*x.0, *x.1)).collect();
            ret.push(self.get_patch(&range, None));
        }
        ret
    }

    fn gather(&self, dim: usize, index: &Self) -> Self {
        if self.size().len() != index.size().len() {
            panic!("gather need two input has the same number of dim: {}, {}", self.size().len(), index.size().len());
        }
        for i in 0..self.size().len() {
            if i != dim && self.size()[i] != index.size()[i] {
                panic!("gather want the same shape (but see {:?}, {:?}) except dim {}", self.size(), index.size(), dim);
            }
        }
        
        let mut ret = index.clone();

        //let total_task: usize = index.size().iter().filter(|x| **x != dim).product();
        let mut outer_index = vec![0; dim];
        let mut inner_index = vec![0; index.size().len()-dim -1 ];

        loop {
            //println!("outer_index, {:?}, inner_index: {:?}", outer_index, inner_index);

            let mut outer_seg: Vec<(usize, usize)> = outer_index.iter().map(|x| (*x, x+1)).collect();
            outer_seg.push((0, self.size()[dim]));
            let mut inner_seg: Vec<(usize, usize)> = inner_index.iter().map(|x| (*x, x+1)).collect();
            outer_seg.append(&mut inner_seg);
            //println!("outer_seg {:?}", outer_seg);
            let current_scope = self.get_patch(&outer_seg, None);

            for i in 0..index.size()[dim] {
                let mut current_index = outer_index.to_vec();
                current_index.push(i);
                current_index.append(&mut inner_index.to_vec());

                let gather_index = index.get(&current_index).to_usize().expect("");
                ret.set(&current_index, current_scope.get_data()[gather_index]);
            }
            
            let mut can_continue = false;
            for i in 0..index.size().len()-dim-1 {
                inner_index[dim - i - 1] += 1;
                if inner_index[dim - i - 1] >= index.size()[index.size().len() - i -1] {
                    inner_index[dim - i - 1] = 0;
                } else {
                    can_continue = true;
                    break;
                }
            }
            if can_continue {
                continue;
            }

            for i in 0..dim {
                outer_index[dim-i-1] += 1;
                if outer_index[dim-i-1] >= index.size()[dim - i - 1] {
                    outer_index[dim-i-1] = 0;
                } else {
                    break
                }
            }
            
            if inner_index == vec![0; index.size().len()-dim -1 ] && outer_index == vec![0; dim] {
                break;
            }
        };
        ret
    }
    fn index_select(&self, dim: usize, index: &Self) -> Self {
        if dim >= self.size().len() {
            panic!("index_select needs better dim: {:?}, {:?}", self.size(), dim);
        }
        if index.size().len() > 1 {
            panic!("index_select needs 1-D index, get: {:?}", index.size());
        }
        for i in index.get_data() {
            if *i >= T::from(self.size()[dim]).expect("") {
                panic!("index_select gets out of range number at dim {:?}, given {:?}", self.size()[dim], i.to_usize());
            }
        }

        let mut ret_dim = self.size().to_vec();
        ret_dim[dim] = index.numel();
        let mut ret = GenTensor::zeros(&ret_dim);

        for (row_index, i) in index.get_data().iter().enumerate() {
            let mut start = vec![0; self.size().len()];
            let mut end = self.size().to_vec();
            start[dim] = i.to_usize().expect("");
            end[dim] = start[dim] + 1;

            let mut ret_start = start.to_vec();
            let mut ret_end = end.to_vec();
            ret_start[dim] = row_index;
            ret_end[dim] = row_index + 1;

            let range: Vec::<(usize, usize)> = start.iter().zip(end.iter()).map(|x| (*x.0, *x.1)).collect();
            let ret_range: Vec::<(usize, usize)> = ret_start.iter().zip(ret_end.iter()).map(|x| (*x.0, *x.1)).collect();
            let patch = self.get_patch(&range, None);
            ret.set_patch(&patch, &ret_range, None);
        }
        
        ret
    }

    fn reshape(&self, new_shape: &[usize]) -> Self {
        if self.size().iter().product::<usize>() != new_shape.iter().product::<usize>() {
            panic!("reshape expects the same number of elements {:?}, {:?}", self.size(), new_shape);
        }
        GenTensor::new_raw(self.get_data(), new_shape)
    }
    
    /// Splits the tensor into chunks. Each chunk is a view of the original tensor.
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self> {
        let total_dim = self.size();
        if sections.iter().sum::<usize>() != total_dim[dim] {
            panic!("sum of sections should be the size on dim.");
        }

        let mut outer_size = 1;
        let mut inner_size = 1;
        for (index, i) in total_dim.iter().enumerate() {
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
            for (index, j) in total_dim.iter().enumerate() {
                if index == dim {
                    t_size.push(*i);
                } else {
                    t_size.push(*j);
                }
            }
            let t = GenTensor::zeros(&t_size);
            ret.push(t);
        }

        for i in 0..outer_size {
            let mut start = 0;
            for (j, index) in ret.iter_mut().zip(0..sections.len()) {
                j.get_data_mut()[i*inner_size*sections[index]..(i+1)*inner_size*sections[index]].clone_from_slice(
                    &self.get_data()[i*inner_size*total_dim[dim] + start..i*inner_size*total_dim[dim] + start + sections[index]*inner_size]);
                start += sections[index]*inner_size;
            }
        }
        
        ret
    }

    fn squeeze(&self, dim: Option<usize>) -> Self {
        let mut new_shape = Vec::new();
        for i in 0..new_shape.len() {
            if (new_shape[i] == 1 && dim.is_some() && i == dim.unwrap()) ||
                (new_shape[i] == 1 && dim.is_none()) {
                continue
            } else {
                new_shape.push(new_shape[i]);
            }
        }
        GenTensor::new_raw(self.get_data(), &new_shape)
    }

    /// Concatenates sequence of tensors along a new dimension.
    ///
    /// All tensors need to be of the same size.
    /// ```
    /// # use crate::tensor_rs::tensor::gen_tensor::*;
    /// # use crate::tensor_rs::tensor::index_slicing::IndexSlicing;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![3,2]);
    /// let result = GenTensor::<f64>::stack(&vec![&m1, &m2], 1);
    /// let raw = result.get_raw();
    /// for i in raw {
    ///     println!("{}", i);
    /// }
    /// assert_eq!(*result.size(), vec![3,2,2]);
    /// ```
    fn stack(tensors: &[&Self], dim: usize) -> Self {
        
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
            for j in tensors {
                for k in 0..inner_loop {
                    d.push(j.get_data()[k + i*inner_loop]);
                }
            }
        }
        GenTensor::new_raw(&d, &odim)
    }

    fn t(&self) -> Self {
        let n = self.size().len();
        let mut di: Vec<usize> = (0..n).collect();
        di[n-1] = n-2;
        di[n-2] = n-1;
        self.permute(&di)
    }

    /// Returns a new tensor with the elements of input at the given indices. 
    /// The input tensor is treated as if it were viewed as a 1-D tensor.
    /// The result takes the same shape as the indices.
    fn take(&self, index: &[usize]) -> Self {
        let mut ret = Vec::<T>::with_capacity(index.len());
        for i in index {
            ret.push(self.get_data()[*i]);
        }
        GenTensor::new_raw(&ret, &[index.len()])
    }

    /// Permute the dimensions of this tensor.
    ///
    /// ```
    /// # use tensor_rs::tensor::gen_tensor::*;
    /// # use crate::tensor_rs::tensor::index_slicing::IndexSlicing;
    /// let mut m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
    /// m1.permute(&vec![2, 0, 1]);
    /// ```
    fn permute(&self, dims: &[usize]) -> Self {

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
                if next_dim == 0 {
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
    
    fn unsqueeze(&self, dim: usize) ->  Self {
        let mut new_dim = Vec::new();
        for i in 0..self.size().len() {
            if i == dim {
                new_dim.push(1);
            }
            new_dim.push(self.size()[i]);
        }
        if dim == self.size().len() {
            new_dim.push(1);
        }
        GenTensor::new_raw(self.get_data(), &new_dim)
    }

    fn conditional_select(&self, x: &Self, y: &Self) -> Self {
        if self.size() != x.size() || self.size() != y.size() {
            panic!("condition_select expect the same size: {:?}, {:?}, {:?}", self.size(), x.size(), y.size());
        }
        let mut data = Vec::with_capacity(self.get_data().len());
        for ((i, j), k) in (self.get_data().iter().zip(x.get_data().iter())).zip(y.get_data().iter()) {
            if *i > T::zero() {
                data.push(*j);
            } else {
                data.push(*k);
            }
        }
        GenTensor::new_raw(&data, self.size())
    }

    fn repeat(&self, sizes: &[usize]) -> Self {
        if self.size().len() != sizes.len() {
            panic!("repeat needs the same number of dimensions. {:?}, {:?}", self.size().len(), sizes.len());
        }

        let mut ret_dim: Vec<usize> = Vec::with_capacity(self.size().len());
        for (i, j) in self.size().iter().zip(sizes.iter()) {
            ret_dim.push(i*j);
        }
        let mut ret = GenTensor::zeros(&ret_dim);

        for i in 0..ret_dim.iter().product() {
            let index = ret.index2dimpos(i);
            let value_index: Vec<usize> = index.iter().zip(self.size().iter()).map(|(x, y)| x % y).collect();
            //println!("index: {:?}, value_index: {:?}", index, value_index);
            ret.set(&index, self.get(&value_index));
        }

        ret
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
    fn gather() {
        let a = GenTensor::new_raw(&[1., 2., 3., 4.,], &[2, 2]);
        let g = GenTensor::new_raw(&[0., 0., 1., 0.,], &[2, 2]);
        let r = a.gather(1, &g);
        println!("{:?}", r);
        assert_eq!(r, GenTensor::new_raw(&[1., 1., 4., 3.,], &[2, 2]));
    }

    #[test]
    fn index_select() {
        let a = GenTensor::new_raw(&GenTensor::<f32>::arange(30).get_data(), &[2, 3, 5]);
        let b = a.index_select(0, &GenTensor::new_raw(&[0., 0., 1., 0., 0.], &[5]));
        println!("{:?}", b);
        assert_eq!(b, GenTensor::new_raw(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], &[5, 3, 5]));
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
        assert_eq!(*m11.size(), vec![5, 2, 3]);

        let m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
        let m22 = m2.permute(&vec![1, 0]);
        assert_eq!(m22.get_raw(), vec![1., 3., 2., 4.]);

        let m3 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4., 5., 6.], &vec![3, 2]);
        let m32 = m3.permute(&vec![1, 0]);
        assert_eq!(m32.get_raw(), vec![1., 3., 5., 2., 4., 6.]);
        assert_eq!(*m32.size(), vec![2, 3]);
    }

    #[test]
    fn repeat() {
        let a = GenTensor::new_raw(&[1., 2., 3.], &[1, 3]);
        let b = a.repeat(&[4, 2]);
        println!("{:?}", b);
        assert_eq!(b, GenTensor::<f32>::new_raw(&[1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3. ], &[4, 6]));
    }
}

/****************/
// Cuda tensor ops
/****************/
#[cfg(feature = "use-cuda")]
impl IndexSlicing for CudaTensor {
    fn cat(&self, tensors: &[&Self], dim: usize) -> Self {
        todo!();
    }
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self> {
        todo!();
    }
    fn gather(&self, dim: usize, index: &Self) -> Self {
        todo!();
    }
    fn index_select(&self, dim: usize, index: &Self) -> Self
    {
        todo!();
    }
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}
    fn reshape(&self, new_shape: &[usize]) -> Self {
        todo!();
    }
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self> {
        todo!();
    }
    fn squeeze(&self, dim: Option<usize>) -> Self {
        todo!();
    }
    fn stack(tensors: &[&Self], dim: usize) -> Self {
        todo!();
    }
    //pub fn t() {}
    fn take(&self, index: &[usize]) -> Self {
        todo!();
    }
    //pub fn transpose() {}
    //pub fn unbind() {}
    fn permute(&self, dims: &[usize]) -> Self {
        todo!();
    }
    fn unsqueeze(&self, dim: usize) -> Self {
        todo!();
    }
    //pub fn condition() {} // this is pytorch where
    fn conditional_select(&self, x: &Self, y: &Self) -> Self {
        todo!();
    }
    fn repeat(&self, sizes: &[usize]) -> Self {
        todo!();
    }

}
