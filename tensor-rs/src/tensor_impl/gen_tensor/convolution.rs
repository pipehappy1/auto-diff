use std::collections::BTreeMap;
use crate::tensor::PaddingMode;
use super::GenTensor;
use crate::tensor_trait::convolution::Convolution;

impl<T> Convolution for GenTensor<T> where T: num_traits::Float {

    // conv2d ops
    fn conv2d(&self, filter: &GenTensor<T>,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode
    ) -> Self {
        self.conv_gen(filter,
                      &[stride.0, stride.1],
                      &[padding.0, padding.1],
                      &[dilation.0, dilation.1],
                      padding_mode)
    }
    fn conv2d_grad(&self, filter: &GenTensor<T>,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &GenTensor<T>
    ) -> (Self, Self){
            self.conv_grad_gen(filter,
                           &[stride.0, stride.1],
                           &[padding.0, padding.1],
                           &[dilation.0, dilation.1],
                           padding_mode,
                           output_grad)
    }

    // gneral convolutional operator, should work for 2d and 3d cases.
    fn conv_gen(&self, filter: &GenTensor<T>,
                    stride: &[usize],
                    padding: &[usize],
                    dilation: &[usize],
                    padding_mode: PaddingMode
    ) -> GenTensor<T> {
        let self_dim = self.size();
        let filter_dim = filter.size();
        if self_dim.len() != filter_dim.len() {
            panic!("covn2d expects input and filter has the same dims, get {:?}, {:?}", self_dim, filter_dim);
        }
        if stride.len() != padding.len() || stride.len() != dilation.len() || stride.len() != (self_dim.len() - 2) {
            panic!("stride, padding, stride should have the same # of dims, {:?}, {:?}, {:?}", stride, padding, dilation);
        }
        if stride.iter().any(|x| *x < 1) {
            panic!("stride should be at least 1, get {:?}", stride);
        }
        if dilation.iter().any(|x| *x < 1) {
            panic!("dilation should be at least 1, get {:?}", dilation);
        }

        let out_channels = filter_dim[0];
        let in_channels = filter_dim[1];
        let sample_size = self_dim[0];
        let data_channels = self_dim[1];
        if in_channels != data_channels {
            panic!("covn2d expects input data channel size matches depth in filter {:?}, {:?}", self_dim, filter_dim);
        }
        
        // prepare the padded input
        let mut padded_dim = Vec::new();
        for i in 2..self_dim.len() {
            padded_dim.push(self_dim[i] + padding[i-2]*2);
        }
        //println!("padded_dim: {:?}", padded_dim);

        // find the coordinate of
        // start center point in a filter in padded dimension
        // in case filter_dim[i] is even, start_point will be the half.
        // in case filter_dim[i] is odd, start_point will be the center.
        let mut start_point = Vec::new();
        for i in 0..stride.len() {
            let half = filter_dim[2+i]/2;
            let dilated = half*dilation[i];
            start_point.push(dilated);
        }
        //println!("start_point: {:?}", start_point);

        let mut output_size = Vec::new();
        //println!("{:?}, {:?}", padded_dim, stride);
        for i in 0..stride.len() {
            let output_dim = (padded_dim[i] - dilation[i]*(filter_dim[2+i]-1)-1)/stride[i] + 1;
            output_size.push(output_dim);
        }
        let mut output_tensor_size = vec![sample_size, out_channels];
        output_tensor_size.append(&mut output_size.clone()); // output_size moved.
        let output_inner_size = output_size.iter().product::<usize>();
        //println!("output_size: {:?}", output_size);
        //println!("{:?}", output_inner_size);
        //println!("{:?}", output_tensor_size);
        
        let mut ret = GenTensor::<T>::zeros(&output_tensor_size);

        let conv_size = filter_dim.iter().product::<usize>()/out_channels; // this is Cin xd1xd2xd3...
        let mut data_block = vec![T::zero(); conv_size];
        let mut filter_block = vec![T::zero(); conv_size];

        let inner_steps = output_inner_size*out_channels;
        let filter_step = conv_size;
        
        for i in 0..sample_size {
            for j in 0..out_channels {
                filter_block.copy_from_slice(&filter.get_data()[(j)*filter_step..(j+1)*filter_step]);

                let mut left_upper = vec![0; stride.len()];
                for k in 0..output_inner_size {
                    //println!("left_upper: {:?}", left_upper);

                    // get_data_block
                    let mut current_data_elem = left_upper.to_vec();
                    for in_channel_index in 0..in_channels {
                        for inner_index in 0..conv_size/in_channels {

                            // assign single scale to the tmp tensor.
                            let mut push_value = T::zero();
                            let mut in_margin = false;
                            for i in 0..current_data_elem.len() {
                                if current_data_elem[i] < padding[i] || current_data_elem[i] >= (padding[i] + self_dim[i+2]){
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            push_value = T::zero();
                                            in_margin = true;
                                            break;
                                        },
                                        _ => {unimplemented!();}
                                    }
                                }
                            }
                            if ! in_margin {
                                let real_data_elem = current_data_elem.iter().zip(padding.iter()).map(|(x, y)| x - y).collect::<Vec::<usize>>();
                                let mut real_data_elem2 = vec![i, in_channel_index];
                                real_data_elem2.append(&mut real_data_elem.clone());
                                push_value = self.get(&real_data_elem2);
                            }

                            data_block[in_channel_index*(conv_size/in_channels) + inner_index] = push_value;


                            // update to the next position.
                            let mut current_pos = current_data_elem.len()-1;
                            loop {
                                current_data_elem[current_pos] += dilation[current_pos];
                                if current_data_elem[current_pos] >= dilation[current_pos]*filter_dim[current_pos+2] + left_upper[current_pos] {
                                    current_data_elem[current_pos] = left_upper[current_pos];
                                    if current_pos > 0 {
                                        current_pos -= 1;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            };
                        }
                    };
                
                    //let value = data_block.iter().zip(&filter_block).map(|(x, y)|
                    //                                                     (*x)*(*y)
                    //).sum::<T>();
                    let mut value = T::zero();
                    for (x, y) in data_block.iter().zip(&filter_block) {
                        value = value + (*x)*(*y);
                    }
                    //println!("index: {}, {}, {}", i, j, k);
                    //println!("raw index: {}", i*inner_steps + j*output_inner_size + k);
                    //ret.d[i*inner_steps + j*output_inner_size + k] = value;
                    ret.set_1d(i*inner_steps + j*output_inner_size + k, value);

                    // update for next prodsum position
                    let mut current_pos = left_upper.len()-1;
                    loop {
                        left_upper[current_pos] += stride[current_pos];
                        let mut compare_pos = padded_dim[current_pos] - start_point[current_pos]*2;
                        if filter_dim[current_pos+2] % 2 == 0 {
                            compare_pos += 1;
                        }
                        if left_upper[current_pos] >= compare_pos {
                            left_upper[current_pos] = 0;
                            if current_pos > 0 {
                                current_pos -= 1;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    };

                }
            }
        }
        
        ret
    }

    // the 1st return is the gradient for w
    // the 2nd return is the gradient for the input, given the output_grad
    fn conv_grad_gen(&self, filter: &GenTensor<T>,
                         stride: &[usize],
                         padding: &[usize],
                         dilation: &[usize],
                         padding_mode: PaddingMode,
                         output_grad: &GenTensor<T>,
    ) -> (GenTensor<T>, GenTensor<T>) {
        let self_dim = self.size();
        let filter_dim = filter.size();
        let output_grad_dim = output_grad.size();
        if self_dim.len() <= 2 {
            panic!("input data for conv has not enough dim {:?}.", self_dim);
        }
        if filter_dim.len() <= 2 {
            panic!("filter for conv has not enough dim {:?}.", filter_dim);
        }
        if output_grad_dim.len() <= 2 {
            panic!("output gradient for conv has not enough dim {:?}.", filter_dim);
        }
        if self_dim.len() != filter_dim.len() || self_dim.len() != output_grad_dim.len() {
            panic!("covn2d expects input, output gradient and filter has the same dims, get {:?}, {:?}, {:?}", self_dim, filter_dim, output_grad_dim);
        }
        if filter_dim[1] != self_dim[1] {
            panic!("covn2d expects input data channel size matches depth in filter {:?}, {:?}", self_dim, filter_dim);
        }
        if self_dim[0] != output_grad_dim[0] {
            panic!("conv2d expects input and output has the same N: {:?}, {:?}", self_dim, output_grad_dim);
        }
        if filter_dim[0] != output_grad_dim[1] {
            panic!("conv2d expects filter and output has the same Cout: {:?}, {:?}", filter_dim, output_grad_dim);
        }
        if stride.len() != padding.len() || stride.len() != dilation.len() {
            panic!("stride, padding, stride should have the same # of dims, {:?}, {:?}, {:?}", stride, padding, dilation);
        }
        if stride.len()+2 != filter_dim.len() {
            panic!("expect the same inner size, {:?}, {:?}", stride, filter_dim);
        }
        
        let filter_size = filter.size();
        let n_c_out = filter_size[0];
        let n_c_in = filter_size[1];
        let n_n = self_dim[0];
        //let n_d_dd = self_dim.iter().product::<usize>()/n_n/n_c_in;
        let n_f_dd = filter_dim.iter().product::<usize>()/n_c_out/n_c_in;
        let d_inner = self_dim.len() - 2;

        let output_dd = output_grad_dim.iter().product::<usize>()/n_n/n_c_out;

        // save all the record
        let mut w_grad: BTreeMap<usize, Vec<T>> = BTreeMap::new();
        let mut x_grad: BTreeMap<usize, Vec<T>> = BTreeMap::new();

        for i in 0..n_n {
            for j in 0..n_c_out {
                // left_upper in padded dimension.
                let mut left_upper = vec![0; d_inner];

                let mut output_index = 0;
                
                loop {
                    //println!("left_upper: {:?}", left_upper);

                    // get the current output_gradient
                    let output_real_index = j*output_dd + i*n_c_out*output_dd + output_index;
                    //println!("output_real_index: {:?}", output_real_index);
                    let output_dimpos = output_grad.index2dimpos(output_real_index);
                    //println!("output_dimpos: {:?}", output_dimpos);
                    let output_gradient_value = output_grad.get(&output_dimpos);
                    //println!("output_gradient_value: {:?}", output_gradient_value.to_f32());

                    // remember where to get data.
                    // let mut data_loc = BTreeMap::<Vec::<usize>, >::new();

                    for cin_index in 0..n_c_in {
                        for dd_index in 0..n_f_dd {

                            // get current position for filter elements.
                            let mut filter_elem = Vec::new();
                            let mut reminder = dd_index;
                            for dim_pos in 0..d_inner {
                                let left_product = filter_size[dim_pos+3..filter_size.len()]
                                    .iter()
                                    .product::<usize>();
                                filter_elem.push(reminder / left_product);
                                reminder %= left_product;
                            }
                            //println!("filter_elem: {:?}", filter_elem);

                            
                            // get current position for data elements in padded dimension
                            let mut data_elem = left_upper.to_vec();
                            for dim_pos in 0..d_inner {
                                data_elem[dim_pos] += filter_elem[dim_pos]*dilation[dim_pos];
                            }
                            //println!("data_elem: {:?}", data_elem);


                            // find real current position from filter
                            let mut full_filter_elem = vec![j, cin_index];
                            full_filter_elem.append(&mut filter_elem.clone());
                            // println!("filter_value: {}", filter_value.to_f32().expect(""));
                            // println!("full_filter_elem: {:?}", full_filter_elem);

                            // find real current position from data
                            let mut zero_padded_flag = false;
                            let mut unpadded_elem = data_elem.clone();
                            //println!("data_elem: {:?}", data_elem);
                            for dim_pos in 0..d_inner {
                                if data_elem[dim_pos] < padding[dim_pos] {
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            zero_padded_flag = true;
                                        },
                                        PaddingMode::Reflect => {
                                            unpadded_elem[dim_pos] = padding[dim_pos] - data_elem[dim_pos] - 1;
                                        },
                                        PaddingMode::Replicate => {
                                            unpadded_elem[dim_pos] = 0;
                                        },
                                        PaddingMode::Circular => {
                                            unpadded_elem[dim_pos] = self_dim[dim_pos+2] - (padding[dim_pos] - data_elem[dim_pos]);
                                        },
                                    }
                                } else if data_elem[dim_pos] >= self_dim[dim_pos + 2] + padding[dim_pos] {
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            zero_padded_flag = true;
                                        },
                                        PaddingMode::Reflect => {
                                            unpadded_elem[dim_pos] = self_dim[dim_pos+2] - (data_elem[dim_pos] - (self_dim[dim_pos + 2] + padding[dim_pos]) + 1);
                                        },
                                        PaddingMode::Replicate => {
                                            unpadded_elem[dim_pos] = self_dim[dim_pos + 2]-1;
                                        },
                                        PaddingMode::Circular => {
                                            unpadded_elem[dim_pos] = data_elem[dim_pos] - (self_dim[dim_pos + 2] + padding[dim_pos]);
                                        },
                                    }
                                } else {
                                    unpadded_elem[dim_pos] -= padding[dim_pos];
                                }
                            }

                            if zero_padded_flag {
                                continue;
                            } else {
                                //println!("unpadded_elem: {:?}", unpadded_elem);
                                let mut full_data_elem = vec![i, cin_index];
                                full_data_elem.append(&mut unpadded_elem.clone());
                                //println!("full_data_elem: {:?}", full_data_elem);
                                
                                let filter_value = filter.get(&full_filter_elem);
                                let data_value = self.get(&full_data_elem);
                                
                                // collect all the data.
                                let w_grad_value = output_gradient_value*data_value;
                                let x_grad_value = output_gradient_value*filter_value;
                                
                                let total_w_index = filter.dimpos2index(&full_filter_elem);
                                let total_x_index = self.dimpos2index(&full_data_elem);
                                
                                //println!("full_data_elem: {:?}, total_x_index: {:?}, data_value: {:?}",
                                //         full_data_elem,
                                //         total_x_index,
                                //         data_value.to_f32());
                                //println!("full_filter_elem: {:?}, total_w_index: {:?}, filter_value: {:?}, w_grad_value: {:?}, output_gradient_value: {:?}, data_vluae: {:?}",
                                //         full_filter_elem,
                                //         total_w_index,
                                //         filter_value.to_f32(),
                                //         w_grad_value.to_f32(),
                                //         output_gradient_value.to_f32(),
                                //         data_value.to_f32());
                                
                                if let std::collections::btree_map::Entry::Vacant(e) = w_grad.entry(total_w_index) {
                                    e.insert(vec![w_grad_value]);
                                } else {
                                    w_grad.get_mut(&total_w_index).expect("").push(w_grad_value);
                                }
                                
                                if let std::collections::btree_map::Entry::Vacant(e) = x_grad.entry(total_x_index) {
                                     e.insert(vec![x_grad_value]);
                                 } else {
                                     x_grad.get_mut(&total_x_index).expect("").push(x_grad_value);
                                 }    
                            }
                            
                        }
                    }

                    // update left_upper to the next position.
                    for current_pos in 0..d_inner {
                        let real_pos = d_inner - current_pos - 1;
                        left_upper[real_pos] += stride[real_pos];
                        
                        let compare_pos = self_dim[real_pos+2]
                            + padding[real_pos]*2
                            - ((filter_dim[real_pos + 2]-1)*dilation[real_pos] + 1);
                        
                        if left_upper[real_pos] > compare_pos {
                            left_upper[real_pos] = 0;
                        } else {
                            break;
                        }
                    }
                    if left_upper.iter().sum::<usize>() == 0 {
                        break;
                    }
                    output_index += 1;
                };
            }
        }

        let mut ret_w_grad = GenTensor::zeros(filter.size());
        let mut ret_x_grad = GenTensor::zeros(self.size());

        for i in w_grad.keys() {
            //println!("i: {:?}", i);
            let mut sum = T::zero();
            for w_value in w_grad.get(i).expect("") {
                sum = sum + *w_value;
                //println!("w_value: {}", w_value.to_f32().expect("") );
            }
            //ret_w_grad.d[*i] = sum/T::from(w_grad.get(i).expect("").len()).expect("");
            //ret_w_grad.d[*i] = sum;
            ret_w_grad.set_1d(*i, sum);
        }
        for i in x_grad.keys() {
            //println!("i: {:?}", i);
            let mut sum = T::zero();
            for x_value in x_grad.get(i).expect("") {
                sum = sum + *x_value;
                //println!("x_value: {}", x_value.to_f32().expect("") );
            }
            //ret_x_grad.d[*i] = sum/T::from(x_grad.get(i).expect("").len()).expect("");
            //ret_x_grad.d[*i] = sum;
            ret_x_grad.set_1d(*i, sum);
        }
        
        (ret_w_grad, ret_x_grad)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor_impl::gen_tensor::GenTensor;
    use crate::tensor_trait::index_slicing::IndexSlicing;
    use super::*;

    #[test]
    fn conv_gen() {

        {
            let data = GenTensor::<f32>::arange(30).reshape(&vec![2, 3, 5]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 3, 3]);
            let stride = vec![1];
            let padding = vec![0];
            let dilation = vec![1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![312.0, 348.0, 384.0, 798.0, 915.0, 1032.0, 852.0, 888.0, 924.0, 2553.0, 2670.0, 2787.0], &vec![2, 2, 3]));
        }

        {
            let mut raw_data = Vec::new();
            for i in 0..75 {
                raw_data.push(i as f32);
            }
            let data = GenTensor::<f32>::new_raw(&raw_data, &vec![1, 3, 5, 5]);
            let mut raw_data = Vec::new();
            for i in 0..54 {
                raw_data.push(i as f32);
            }
            let filter = GenTensor::<f32>::new_raw(&raw_data, &vec![2, 3, 3, 3]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![15219.0, 15570.0, 15921.0, 16974.0, 17325.0, 17676.0, 18729.0, 19080.0, 19431.0, 37818.0, 38898.0, 39978.0, 43218.0, 44298.0, 45378.0, 48618.0, 49698.0, 50778.0], &vec![1, 2, 3, 3]));    
        }
        
        {
            let mut raw_data = Vec::new();
            for i in 0..60 {
                raw_data.push(i as f32);
            }
            let data = GenTensor::<f32>::new_raw(&raw_data, &vec![1, 3, 5, 4]);
            let mut raw_data = Vec::new();
            for i in 0..36 {
                raw_data.push(i as f32);
            }
            let filter = GenTensor::<f32>::new_raw(&raw_data, &vec![2, 3, 3, 2]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![5289.0, 5442.0, 5595.0, 5901.0, 6054.0, 6207.0, 6513.0, 6666.0, 6819.0, 13227.0, 13704.0, 14181.0, 15135.0, 15612.0, 16089.0, 17043.0, 17520.0, 17997.0], &vec![1, 2, 3, 3]));    
        }

        {
            let data = GenTensor::<f32>::arange(375).reshape(&vec![1, 3, 5, 5, 5]);
            let filter = GenTensor::<f32>::arange(162).reshape(&vec![2, 3, 3, 3, 3]);
            let stride = vec![1, 1, 1];
            let padding = vec![0, 0, 0];
            let dilation = vec![1, 1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![700704.0, 703944.0, 707184.0, 716904.0, 720144.0, 723384.0, 733104.0, 736344.0, 739584.0, 781704.0, 784944.0, 788184.0, 797904.0, 801144.0, 804384.0, 814104.0, 817344.0, 820584.0, 862704.0, 865944.0, 869184.0, 878904.0, 882144.0, 885384.0, 895104.0, 898344.0, 901584.0, 1724220.0, 1734021.0, 1743822.0, 1773225.0, 1783026.0, 1792827.0, 1822230.0, 1832031.0, 1841832.0, 1969245.0, 1979046.0, 1988847.0, 2018250.0, 2028051.0, 2037852.0, 2067255.0, 2077056.0, 2086857.0, 2214270.0, 2224071.0, 2233872.0, 2263275.0, 2273076.0, 2282877.0, 2312280.0, 2322081.0, 2331882.0], &vec![1, 2, 3, 3, 3]));
        }

        {
            let data = GenTensor::<f32>::arange(16).reshape(&vec![1, 1, 4, 4]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 1, 3, 3]);
            let stride = vec![1, 1];
            let padding = vec![1, 1];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("final output size: {:?}", result.size());
            println!("final output: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![73.0, 121.0, 154.0, 103.0, 171.0, 258.0, 294.0, 186.0, 279.0, 402.0, 438.0, 270.0, 139.0, 187.0, 202.0, 113.0, 163.0, 283.0, 370.0, 265.0, 414.0, 663.0, 780.0, 537.0, 738.0, 1131.0, 1248.0, 837.0, 517.0, 781.0, 850.0, 563.0], &vec![1, 2, 4, 4]));
        }

        {
            let data = GenTensor::<f32>::arange(49).reshape(&vec![1, 1, 7, 7]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 1, 3, 3]);
            let stride = vec![2, 2];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("final output size: {:?}", result.size());
            println!("final output: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![420.0, 492.0, 564.0, 924.0, 996.0, 1068.0, 1428.0, 1500.0, 1572.0, 1068.0, 1302.0, 1536.0, 2706.0, 2940.0, 3174.0, 4344.0, 4578.0, 4812.0], &vec![1, 2, 3, 3]));
        }
    }

    #[test]
    fn conv_grad_gen() {

        {
            let data = GenTensor::<f32>::arange(75).reshape(&vec![1, 3, 5, 5]);
            let filter = GenTensor::<f32>::arange(54).reshape(&vec![2, 3, 3, 3]);
            let output_grad = GenTensor::<f32>::arange(18).reshape(&vec![1, 2, 3, 3]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let (w_grad, x_grad) = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            println!("w_grad: {:?}", w_grad);
            println!("x_grad: {:?}", x_grad);
        
            assert_eq!(w_grad, GenTensor::new_raw(&vec![312.0, 348.0, 384.0, 492.0, 528.0, 564.0, 672.0, 708.0, 744.0, 1212.0, 1248.0, 1284.0, 1392.0, 1428.0, 1464.0, 1572.0, 1608.0, 1644.0, 2112.0, 2148.0, 2184.0, 2292.0, 2328.0, 2364.0, 2472.0, 2508.0, 2544.0, 798.0, 915.0, 1032.0, 1383.0, 1500.0, 1617.0, 1968.0, 2085.0, 2202.0, 3723.0, 3840.0, 3957.0, 4308.0, 4425.0, 4542.0, 4893.0, 5010.0, 5127.0, 6648.0, 6765.0, 6882.0, 7233.0, 7350.0, 7467.0, 7818.0, 7935.0, 8052.0], &vec![2, 3, 3, 3]));
        }

        {
        
            let data = GenTensor::<f32>::arange(60).reshape(&vec![1, 3, 5, 4]);
            let filter = GenTensor::<f32>::arange(36).reshape(&vec![2, 3, 3, 2]);
            let output_grad = GenTensor::<f32>::arange(18).reshape(&vec![1, 2, 3, 3]);
            //println!("output_grad: {:?}", output_grad);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let (w_grad, x_grad) = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            println!("{:?}, {:?}, {:?}", w_grad, x_grad, output_grad);
            //println!("w_grad: {:?}", w_grad);
            assert_eq!(w_grad, GenTensor::new_raw(&vec![258.0, 294.0, 402.0, 438.0, 546.0, 582.0, 978.0, 1014.0, 1122.0, 1158.0, 1266.0, 1302.0, 1698.0, 1734.0, 1842.0, 1878.0, 1986.0, 2022.0, 663.0, 780.0, 1131.0, 1248.0, 1599.0, 1716.0, 3003.0, 3120.0, 3471.0, 3588.0, 3939.0, 4056.0, 5343.0, 5460.0, 5811.0, 5928.0, 6279.0, 6396.0], &vec![2, 3, 3, 2]));
        
        }


        {
            let data = GenTensor::<f32>::arange(75).reshape(&vec![1, 3, 5, 5]);
            let filter = GenTensor::<f32>::arange(54).reshape(&vec![2, 3, 3, 3]);
            let output_grad = GenTensor::<f32>::arange(50).reshape(&vec![1, 2, 5, 5]);
            
            let stride = vec![1, 1];
            let padding = vec![1, 1]; // <- THIS IS THE CHANGE
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let (w_grad, x_grad) = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            println!("w_grad: {:?}", w_grad);
            println!("x_grad: {:?}", x_grad);
        
            assert_eq!(w_grad, GenTensor::new_raw(&vec![2680.0, 3420.0, 2760.0, 3900.0, 4900.0, 3900.0, 2760.0, 3420.0, 2680.0, 8680.0, 10670.0, 8360.0, 10150.0, 12400.0, 9650.0, 6760.0, 8170.0, 6280.0, 14680.0, 17920.0, 13960.0, 16400.0, 19900.0, 15400.0, 10760.0, 12920.0, 9880.0, 6280.0, 8170.0, 6760.0, 9650.0, 12400.0, 10150.0, 8360.0, 10670.0, 8680.0, 22280.0, 27920.0, 22360.0, 28400.0, 35525.0, 28400.0, 22360.0, 27920.0, 22280.0, 38280.0, 47670.0, 37960.0, 47150.0, 58650.0, 46650.0, 36360.0, 45170.0, 35880.0], &vec![2, 3, 3, 3]));
        }

        {
            let data = GenTensor::<f32>::arange(75).reshape(&vec![1, 3, 5, 5]);
            let filter = GenTensor::<f32>::arange(150).reshape(&vec![2, 3, 5, 5]);
            let output_grad = GenTensor::<f32>::arange(50).reshape(&vec![1, 2, 5, 5]);
            
            let stride = vec![1, 1];
            let padding = vec![2, 2]; // <- THIS IS THE CHANGE
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let (w_grad, x_grad) = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            println!("w_grad: {:?}", w_grad);
            println!("x_grad: {:?}", x_grad);
        
            assert_eq!(w_grad, GenTensor::new_raw(&vec![1128.0, 1580.0, 2065.0, 1700.0, 1308.0, 1964.0, 2680.0, 3420.0, 2760.0, 2084.0, 2905.0, 3900.0, 4900.0, 3900.0, 2905.0, 2084.0, 2760.0, 3420.0, 2680.0, 1964.0, 1308.0, 1700.0, 2065.0, 1580.0, 1128.0, 5178.0, 6830.0, 8440.0, 6650.0, 4908.0, 6614.0, 8680.0, 10670.0, 8360.0, 6134.0, 7780.0, 10150.0, 12400.0, 9650.0, 7030.0, 5234.0, 6760.0, 8170.0, 6280.0, 4514.0, 3108.0, 3950.0, 4690.0, 3530.0, 2478.0, 9228.0, 12080.0, 14815.0, 11600.0, 8508.0, 11264.0, 14680.0, 17920.0, 13960.0, 10184.0, 12655.0, 16400.0, 19900.0, 15400.0, 11155.0, 8384.0, 10760.0, 12920.0, 9880.0, 7064.0, 4908.0, 6200.0, 7315.0, 5480.0, 3828.0, 2478.0, 3530.0, 4690.0, 3950.0, 3108.0, 4514.0, 6280.0, 8170.0, 6760.0, 5234.0, 7030.0, 9650.0, 12400.0, 10150.0, 7780.0, 6134.0, 8360.0, 10670.0, 8680.0, 6614.0, 4908.0, 6650.0, 8440.0, 6830.0, 5178.0, 12153.0, 16280.0, 20440.0, 16400.0, 12333.0, 16664.0, 22280.0, 27920.0, 22360.0, 16784.0, 21280.0, 28400.0, 35525.0, 28400.0, 21280.0, 16784.0, 22360.0, 27920.0, 22280.0, 16664.0, 12333.0, 16400.0, 20440.0, 16280.0, 12153.0, 21828.0, 29030.0, 36190.0, 28850.0, 21558.0, 28814.0, 38280.0, 47670.0, 37960.0, 28334.0, 35530.0, 47150.0, 58650.0, 46650.0, 34780.0, 27434.0, 36360.0, 45170.0, 35880.0, 26714.0, 19758.0, 26150.0, 32440.0, 25730.0, 19128.0], &vec![2, 3, 5, 5]));
        }

        {
            let data = GenTensor::<f32>::arange(75).reshape(&vec![1, 3, 5, 5]);
            let filter = GenTensor::<f32>::arange(150).reshape(&vec![2, 3, 5, 5]);
            let output_grad = GenTensor::<f32>::arange(18).reshape(&vec![1, 2, 3, 3]);
            
            let stride = vec![2, 2]; // <- THIS IS THE CHANGE
            let padding = vec![2, 2]; 
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let (w_grad, x_grad) = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            println!("w_grad: {:?}", w_grad);
            println!("x_grad: {:?}", x_grad);
        
            assert_eq!(w_grad, GenTensor::new_raw(&vec![176.0, 200.0, 284.0, 172.0, 192.0, 296.0, 320.0, 449.0, 272.0, 292.0, 420.0, 447.0, 624.0, 375.0, 396.0, 164.0, 176.0, 233.0, 128.0, 136.0, 224.0, 236.0, 308.0, 168.0, 176.0, 776.0, 800.0, 1109.0, 672.0, 692.0, 896.0, 920.0, 1274.0, 772.0, 792.0, 1095.0, 1122.0, 1524.0, 900.0, 921.0, 464.0, 476.0, 608.0, 328.0, 336.0, 524.0, 536.0, 683.0, 368.0, 376.0, 1376.0, 1400.0, 1934.0, 1172.0, 1192.0, 1496.0, 1520.0, 2099.0, 1272.0, 1292.0, 1770.0, 1797.0, 2424.0, 1425.0, 1446.0, 764.0, 776.0, 983.0, 528.0, 536.0, 824.0, 836.0, 1058.0, 568.0, 576.0, 392.0, 452.0, 662.0, 424.0, 480.0, 692.0, 752.0, 1097.0, 704.0, 760.0, 1014.0, 1095.0, 1596.0, 1023.0, 1098.0, 560.0, 608.0, 881.0, 560.0, 604.0, 800.0, 848.0, 1226.0, 780.0, 824.0, 1892.0, 1952.0, 2837.0, 1824.0, 1880.0, 2192.0, 2252.0, 3272.0, 2104.0, 2160.0, 3039.0, 3120.0, 4521.0, 2898.0, 2973.0, 1760.0, 1808.0, 2606.0, 1660.0, 1704.0, 2000.0, 2048.0, 2951.0, 1880.0, 1924.0, 3392.0, 3452.0, 5012.0, 3224.0, 3280.0, 3692.0, 3752.0, 5447.0, 3504.0, 3560.0, 5064.0, 5145.0, 7446.0, 4773.0, 4848.0, 2960.0, 3008.0, 4331.0, 2760.0, 2804.0, 3200.0, 3248.0, 4676.0, 2980.0, 3024.0], &vec![2, 3, 5, 5]));
        }
    }
    
}
