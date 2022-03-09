use crate::tensor_impl::gen_tensor::GenTensor;
use crate::tensor_trait::index_slicing::IndexSlicing;
use crate::tensor::PaddingMode;
#[cfg(feature = "use-blas-lapack")]
use super::blas_api::BlasAPI;

#[cfg(feature = "use-blas-lapack")]
macro_rules! blas_conv {
    ($a:ty, $b: ident) => {
        pub fn $b(
            data: &GenTensor<$a>,
            filter: &GenTensor<$a>,
            stride: &[usize],
            padding: &[usize],
            dilation: &[usize],
            padding_mode: PaddingMode
        ) -> GenTensor<$a> {
            let self_dim = data.size();
            let filter_dim = filter.size();
        
            let out_channels = filter_dim[0];
            let in_channels = filter_dim[1];
            let sample_size = self_dim[0];
        
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
                
            let conv_size = filter_dim.iter().product::<usize>()/out_channels; // this is Cin xd1xd2xd3...
	    //let mut data_block = vec![0.; conv_size];
	    //let mut filter_block = vec![0.; conv_size];
        
            //println!("sample_size*output_inner_size*conv_size: {:?}", sample_size*output_inner_size*conv_size);
            let mut columned_data = Vec::<$a>::with_capacity(sample_size*output_inner_size*conv_size);
            //let columned_filter = Vec::<f32>::with_capacity(out_channels*conv_size);
        
            let mut left_upper = vec![0; stride.len()];
            let mut current_data_elem = left_upper.to_vec();
            let mut push_value: $a;
            let mut in_margin: bool;
        
            for i in 0..sample_size {
                left_upper.iter_mut().map(|x| *x = 0).count();
        
                for _k in 0..output_inner_size { // every possible data bl
                    // get_data_block
                    //let mut current_data_elem = left_upper.to_vec();
                    current_data_elem.clone_from_slice(&left_upper);
                    for in_channel_index in 0..in_channels {
                        for _inner_index in 0..conv_size/in_channels {
                    
                            // assign single scale to the tmp tensor.
                            push_value = 0.;
                            in_margin = false;
                            for i in 0..current_data_elem.len() {
                                 if current_data_elem[i] < padding[i]
                                    || current_data_elem[i] >= (padding[i] + self_dim[i+2]) {
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            push_value = 0.;
                                            in_margin = true;
                                            break;
                                        },
                                        _ => {unimplemented!();}
                                    }
                                }
                            }
                            if ! in_margin {
                                let real_data_elem = current_data_elem.iter()
                                    .zip(padding.iter())
                                    .map(|(x, y)| x - y)
                                    .collect::<Vec::<usize>>();
                                let mut real_data_elem2 = vec![i, in_channel_index];
                                real_data_elem2.append(&mut real_data_elem.clone());
                                push_value = data.get(&real_data_elem2);
                            }
                    
                            //data_block[in_channel_index*(conv_size/in_channels) + inner_index] = push_value;
                            columned_data.push(push_value);
                    
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
        
            //println!("columned_data: {:?}", columned_data);
            //println!("filter: {:?}", filter.get_data());
            //println!("sample_size*out_channels*output_inner_size: {:?}", sample_size*out_channels*output_inner_size);
            //println!("{:?}, {:?}, {:?}", sample_size*output_inner_size, out_channels, conv_size);
        
            let mut columned_result = vec![0.; sample_size*out_channels*output_inner_size];
            BlasAPI::<$a>::gemm(true, false, sample_size*output_inner_size, out_channels, conv_size,
                                 1., &columned_data, conv_size,
                                 filter.get_data(), conv_size,
                                 1., &mut columned_result, sample_size*output_inner_size
            );
        
            //println!("columned_result: {:?}", columned_result);
        
            let mut result_dim = output_tensor_size.to_vec();
	    result_dim.swap(0, 1);
            let mut result = GenTensor::<$a>::new_move(columned_result.to_vec(),
                                                        result_dim);
            let mut permute_dim: Vec<usize> = (0..output_tensor_size.len()).collect();
            permute_dim[0] = 1;
            permute_dim[1] = 0;
            result = result.permute(&permute_dim);
            result
            
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
blas_conv!(f32, gemm_conv_f32);

#[cfg(feature = "use-blas-lapack")]
blas_conv!(f64, gemm_conv_f64);


#[cfg(test)]
mod tests {
    use crate::tensor_impl::gen_tensor::GenTensor;
    use super::*;


    // gemm_conv
    #[test]
    #[cfg(feature = "use-blas-lapack")]
    fn test_gemm_conv() {
        {
            let data = GenTensor::<f32>::arange(30).reshape(&vec![2, 3, 5]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 3, 3]);
            let stride = vec![1];
            let padding = vec![0];
            let dilation = vec![1];
            let padding_mode = PaddingMode::Zeros;
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
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
            
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
            
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
            
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
            
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
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
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
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
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
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
            println!("final output size: {:?}", result.size());
            println!("final output: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![420.0, 492.0, 564.0, 924.0, 996.0, 1068.0, 1428.0, 1500.0, 1572.0, 1068.0, 1302.0, 1536.0, 2706.0, 2940.0, 3174.0, 4344.0, 4578.0, 4812.0], &vec![1, 2, 3, 3]));
        }
        
        {
            
            let data = GenTensor::<f32>::arange(49).reshape(&vec![1, 1, 7, 7]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 1, 3, 3]);
            let stride = vec![2, 2];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = gemm_conv_f32(&data, &filter, &stride, &padding, &dilation, padding_mode);
            //println!("final output size: {:?}", result.size());
            //println!("final output: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![420.0, 492.0, 564.0, 924.0, 996.0, 1068.0, 1428.0, 1500.0, 1572.0, 1068.0, 1302.0, 1536.0, 2706.0, 2940.0, 3174.0, 4344.0, 4578.0, 4812.0], &vec![1, 2, 3, 3]));
        }

        {
            
            let data = GenTensor::<f64>::arange(49).reshape(&vec![1, 1, 7, 7]);
            let filter = GenTensor::<f64>::arange(18).reshape(&vec![2, 1, 3, 3]);
            let stride = vec![2, 2];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = gemm_conv_f64(&data, &filter, &stride, &padding, &dilation, padding_mode);
            //println!("final output size: {:?}", result.size());
            //println!("final output: {:?}", result.get_data());
            assert_eq!(result, GenTensor::<f64>::new_raw(&vec![420.0, 492.0, 564.0, 924.0, 996.0, 1068.0, 1428.0, 1500.0, 1572.0, 1068.0, 1302.0, 1536.0, 2706.0, 2940.0, 3174.0, 4344.0, 4578.0, 4812.0], &vec![1, 2, 3, 3]));
        }
    }
}
