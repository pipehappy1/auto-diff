pub mod compare_tensor;
pub mod convolution;
pub mod elemwise;
pub mod index_slicing;
pub mod linalg;
pub mod reduction;


// There are several rust library built around cuda:
// 1. There are nightly rust support and ptx support to write cuda kernel in rust.
// 2. There are cuda library wrapper for cuda ffi
// 3. There are high leverl wrapper for cuda.
//
// https://github.com/rust-cuda is the ffi wrapper
// https://github.com/bheisler/RustaCUDA is a high level wrapper, based on cuda-sys from https://github.com/rust-cuda
// https://github.com/denzp/rust-ptx-builder is used in build.rs to compile rust to ptx
// https://github.com/spearow/juice rely on a generated ffi, which is generated by a unknonw process!!!.

use std::rc::Rc;
use std::cell::RefCell;
#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize, Serializer};

use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpy, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind, check_cuda_status, cudaStream_t};
use cuda11_cutensor_sys::{self, cutensorHandle_t, check_cutensor_status, cutensorInit, cudaDataType_t,cutensorOperator_t_CUTENSOR_OP_IDENTITY, cutensorTensorDescriptor_t, cutensorInitTensorDescriptor, cutensorPermutation, cutensorOperator_t_CUTENSOR_OP_ADD,cutensorElementwiseBinary};
use crate::tensor::gen_tensor::{GenTensor};
use crate::tensor::cuda_helper::*;



//
// The tensor with cuda backend
//
pub struct CudaTensor {
    device_data: *mut f32,
    dim: Vec<usize>,
    stream: Rc<StreamCell>,
    //cutensor: Option<Rc<RefCell<CudaCutensor>>>,
}

impl CudaTensor {
    pub fn new() -> CudaTensor {
        CudaTensor {
            device_data: std::ptr::null_mut(),
            dim: Vec::new(),
            stream: Rc::new(StreamCell::new()),
            //cutensor: None,
        }
    }
    pub fn new_raw(data: &[f32], shape: &[usize]) -> CudaTensor {
        
        let mut device_data: *mut f32 = std::ptr::null_mut();
        let elems: usize = shape.iter().product();
        if elems != data.len() {
            panic!();
        }

        unsafe {
            //println!("cudaMalloc");
            check_cuda_status(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                         std::mem::size_of::<f32>()*elems));
            //println!("cudaMemcpy");
            cudaMemcpy(device_data as *mut _,
                       data.as_ptr() as *mut _,
                       std::mem::size_of::<f32>()*elems,
                       cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
            
        CudaTensor {
            device_data: device_data,
            dim: shape.to_vec(),
            stream: Rc::new(StreamCell::new()),
            //cutensor: None,
        }
    }

    pub fn new_move(data: Vec::<f32>, shape: Vec::<usize>) -> CudaTensor {
        let mut device_data: *mut f32 = std::ptr::null_mut();
        let elems: usize = shape.iter().product();
        if elems != data.len() {
            panic!();
        }

        unsafe {
            //println!("cudaMalloc");
            check_cuda_status(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                         std::mem::size_of::<f32>()*elems));
            //println!("cudaMemcpy");
            cudaMemcpy(device_data as *mut _,
                       data.as_ptr() as *mut _,
                       std::mem::size_of::<f32>()*elems,
                       cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
            
        CudaTensor {
            device_data: device_data,
            dim: shape.to_vec(),
            stream: Rc::new(StreamCell::new()),
            //cutensor: None,
        }
    }

    pub fn _get_stream(&self) -> cudaStream_t {
         self.stream.get_stream().raw_stream()
    }
    /// NEVER call _flush when _get_stream scope is still valid!
    /// Always call _flush when final result is required.
    pub fn _flush(&self) {
        unsafe {
            cudaStreamSynchronize(self._get_stream() as _);
        }
    }

    pub fn _get_cutensor(&self) -> Option<Rc<CudaCutensor>>{
        todo!();
    }

    /// copy data from GPU memory to mm.
    pub fn to_GenTensor(&self) -> GenTensor<f32> {
        let mut data = vec![0.; self.numel()];
        // finish all jobs on stream before output to main memory.
        self._flush(); // This makes sure when copy to main memory, the data is there.
        // However, this is not optimal case, as the stream is shared between
        // multiple cuda tensor. Un related ops may prevent the copying.
        // The ideal case would be set up callback event when the tensor op is issued
        // and wait only on that event.
        
        unsafe {
            //println!("cudaMemcpy");
            cudaMemcpy(data.as_mut_ptr() as *mut _,
                       self.device_data as *mut _,
                       std::mem::size_of::<f32>()*self.numel(),
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }

        GenTensor::<f32>::new_move(data, self.dim.clone())
    }

    /// copy data from mm to GPU
    pub fn from_GenTensor(data: &GenTensor<f32>) -> CudaTensor {
        CudaTensor::new_raw(data.get_data(), data.size())
    }

    /// Convert 1 dim index to multi-dim index.
    pub fn index2dimpos(&self, index: usize) -> Vec::<usize> {
        if index >= self.numel() {
            panic!("index out of range, {:?}, {:?}", index, self.numel());
        }
        let mut ret = Vec::new();
        let mut reminder = index;
        for i in &self.stride() {
            //println!("{}", reminder);
            ret.push(reminder/i);
            reminder %= i;
        }
        ret
    }

    /// Convert multi-dim index to 1 dim index.
    pub fn dimpos2index(&self, dimpos: &[usize]) -> usize {
        if dimpos.len() != self.dim.len() {
            panic!("get expects the same dim self.dim: {:?}, o: {:?}", self.dim, dimpos);
        }
        for (i, j) in self.dim.iter().zip(dimpos.iter()) {
            if j >= i {
                panic!("get expects the dim within range self.dim: {:?}, o: {:?}", self.dim, dimpos);
            }
        }
        let mut ret = 0;
        for (st, i) in self.stride().iter().zip(dimpos.iter()) {
            //println!("{}", reminder);
            ret += st*i;
        }
        ret
    }

    // 
    // as_tensor
    // as_strided
    // from_ndarray
    // zeros
    pub fn zeros(size: &[usize]) -> CudaTensor {
        let cap = size.iter().product();
        CudaTensor::new_raw(&vec![0.; cap], size)
    }
    // zeros_like
    pub fn zeros_like(&self) -> CudaTensor {
        let cap = self.dim.iter().product();
        CudaTensor::new_raw(&vec![0.; cap], &self.dim)
    }

    // ones
    pub fn ones(size: &[usize]) -> CudaTensor {
        let cap = size.iter().product();
        CudaTensor::new_raw(&vec![1.; cap], size)
    }
    // ones_like
    pub fn ones_like(&self) -> CudaTensor {
        let cap = self.dim.iter().product();
        CudaTensor::new_raw(&vec![1.; cap], &self.dim)
    }
    // arange
    pub fn arange(end: usize) -> CudaTensor {
        let mut d: Vec<f32> = vec![0.; end];
        for i in 0..end {
            d[i] = i as f32;
        }
        CudaTensor::new_raw(&d, &vec![1])
    }
    // range
    // linspace
    // logspace
    // eye
    pub fn empty(shape: &[usize]) -> CudaTensor {
        let mut device_data: *mut f32 = std::ptr::null_mut();
        let elems: usize = shape.iter().product();

        unsafe {
            //println!("cudaMalloc");
            check_cuda_status(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                         std::mem::size_of::<f32>()*elems));
        }
            
        let mut ret = CudaTensor {
            device_data: device_data,
            dim: shape.to_vec(),
            stream: Rc::new(StreamCell::new()),
            //cutensor: None,
        };
        ret
    }
    // empty_like
    pub fn empty_like(&self) -> CudaTensor {
        let mut device_data: *mut f32 = std::ptr::null_mut();
        let elems: usize = self.dim.iter().product();

        unsafe {
            //println!("cudaMalloc");
            check_cuda_status(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                         std::mem::size_of::<f32>()*elems));
        }
            
        let mut ret = CudaTensor {
            device_data: device_data,
            dim: self.dim.to_vec(),
            stream: self.stream.clone(), // share the cuda stream!!!
            //cutensor: None,
        };
        ret
    }
    // empty_stided
    // full
    // full_like
    // quantize_per_tensor
    // quantize_per_channel
    // 

    /// Create a tensor filled with the same value d
    ///
    /// ```
    /// # use tensor_rs::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
    /// ```
    pub fn fill(d: f32, shape: &[usize]) -> CudaTensor {
        let elems: usize = shape.iter().product();
        let d: Vec<f32> =  vec![d; elems];

        CudaTensor::new_raw(&d, shape)
    }
    /// assign a row.
    pub fn from_record(&mut self, row: usize, record: &[f32]) -> Result<(), ()> {
        if record.len() != self.dim[self.dim.len() - 1] {
            Err(())
        } else {
            unsafe {
            //println!("cudaMemcpy");
                cudaMemcpy(((self.device_data as usize)
                            + row*self.dim[self.dim.len()-1]*std::mem::size_of::<f32>()) as _,
                           record.as_ptr() as *mut _,
                           std::mem::size_of::<f32>()*record.len(),
                           cudaMemcpyKind::cudaMemcpyHostToDevice);
            }
            Ok(())
        }
    }

    /// Right dimension changes fastest.
    /// Right dimension has the stride 1.
    ///
    /// ```
    /// # use tensor_rs::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![0.; 3*5*2], &vec![3,5,2]);
    /// assert_eq!(m1.stride(), vec![10,2,1]);
    /// ```
    pub fn stride(&self) -> Vec<usize> {
        let mut ret = vec![0; self.dim.len()];
        let dsize = ret.len();
        for i in 0..dsize {
            if i == 0 {
                ret[dsize-1] = 1;
            } else {
                ret[dsize-i-1] = ret[dsize-i]*self.dim[dsize-i];
            }
        }
        ret
    }
    
    /// Return value at the index of the tensor.
    ///
    /// ```
    /// # use tensor_rs::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![2,3]);
    /// assert_eq!(m1.get(&vec![1,1]), 5.);
    /// ```
    pub fn get(&self, o: &[usize]) -> f32 {
        let index = self.dimpos2index(o);
        //println!("index: {:?}", index);

        let mut data: Vec<f32> = vec![0.0];
        unsafe {
            //println!("cudaMemcpy");
            cudaMemcpy(data.as_mut_ptr() as *mut _,
                       ((self.device_data as usize)
                        + std::mem::size_of::<f32>()*index) as *mut _,
                       std::mem::size_of::<f32>(),
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }
        data[0]
    }
    pub fn set(&mut self, o: &[usize], v: f32) {
        let index = self.dimpos2index(o);
        //println!("index: {:?}", index);

        let mut data: Vec<f32> = vec![v];
        unsafe {
            //println!("cudaMemcpy");
            cudaMemcpy(((self.device_data as usize)
                        + std::mem::size_of::<f32>()*index) as *mut _,
                       data.as_mut_ptr() as *mut _,
                       std::mem::size_of::<f32>(),
                       cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
    }
    pub fn set_1d(&mut self, o: usize, v: f32) {

        let mut data: Vec<f32> = vec![v];
        unsafe {
            //println!("cudaMemcpy");
            cudaMemcpy(((self.device_data as usize)
                        + std::mem::size_of::<f32>()*o) as *mut _,
                       data.as_mut_ptr() as *mut _,
                       std::mem::size_of::<f32>(),
                       cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
    }
    pub fn get_mut(&mut self, o: &[usize]) -> &mut f32 {
        unimplemented!("This deprecated, use set()");
    }

    /// dump the underlying vec
    pub fn get_raw(&self) -> Vec<f32> {
        let mut data: Vec<f32> = vec![0.0; self.numel()];
        unsafe {
            //println!("cudaMemcpy");
            cudaMemcpy(data.as_mut_ptr() as *mut _,
                       self.device_data as *mut _,
                       std::mem::size_of::<f32>()*self.numel(),
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }
        data
    }
    pub fn get_u8(&self) -> Option<Vec<u8>> {
        self.to_GenTensor().get_u8()
    }
    
    /// dump the single value in the tensor
    /// if it is the single value in the tensor.
    pub fn get_scale(&self) -> f32 {
        if self.dim.len() <= 1 && self.dim[0] == 1 {
            return self.to_GenTensor().get_scale();
        } else {
            panic!("Only one element tensor can get_scale()");
        }
    }

    // get NCHW elements
    /// get NCHW elements, always return the size of left most dimension.
    pub fn get_n(&self) -> CudaTensor {
        CudaTensor::new_raw(&vec![self.dim[0] as f32], &vec![1])
    }
    /// get NCHW elements, always return the size of second left most dimension.
    pub fn get_c(&self) -> CudaTensor {
        CudaTensor::new_raw(&vec![self.dim[1] as f32], &vec![1])
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions.
    pub fn get_d(&self) -> CudaTensor {
        if self.dim.len() == 5 {
            CudaTensor::new_raw(&vec![self.dim[2] as f32], &vec![1])
        } else {
            panic!("Bad shape for get_D");
        }

    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions or 4 dimensions.
    pub fn get_h(&self) -> CudaTensor {
        if self.dim.len() == 5 {
            CudaTensor::new_raw(&vec![self.dim[3] as f32], &vec![1])
        } else if self.dim.len() == 4 {
            CudaTensor::new_raw(&vec![self.dim[2] as f32], &vec![1])
        } else {
            panic!("Bad shape for get_D");
        }
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions or 4 dimensions.
    pub fn get_w(&self) -> CudaTensor {
        if self.dim.len() == 5 {
            CudaTensor::new_raw(&vec![self.dim[4] as f32], &vec![1])
        } else if self.dim.len() == 4 {
            CudaTensor::new_raw(&vec![self.dim[3] as f32], &vec![1])
        } else {
            panic!("Bad shape for get_D");
        }
    }

    /// Returns the size of the self tensor.
    pub fn size(&self) -> &Vec<usize> {
        &self.dim
    }
    pub fn get_data(&self) -> &Vec<f32> {
        unimplemented!("tensor on device cannot get mut reference");
    }
    pub fn get_data_mut(&mut self) -> &mut Vec<f32> {
        unimplemented!("tensor on device cannot get mut reference");
    }
    pub fn _get_device_data(&self) -> *mut f32 {
        self.device_data
    }

    /// Returns the total number of elements of the tensor
    /// Return usize
    pub fn numel(&self) -> usize {
        self.dim.iter().product()
    }

    /// Returns the total number of elements in the input tensor
    pub fn numel_tensor(&self) -> CudaTensor {
        CudaTensor::new_move(vec![self.dim.iter().map(|x| *x as f32).product()], vec![1])
    }

    pub fn get_patch(&self, range: &[(usize, usize)], step: Option<&[usize]>) -> CudaTensor {
        todo!();
    }

    pub fn set_patch(&mut self, val: &CudaTensor, range: &[(usize, usize)], step: Option<&[usize]>) {
        todo!();
    }

    /// element-wise add with right-hand broadcast.
    ///
    ///
    pub fn add(&self, o: &CudaTensor) -> CudaTensor {
        let mut ret = o.clone();
        
        unsafe {
            let mut stream: cudaStream_t = self._get_stream();
        
            let mut handle:cutensorHandle_t = std::mem::uninitialized();
            check_cutensor_status(cutensorInit(&mut handle as *mut _));
            
            let alpha: f32 = 1.0;
            let gamma: f32 = 1.0;
            
            let extent: Vec<i64> = self.size().iter().map(|x| *x as i64).collect();
            
            let mut descA: cutensorTensorDescriptor_t = std::mem::uninitialized();
            let mut descC: cutensorTensorDescriptor_t = std::mem::uninitialized();
            
            check_cutensor_status(cutensorInitTensorDescriptor( &mut handle,
                                                                 &mut descA,
                                                                 self.size().len() as _,
                                                                 extent.as_ptr(),
                                                                 std::ptr::null(),/*stride*/
                                                                 cudaDataType_t::CUDA_R_32F,
                                                                 cutensorOperator_t_CUTENSOR_OP_IDENTITY));
            check_cutensor_status(cutensorInitTensorDescriptor( &mut handle,
                                                                 &mut descC,
                                                                 self.size().len() as _,
                                                                 extent.as_ptr(),
                                                                 std::ptr::null(),/*stride*/
                                                                 cudaDataType_t::CUDA_R_32F,
                                                                 cutensorOperator_t_CUTENSOR_OP_IDENTITY));
    
            let mut modeA: Vec<i32> = vec![32; self.size().len()];
            let mut modeC: Vec<i32> = vec![32; self.size().len()];

            for i in 0..self.size().len() {
                modeA[i] = modeA[i] + i as i32;
                modeC[i] = modeC[i] + i as i32;
            }
            
            check_cutensor_status(cutensorElementwiseBinary(&handle,
                                                            &alpha as *const _ as _,
                                                            self._get_device_data() as _,
                                                            &descA as _,
                                                            modeA.as_ptr(),
                                                            &gamma as *const _ as _,
                                                            ret._get_device_data() as _,
                                                            &descC as _,
                                                            modeC.as_ptr(),
                                                            ret._get_device_data() as _,
                                                            &descC as _,
                                                            modeC.as_ptr(),
                                                            cutensorOperator_t_CUTENSOR_OP_ADD,
                                                            cudaDataType_t::CUDA_R_32F,
                                                            stream as _
            ));

            // this is called in CudaTensor::_flush() !!!
            //cudaStreamSynchronize(stream as _);
        }
        ret
        //unimplemented!();
    }
    pub fn sub(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn mul(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn div(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn mm(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn matmul(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn outer(&self, o: &CudaTensor, avg: Option<bool>) -> CudaTensor {
        unimplemented!();
    }
    pub fn squared_error(t1: &Self, t2: &Self) -> CudaTensor {
        unimplemented!();
    }

    // Comparison Ops
    pub fn all_close(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn arg_sort(&self, dim: usize, descending: bool) -> CudaTensor {
        unimplemented!();
    }
    pub fn eq_t(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn equal(&self, o: &CudaTensor) -> bool {
        unimplemented!();
    }
    pub fn ge(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn gt(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn le(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn lt(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    pub fn ne(&self, o: &CudaTensor) -> CudaTensor {
        unimplemented!();
    }
    
}

impl Drop for CudaTensor {
    fn drop(&mut self) {
        if self.device_data != std::ptr::null_mut() {
            unsafe {
                //println!("cudaFree");
                check_cuda_status(cudaFree(self.device_data as _));                    
            }
        }
    }
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}\n", self.to_GenTensor())
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        let mut device_data: *mut f32 = std::ptr::null_mut();
        
        unsafe {
            //println!("cudaMalloc");
            check_cuda_status(cudaMalloc(&mut device_data as *mut _ as *mut _,
                                         std::mem::size_of::<f32>()*self.numel()));
            //println!("cudaMemcpy");
            cudaMemcpy(device_data as _,
                       self.device_data as _,
                       std::mem::size_of::<f32>()*self.numel(),
                       cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
        
        CudaTensor {
            device_data: device_data,
            dim: self.dim.clone(),
            stream: self.stream.clone()
        }
    }
}

#[cfg(feature = "use-serde")]
impl Serialize for CudaTensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let t = self.to_GenTensor();
        let mut state = serializer.serialize_struct("CudaTensor", 3)?;
        //state.serialize
        state.end()
    }
}


#[cfg(all(test, feature = "use-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_stream() {
        let mut stream = CudaStream::new();
    }

    #[test]
    fn cuda_memcpy() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        //println!("{:?}", input);
    }

    #[test]
    fn cuda_to_GenTensor() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        let local = input.to_GenTensor();
        //println!("{:?}", local);
        assert_eq!(local.numel(), 9);
        assert_eq!(local.get_data().clone(), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }

    #[test]
    fn cuda_from_record() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        input.from_record(1, &vec![11., 12., 13.]);
        //println!("{:?}", input.to_GenTensor());
        assert_eq!(input.to_GenTensor().get_data().clone(), vec![1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn cuda_get() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        //println!("{:?}", input.get(&vec![0,0,1,1]));
        assert_eq!(input.get(&vec![0,0,1,1]), 5.);
    }

    #[test]
    fn cuda_set() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        //println!("{:?}", input.get(&vec![0,0,1,1]));
        input.set(&vec![0,0,1,1], 15.);
        assert_eq!(input.get(&vec![0,0,1,1]), 15.);
    }

    #[test]
    fn cuda_set_1d() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        //println!("{:?}", input.get(&vec![0,0,1,1]));
        input.set_1d(4, 15.);
        assert_eq!(input.get(&vec![0,0,1,1]), 15.);
    }

    #[test]
    fn cuda_numel() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        //println!("{:?}", input.numel());
        assert_eq!(input.numel(), 9);
    }

    // binary ops
    #[test]
    fn cuda_add() {
        let m1 = CudaTensor::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
        let m2 = CudaTensor::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
        let m3 = m1.add(&m2);
        println!("{:?}", m3);
    }

    #[test]
    fn cuda_clone() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                            &vec![1, 1, 3, 3]);
        let input2 = input.clone();
        //println!("{:?}", input2.to_GenTensor());
        assert_eq!(input2.to_GenTensor(), input.to_GenTensor());
    }
}
