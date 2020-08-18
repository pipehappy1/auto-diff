// wrap cuda stream and other session status.
//
//
use std::rc::Rc;
use std::cell::RefCell;
use cuda11_cudart_sys::{self, cudaStreamCreate, cudaStreamSynchronize, cudaStreamDestroy, check_cuda_status, cudaStream_t};
use cuda11_cutensor_sys::{self, cutensorHandle_t, check_cutensor_status, cutensorInit};

///
/// Raw Cuda stream
///
pub struct CudaStream {
    stream: cudaStream_t,
}

impl CudaStream {
    pub fn new() -> CudaStream {
        let mut stream = std::ptr::null_mut();
        unsafe {
            check_cuda_status(cudaStreamCreate(&mut stream as *mut _ as _));            
        }
        CudaStream {
            stream: stream,
        }
    }
    pub fn empty() -> CudaStream {
        CudaStream {
            stream: std::ptr::null_mut(),
        }
    }
    pub fn raw_stream(&self) -> cudaStream_t {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.stream != std::ptr::null_mut() {
            unsafe {
                check_cuda_status(cudaStreamDestroy(self.stream as _));
                //println!("cudaFree");
            }
        }
    }
}

/// Lazy initialize cuda stream.
/// Only initialize it when get_stream method is called.
///
pub struct StreamCell {
    stream: RefCell<Option<CudaStream>>,
}

impl StreamCell {
    pub fn new() -> StreamCell {
        StreamCell {
            stream: RefCell::new(None),
        }
    }
    pub fn get_stream(&self) -> StreamCellGuard {
        let stream = match self.stream.borrow_mut().take() {
            None => {CudaStream::new()},
            Some(strm) => {strm}
        };
        StreamCellGuard {
            stream_cell: self,
            stream: stream,
        }
    }
}

/// move value out and back in.
pub struct StreamCellGuard<'a> {
    stream_cell: &'a StreamCell,
    stream: CudaStream,
}
impl<'a> Drop for StreamCellGuard<'a> {
    fn drop(&mut self) {
        let stream = std::mem::replace(&mut self.stream, CudaStream::empty());
        *self.stream_cell.stream.borrow_mut() = Some(stream);
    }
}
impl<'a> std::ops::Deref for StreamCellGuard<'a> {
    type Target = CudaStream;

    fn deref(&self) -> &CudaStream {
        // This increases the ergnomics of a `DynamicImageGuard`. Because
        // of this impl, most uses of `DynamicImageGuard` can be as if
        // it were just a `&DynamicImage`.
        &self.stream
    }
}

#[cfg(all(test, feature = "use-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_stream() {
        let mut stream = CudaStream::new();
        let raw_stream = stream.raw_stream();
        assert!((raw_stream as *const _) != std::ptr::null());
    }

    #[test]
    fn cuda_stream_cell() {
        {
            let stream = StreamCell::new();

            let mut str1: cudaStream_t = std::ptr::null_mut();
            let mut str2: cudaStream_t = std::ptr::null_mut();
            str1 = stream.get_stream().raw_stream();
            str2 = stream.get_stream().raw_stream();
            
            //println!("{:?}, {:?}", str1, str2);
            assert_eq!(str1, str2);
        }

        {
            let s1 = Rc::new(StreamCell::new());
            let s2 = s1.clone();
            let str1 = s1.get_stream().raw_stream();
            let str2 = s2.get_stream().raw_stream();
            //println!("{:?}, {:?}", str1, str2);
        }
    }

    
}

//
// Cuda cutensor
//
pub struct CudaCutensor {
    handle: cutensorHandle_t,
}

impl CudaCutensor {
    pub fn new() -> CudaCutensor {
        unsafe {
            let mut handle:cutensorHandle_t = std::mem::uninitialized();
            check_cutensor_status(cutensorInit(&mut handle as *mut _));

            CudaCutensor {
                handle: handle,
            }
        }
    }
}

impl Drop for CudaCutensor {
    fn drop(&mut self) {
        unsafe {
            
        }
    }
}
