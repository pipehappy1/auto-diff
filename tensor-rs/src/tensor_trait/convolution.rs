use crate::tensor::PaddingMode;

pub trait Convolution where Self: std::marker::Sized {

    fn conv2d(&self, filter: &Self,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode
    ) -> Self;

    fn conv2d_grad(&self, filter: &Self,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &Self
    ) -> (Self, Self);

    fn conv_gen(&self, filter: &Self,
                    stride: &[usize],
                    padding: &[usize],
                    dilation: &[usize],
                    padding_mode: PaddingMode
    ) -> Self;

    fn conv_grad_gen(&self, filter: &Self,
                         stride: &[usize],
                         padding: &[usize],
                         dilation: &[usize],
                         padding_mode: PaddingMode,
                         output_grad: &Self,
    ) -> (Self, Self);
}
