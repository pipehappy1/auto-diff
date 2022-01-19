#![allow(clippy::comparison_chain)]
use std::cmp;
use super::gen_tensor::GenTensor;
use super::reduction::ReduceTensor;
use super::elemwise::ElemwiseTensorOp;
use super::index_slicing::IndexSlicing;

pub trait LinearAlgbra {
    type TensorType;
    type ElementType;

    fn norm(&self) -> Self::TensorType;
    fn normalize_unit(&self) -> Self::TensorType;
    fn lu(&self) -> Option<[Self::TensorType; 2]>;
    fn lu_solve(&self, y: &Self::TensorType) -> Option<Self::TensorType>;
    fn qr(&self) -> Option<[Self::TensorType; 2]>;
    fn eigen(&self) -> Option<[Self::TensorType; 2]>;
    fn cholesky(&self) -> Option<Self::TensorType>;
    fn det(&self) -> Option<Self::ElementType>;
    fn svd(&self) -> Option<[Self::TensorType; 3]>;
    fn inv(&self) -> Option<Self::TensorType>;
    fn pinv(&self) -> Self::TensorType;
}


impl<T> LinearAlgbra for GenTensor<T>
where T: num_traits::Float {
    type TensorType = GenTensor<T>;
    type ElementType = T;

    fn norm(&self) -> Self::TensorType {
        // TODO: support 'fro', 'nuc', 'inf', '-inf'...
        self.mul(self).sum(None, false).sqrt()
    }

    fn normalize_unit(&self) -> Self::TensorType {
        let s = self.mul(self).sum(None, false);
        self.div(&s.sqrt())
    }

    fn lu(&self) -> Option<[Self::TensorType; 2]> {
        // lu is for square matrix only.
        // TODO; handle the batched/3d case.
        if self.size().len() != 2 {
            return None;
        }
        if self.size()[0] != self.size()[1] {
            return None;
        }
        let nr = self.size()[0];
        let mut l = GenTensor::<T>::eye(nr, nr);
        let mut u = self.clone();
        for i in 0..nr-1 {
            let leading = u.get(&[i, i]);
            for j in i+1..nr {
                let multiplier = u.get(&[j, i])/leading;
                l.set(&[j, i], multiplier);
                for k in i..nr {
                    u.set(&[j, k], u.get(&[j, k]) - u.get(&[i, k])*multiplier);
                }
            }
        }

        Some([l, u])
    }

    fn lu_solve(&self, b: &Self::TensorType) -> Option<Self::TensorType> {
        if self.size().len() != 2 {
            return None;
        }
        if self.size()[0] != self.size()[1] {
            return None;
        }
        let n = self.size()[0];
        if b.size().len() != 2 || b.size()[0] != n || b.size()[1] != 1 {
            return None;
        }
        
        match self.lu() {
            Some([l, u]) => {
         
                let mut y = GenTensor::<T>::zeros(&[n, 1]);
                for i in 0..n {
                    y.set(&[i, 0],
                          (b.get(&[i, 0]) - y.dot(&l.get_row(i))) / l.get(&[i, i]));
                }
                let mut x = GenTensor::<T>::zeros(&[n, 1]);
                for i in 0..n {
                    x.set(&[n-i-1, 0],
                          (y.get(&[n-i-1, 0]) - x.dot(&u.get_row(n-i-1))) / u.get(&[n-i-1, n-i-1]));
                }
                
                Some(x)
            },
            None => {None}
        }
    }

    fn qr(&self) -> Option<[Self::TensorType; 2]> {
        // qr is for square matrix only.
        // TODO; handle the batched/3d case.
        if self.size().len() != 2 {
            return None;
        }
        //if self.size()[0] != self.size()[1] {
        //    return None;
        //}
        let m = self.size()[self.size().len()-2];
        let n = self.size()[self.size().len()-1];

        let mut q = GenTensor::<T>::zeros(&[m, cmp::min(m, n)]);
        let mut r = GenTensor::<T>::zeros(&[n, n]);
        for i in 0..n {
            let a = self.get_column(i);
            let mut u = a.clone();
            for j in 0..i {
                u = u.sub(&a.proj(&q.get_column(j)));
            }
            if i < cmp::min(m, n) {
                let e = u.normalize_unit();
                q.set_column(&e, i);
            }
            for j in 0..cmp::min(i+1, cmp::min(m, n)) {
                if j <= m {
                    r.set(&[j, i], a.dot(&q.get_column(j)));
                }
            }
        }
        
        Some([q, r])
    }

    fn eigen(&self) -> Option<[Self::TensorType; 2]> {
        // TODO; handle the batched/3d case.
        if self.size().len() != 2 {
            return None;
        }
        if self.size()[0] != self.size()[1] {
            return None;
        }
        let n = self.size()[0];
        let mut cap_a = self.clone();

        let tolerance: f64 = 1e-9;
        let iter_max = 100;

        let mut evec = GenTensor::<T>::zeros(&[n, n]);
        let mut eval = GenTensor::<T>::zeros(&[n, 1]);
        for i in 0..n {
            let mut x = GenTensor::<T>::fill(T::one(), &[n, 1]);
            let mut iter_counter = 0;
            loop {
                if iter_counter > iter_max {
                    break;
                }
                let x1 = x.clone();
                x = cap_a.matmul(&x).normalize_unit();
                if x1.sub(&x).norm().get_scale() < T::from(tolerance).unwrap() {
                    break;
                }
                iter_counter += 1;
            }
            //println!("iter: {:?}", iter_counter);
            let lambda = x.permute(&[1, 0]).matmul(self).matmul(&x).squeeze(None);

            evec.set_column(&x, i);
            eval.set(&[i, 0], lambda.get_scale());

            cap_a = cap_a.sub(&GenTensor::<T>::eye(n, n).mul(&lambda));

            //println!("index: {:?}", i);
        }

        Some([evec, eval])
    }
    fn cholesky(&self) -> Option<Self::TensorType> {
        // TODO; handle the batched/3d case.
        if self.size().len() != 2 {
            return None;
        }
        if self.size()[0] != self.size()[1] {
            return None;
        }
        let n = self.size()[0];

        let mut ret = GenTensor::<T>::zeros(&[n, n]);
        for i in 0..n {
            for j in 0..i {
                ret.set(&[j, i],
                        (self.get(&[j, i]) -
                         ret.get_column(j).dot(&ret.get_column(i)))/ret.get(&[j, j]))
            }
            ret.set(&[i, i],
                    T::sqrt(self.get(&[i,i]) - ret.get_column(i).dot(&ret.get_column(i))));
        }
        Some(ret)
    }
    
    fn det(&self) -> Option<Self::ElementType> {
        if let Some(v) = self.lu() {
            let [_l, u] = v;
            let ret = u.get_diag().prod(None, false).get(&[0]);
            Some(ret)
        } else {
            None
        }
    }

    fn svd(&self) -> Option<[Self::TensorType; 3]> {
        // TODO; handle the batched/3d case.
        // TODO: assume the input is thin matrix.
        let m = self.size()[self.size().len()-2];
        let n = self.size()[self.size().len()-1];

        let cap_a: GenTensor<T>;
        if m > n {
            cap_a = self.permute(&[1, 0]).matmul(self);
        } else if m < n {
            cap_a = self.matmul(&self.permute(&[1, 0]));
        } else {
            cap_a = self.clone();
        }

        let tolerance: f64 = 1e-9;
        let iter_max = 100;

        let mut s: GenTensor<T>;
        let mut v = GenTensor::<T>::eye(n, n);
        let mut iter_counter = 0;
        loop {

            let v1 = v.clone();
            let [qv, r] = cap_a.matmul(&v).qr().unwrap();
            v = qv;
            
            if v1.sub(&v).norm().get_scale() < T::from(tolerance).unwrap() {
                s = r;
                break;
            }

            if iter_counter > iter_max {
                s = r;
                break;
            }

            iter_counter += 1;
            //println!("iter_counter {:?}", iter_counter);
        }

        let u: GenTensor<T>;
        if m > n {
            s = s.sqrt();
            v = v.permute(&[1, 0]);
            let invs = GenTensor::<T>::ones(&[n]).div(&s.get_diag());
            u = self.matmul(&v.permute(&[1, 0])).matmul(&invs);
        } else if m < n {
            s = s.sqrt();
            u = v.permute(&[1, 0]);
            let invs = GenTensor::<T>::ones(&[n]).div(&s.get_diag());
            v = invs.matmul(&u.permute(&[1, 0])).matmul(self);
        } else {
            u = v.permute(&[1, 0]);
        }
        
        Some([u, s, v])
    }

    // may ref to
    // https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
    fn inv(&self) -> Option<Self::TensorType> {unimplemented!();}
    fn pinv(&self) -> Self::TensorType {unimplemented!();}
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit() {
        let m = GenTensor::<f64>::new_raw(&[1., 1., 0., 1., 0., 1., 0., 1., 1.], &[3,3]);
        let nm = m.normalize_unit();
        assert_eq!(nm, GenTensor::<f64>::new_raw(&[0.4082482904638631, 0.4082482904638631, 0.,
                                                   0.4082482904638631, 0., 0.4082482904638631,
                                                   0., 0.4082482904638631, 0.4082482904638631, ],
                                                 &[3,3]));
    }

    #[test]
    fn lu() {
        let m = GenTensor::<f64>::new_raw(&[1., 1., 1., 4., 3., -1., 3., 5., 3.], &[3,3]);
        let [l, u] = m.lu().unwrap();
        let el = GenTensor::<f64>::new_raw(&[1., 0., 0., 4., 1., 0., 3., -2., 1.], &[3,3]);
        let eu = GenTensor::<f64>::new_raw(&[1., 1., 1., 0., -1., -5., 0., 0., -10.], &[3,3]);
        assert_eq!(l, el);
        assert_eq!(u, eu);
    }

    #[test]
    fn lu_solve() {
        let cap_a = GenTensor::<f64>::new_raw(&[7., -2., 1., 14., -7., -3., -7., 11., 18.], &[3,3]);
        let b = GenTensor::<f64>::new_raw(&[12., 17., 5.], &[3,1]);
        let x = cap_a.lu_solve(&b).unwrap();
        let ex = GenTensor::<f64>::new_raw(&[3., 4., -1.,], &[3,1]);
        assert_eq!(x, ex);
    }

    #[test]
    fn det() {
        let m = GenTensor::<f64>::new_raw(&[1., 1., 1., 4., 3., -1., 3., 5., 3.], &[3,3]);
        assert_eq!(m.det(), Some(10.));
    }

    #[test]
    fn qr() {
        let m = GenTensor::<f64>::new_raw(&[1., 1., 0., 1., 0., 1., 0., 1., 1.], &[3,3]);
        let [q, r] = m.qr().unwrap();
        let eq = GenTensor::<f64>::new_raw(&[0.7071067811865475, 0.40824829046386313, -0.5773502691896257,
                                             0.7071067811865475, -0.40824829046386296, 0.577350269189626,
        0., 0.8164965809277261, 0.5773502691896256, ], &[3,3]);
        let er = GenTensor::<f64>::new_raw(&[1.414213562373095, 0.7071067811865475, 0.7071067811865475, 0., 1.2247448713915894, 0.4082482904638632, 0., 0., 1.1547005383792515, ], &[3,3]);
        assert_eq!(q, eq);
        assert_eq!(r, er);
    }

    #[test]
    fn cholesky() {
        let m = GenTensor::<f64>::new_raw(&[4., 12., -16., 12., 37., -43., -16., -43., 98.], &[3,3]);
        let c = m.cholesky().unwrap();
        let ec = GenTensor::<f64>::new_raw(&[2., 6., -8., 0., 1., 5., 0., 0., 3.], &[3,3]);
        assert_eq!(c, ec);
    }

    #[test]
    fn eigen() {
        let m = GenTensor::<f64>::new_raw(&[4., 3., -2., -3.], &[2,2]);
        //let ec = GenTensor::<f64>::new_raw(&[4., 3., -2., -3.], &[2,2]);
        let el = GenTensor::<f64>::new_raw(&[3., -2.], &[2,1]);
        let [_evec, eval] = m.eigen().unwrap();
        //println!("{:?}, {:?}", _evec, eval);
        //println!("{:?}", eval.sub(&el).norm());
        assert!(eval.sub(&el).norm().get_scale() < 1e-6);
    }

    #[test]
    fn svd() {
        let m = GenTensor::<f64>::new_raw(&[4., 12., -16., 12., 37., -43., -16., -43., 98.], &[3,3]);
        let [_u, s, _v] = m.svd().unwrap();
        //println!("{:?}, {:?}, {:?}", u, s, v);
        let es = GenTensor::<f64>::new_raw(&[123.47723179013161, 15.503963229407585, 0.018804980460810704], &[3]);
        assert!(es.sub(&s.get_diag()).norm().get_scale() < 1e-6);

    }
}
