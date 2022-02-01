use std::fs::File;
use std::path::Path;
use std::io;
use std::io::Read;

use tensor_rs::tensor::Tensor;

use tensorboard_rs::summary_writer::SummaryWriter;

pub fn load_images<P: AsRef<Path>>(path: P) -> Tensor {
    let ref mut reader = io::BufReader::new(File::open(path).expect(""));
    let magic = read_as_u32(reader);
    if magic != 2051 {
        panic!("Invalid magic number. expected 2051, got {}", magic)
    }
    let num_image = read_as_u32(reader) as usize;
    let rows = read_as_u32(reader) as usize;
    let cols = read_as_u32(reader) as usize;
    assert!(rows == 28 && cols == 28);

    // read images
    let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
    let _ = reader.read_exact(buf.as_mut());
    let ret: Vec<f32> = buf.into_iter().map(|x| (x as f32) / 255.).collect();
    let ret = Tensor::from_vec_f32(&ret[..], &vec![num_image, rows, cols]);
    ret
}

pub fn load_labels<P: AsRef<Path>>(path: P) -> Tensor {
    let ref mut reader = io::BufReader::new(File::open(path).expect(""));
    let magic = read_as_u32(reader);
    if magic != 2049 {
        panic!("Invalid magic number. Got expect 2049, got {}", magic);
    }
    let num_label = read_as_u32(reader) as usize;
    // read labels
    let mut buf: Vec<u8> = vec![0u8; num_label];
    let _ = reader.read_exact(buf.as_mut());
    let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
    let ret = Tensor::from_vec_f32(&ret[..], &vec![num_label]);
    ret
}

fn read_as_u32<T: Read>(reader: &mut T) -> u32 {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    let _ = reader.read_exact(&mut buf);
    u32::from_be_bytes(buf)
}

#[allow(dead_code)]
pub fn main() {
    let t = load_images("examples/data/mnist/train-images-idx3-ubyte");

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    for i in 0..10 {
        let first_image = t.get_patch(&vec![(i,i+1),(0,28),(0,28)], None);
    //println!("{:?}, {}, {}", first_image.size(), first_image.max(None, None, None), first_image.min(None, None, None));
        let rgb_img = first_image.cat(&vec![&first_image, &first_image], 0);
        let rgb_img = rgb_img.permute(&vec![1, 2, 0]);
        let rgb_img = rgb_img.mul(&Tensor::fill(&vec![1], &Tensor::from_vec_f64(&[255.], &[1])));
        writer.add_image(&"test_image".to_string(), &rgb_img.get_u8().expect("u8")[..], &vec![3, 28, 28][..], i+32);
    }
    
    let first_image = t.get_patch(&vec![(0,1),(0,28),(0,28)], None);
    //println!("{:?}, {}, {}", first_image.size(), first_image.max(None, None, None), first_image.min(None, None, None));
    let rgb_img = first_image.cat(&vec![&first_image, &first_image], 0);
    let rgb_img = rgb_img.permute(&vec![1, 2, 0]);
    let rgb_img = rgb_img.mul(&Tensor::fill(&vec![1], &Tensor::from_vec_f64(&[255.], &[1])));
    writer.add_image(&"test_image".to_string(), &rgb_img.get_u8().expect("u8")[..], &vec![3, 28, 28][..], 12);
    writer.flush();


    let first_image = t.get_patch(&vec![(10,11),(0,28),(0,28)], None);
    //println!("{:?}, {}, {}", first_image.size(), first_image.max(None, None, None), first_image.min(None, None, None));
    let rgb_img = first_image.cat(&vec![&first_image, &first_image], 0);
    let rgb_img = rgb_img.permute(&vec![1, 2, 0]);
    let rgb_img = rgb_img.mul(&Tensor::fill(&vec![1], &Tensor::from_vec_f64(&[255.], &[1])));
    writer.add_image(&"test_image".to_string(), &rgb_img.get_u8().expect("u8")[..], &vec![3, 28, 28][..], 13);
    writer.flush();

    let l = load_labels("examples/data/mnist/train-labels-idx1-ubyte");
    println!("{}, {}", l.get_f32(&vec![0]), l.get_f32(&vec![10]));
}
