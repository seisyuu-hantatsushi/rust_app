use std::{env,mem};
use std::fs::File;
use std::collections::HashMap;
use std::io::prelude::*;
use std::path::Path;

//use linear_transform::matrix::MatrixMxN;
use linear_transform::tensor::tensor_base::Tensor;
use deep_learning::activator::Activator;
use deep_learning::output_layer::Softmax;
use deep_learning::loader::Loader;

use clap::{App,Arg};

static label_table:[char;47] = ['0','1','2','3','4','5','6','7','8','9',
				'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
				'a','b',    'd','e','f','g','h',                   'n',         'q','r','t'];


#[derive(Debug,Clone)]
struct AppContext {
    labels_file: String,
    images_file: String,
    weight_file: String
}

fn load_weight(weight_file:&str) -> Result<HashMap<String,Tensor<f32>>,String> {

    let weight_file_path = Path::new(weight_file);

    if !weight_file_path.exists() {
	return Err("weight file does not exists".to_string());
    }

    Tensor::<f32>::from_hdf5(weight_file)
}

fn mnist_load(labels_file: &str, images_file: &str) -> Result<(Vec<u32>,Vec<Tensor<f32>>),String> {
    Tensor::<f32>::from_mnist(labels_file, images_file, true, true)
}

fn predict(network:&HashMap<String,Tensor<f32>>, x:&Tensor<f32>) -> Tensor<f32> {
    let (w1, w2, w3) = (&network["w1"], &network["w2"], &network["w3"]);
    let (b1, b2, b3) = (&network["b1"], &network["b2"], &network["b3"]);

    let a1 = Tensor::<f32>::affine(x,w1,b1);
    let z1 = a1.sigmoid();
    let a2 = Tensor::<f32>::affine(&z1,w2,b2);
    let z2 = a2.sigmoid();
    let a3 = Tensor::<f32>::affine(&z2,w3,b3);

    return a3.softmax();
}

fn mnist_classify(ctx:AppContext) -> () {

    let (mnist_labels, mnist_images) = match mnist_load(&ctx.labels_file, &ctx.images_file) {
	Ok((ls, is)) => (ls, is),
	Err(e) => {
	    eprintln!("{}", e);
	    return ();
	}
    };

    let weights = match load_weight(&ctx.weight_file) {
	Ok(w) => w,
	Err(es) => {
	    eprintln!("{}",es);
	    return ();
	}
    };

    let mut accurancy_cnt = 0;
    for (x, l) in mnist_images.iter().zip(mnist_labels.iter()) {
	let y = predict(&weights, &x);
	let (p, _) = y.max_element_index();
	if *l == p[1] as u32 {
	    accurancy_cnt += 1;
	}
    }

    println!("Accurancy: {}", (accurancy_cnt as f64)/(mnist_images.len() as f64));

}

fn main() {

    let app_args = App::new("minst_classify")
	.version("0.1.0")
	.arg(Arg::with_name("labels_file")
	     .help("minst label file")
	     .short('l')
	     .long("labels_file")
	     .takes_value(true)
	     .required(true))
	.arg(Arg::with_name("images_file")
	     .help("minst image file")
	     .short('i')
	     .long("images_file")
	     .takes_value(true)
	     .required(true))
	.arg(Arg::with_name("weight_file")
	     .help("specified weight file")
	     .short('w')
	     .long("weight_file")
	     .takes_value(true)
	     .required(true))
	.arg(Arg::with_name("emnist")
	     .help("specified emnist format")
	     .long("emnist")
	     .takes_value(false));

    let ctx:AppContext = match app_args.try_get_matches() {
	Ok(m) => {
	    let ctx = AppContext {
		labels_file: String::from(m.value_of("labels_file").unwrap()),
		images_file: String::from(m.value_of("images_file").unwrap()),
		weight_file: String::from(m.value_of("weight_file").unwrap())
	    };
	    ctx
	},
	Err(e) => {
	    println!("Error {}", e);
	    return;
	}
    };

    mnist_classify(ctx);

}
