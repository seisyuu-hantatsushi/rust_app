/* -*- tab-width:4 -*- */

use clap::{Arg, ArgAction, Command, value_parser};

use std::fmt::Display;
use std::error::Error;
use std::rc::Rc;
use linear_transform::Tensor;

use deep_learning::datasets::*;
use deep_learning::utils::*;
use deep_learning::neural_network::NeuralNetwork;
use deep_learning::neural_network::model::MLPActivator;
use deep_learning::neural_network::optimizer::{SGD,Optimizer,NNOptimizer};

#[derive(Debug)]
enum MyError {
    StringMsg(String)
}

impl Display for MyError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		use self::MyError::*;
		match self {
			StringMsg(s) => write!(f, "{}", s)
		}
	}
}

impl Error for MyError {}

#[derive(Debug,Clone)]
struct AppContext {
	max_epoch:usize,
	batch_size:usize,
	hidden_size:usize,
	activator: MLPActivator,
	num_of_layers: usize
}

fn main() -> Result<(),Box<dyn std::error::Error>> {

	let cmd = Command::new("mnist_classify")
		.version("0.1.0")
		.arg(Arg::new("activator")
			 .help("select activator. sigmod or relu")
			 .short('a')
			 .long("activator")
			 .action(ArgAction::Set)
			 .default_value("sigmod"))
		.arg(Arg::new("layer")
			 .help("specified number of layers.")
			 .short('l')
			 .long("layer")
			 .value_parser(value_parser!(usize))
			 .action(ArgAction::Set)
			 .default_value("1"));

	let ctx:AppContext = match cmd.try_get_matches() {
		Ok(m) => {
			let activator_str:&str = m.get_one::<String>("activator").unwrap();
			let activator = match activator_str {
				"sigmod" => MLPActivator::Sigmoid,
				"relu" => MLPActivator::ReLU,
				_ => unreachable!()
			};
			let num_of_layers:&usize = m.get_one::<usize>("layer").unwrap();
			AppContext {
				max_epoch: 5,
				batch_size: 100,
				hidden_size: 1000,
				activator,
				num_of_layers: *num_of_layers
			}
		},
		Err(e) => {
			println!("argument error {}", e);
			return Ok(());
			//return Err(Box::new(e));
		}
	};

	let train_datasets = mnist::get_dataset(true)?;
	let mut train_datasets_loader : loader::Loader<f64> =
		loader::Loader::new(Box::new(train_datasets), ctx.batch_size, true);
	let mut input_shape = train_datasets_loader.get_sample_shape();
	input_shape[0] = ctx.batch_size;

	match ctx.activator {
		MLPActivator::Sigmoid => println!("activator is sigmode"),
		MLPActivator::ReLU => println!("activator is relu")
	};

	let mut layer_shape:Vec<usize> = (0..ctx.num_of_layers).map(|_| ctx.hidden_size).collect();
	layer_shape.push(10);
	println!("layer shape {:?}", layer_shape);
	let mut nn = NeuralNetwork::<f64>::new();
	let input_x = nn.create_constant("input_x", Tensor::<f64>::zero(&input_shape));
	let teacher_label = nn.create_constant("teacher", Tensor::<f64>::zero(&[ctx.batch_size,1]));
	let mut mlp_model = nn.create_mlp_model("MLP1", &layer_shape, ctx.activator);
	let classfied_result = nn.model_set_inputs(&mut mlp_model, vec![Rc::clone(&input_x)]);
	let mut optimizer = NNOptimizer::<f64>::new(Optimizer::SGD(SGD::new(0.01)),
												&mlp_model);
	let loss = nn.softmax_cross_entropy_error(Rc::clone(&classfied_result[0]), Rc::clone(&teacher_label));

	nn.backward_propagating(0)?;

	for epoch in 0..ctx.max_epoch {
		let (mut sum_loss, mut sum_accuracy):(f64,f64) = (0.0,0.0);
		for (ts,xs) in train_datasets_loader.get_batchs() {

			input_x.borrow_mut().assign(xs);
			teacher_label.borrow_mut().assign(ts.clone());

			nn.forward_propagating(0)?;
			nn.forward_propagating(1)?;
			let classfied_argmax = classfied_result[0].borrow().ref_signal().argmax(1);
			let accuracy:f64 = accuracy(&classfied_argmax,
										&ts);
			println!("accuracy {}", accuracy);
			optimizer.update()?;
			let loss_val = loss.borrow().ref_signal()[vec![0,0]];
			println!("loss {} {}", loss_val, ctx.batch_size);
			sum_loss += loss_val * ctx.batch_size as f64;
			sum_accuracy += accuracy * ctx.batch_size as f64;
		}
		let avg_loss = sum_loss/train_datasets_loader.get_num_of_samples() as f64;
		let avg_accuracy = sum_accuracy/train_datasets_loader.get_num_of_samples() as f64;
		println!("epoch {epoch} avg_loss {avg_loss} avg_accuracy {avg_accuracy}");
	}
	println!("finished training");

	{
		let test_datasets  = mnist::get_dataset(false)?;
		let mut test_datasets_loader : loader::Loader<f64> =
			loader::Loader::new(Box::new(test_datasets), ctx.batch_size, true);
		let mut sum_accuracy:f64 = 0.0;
		for (ts,xs) in test_datasets_loader.get_batchs() {
			input_x.borrow_mut().assign(xs);
			teacher_label.borrow_mut().assign(ts.clone());

			nn.forward_propagating(0)?;

			let classfied_argmax = classfied_result[0].borrow().ref_signal().argmax(1);
			let accuracy:f64 = accuracy(&classfied_argmax,
										&ts);
			println!("accuracy {}", accuracy);
			sum_accuracy += accuracy * ctx.batch_size as f64;
		}
		let avg_accuracy = sum_accuracy/test_datasets_loader.get_num_of_samples() as f64;
		println!("test result: avg_accuracy {avg_accuracy}");
	}

	Ok(())
}
