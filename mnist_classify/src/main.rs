/* -*- tab-width:4 -*- */

use std::rc::Rc;
use linear_transform::Tensor;

use deep_learning::datasets::*;
use deep_learning::utils::*;
use deep_learning::neural_network::NeuralNetwork;
use deep_learning::neural_network::model::MLPActivator;
use deep_learning::neural_network::optimizer::{SGD,Optimizer,NNOptimizer};

fn main() -> Result<(),Box<dyn std::error::Error>> {
	let max_epoch:usize = 5;
	let batch_size:usize = 100;
	let hidden_size = 1000;

	let train_datasets = mnist::get_dataset(true)?;
	let mut train_datasets_loader : loader::Loader<f64> =
		loader::Loader::new(Box::new(train_datasets), batch_size, true);
	let mut input_shape = train_datasets_loader.get_sample_shape();
	input_shape[0] = batch_size;

	let mut nn = NeuralNetwork::<f64>::new();
	let input_x = nn.create_constant("input_x", Tensor::<f64>::zero(&input_shape));
	let teacher_label = nn.create_constant("teacher", Tensor::<f64>::zero(&[batch_size,1]));
	let mut mlp_model = nn.create_mlp_model("MLP1", &[hidden_size,10], MLPActivator::Sigmoid);
	let classfied_result = nn.model_set_inputs(&mut mlp_model, vec![Rc::clone(&input_x)]);
	let mut optimizer = NNOptimizer::<f64>::new(Optimizer::SGD(SGD::new(0.01)),
												&mlp_model);
	let loss = nn.softmax_cross_entropy_error(Rc::clone(&classfied_result[0]), Rc::clone(&teacher_label));

	nn.backward_propagating(0)?;

	for epoch in 0..max_epoch {
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
			println!("loss {} {}", loss_val, batch_size);
			sum_loss += loss_val * batch_size as f64;
			sum_accuracy += accuracy * batch_size as f64;
		}
		let avg_loss = sum_loss/train_datasets_loader.get_num_of_samples() as f64;
		let avg_accuracy = sum_accuracy/train_datasets_loader.get_num_of_samples() as f64;
		println!("epoch {epoch} avg_loss {avg_loss} avg_accuracy {avg_accuracy}");
	}
	println!("finished training");

	{
		let test_datasets  = mnist::get_dataset(false)?;
		let mut test_datasets_loader : loader::Loader<f64> =
			loader::Loader::new(Box::new(test_datasets), batch_size, true);
		let mut sum_accuracy:f64 = 0.0;
		for (ts,xs) in test_datasets_loader.get_batchs() {
			input_x.borrow_mut().assign(xs);
			teacher_label.borrow_mut().assign(ts.clone());

			nn.forward_propagating(0)?;

			let classfied_argmax = classfied_result[0].borrow().ref_signal().argmax(1);
			let accuracy:f64 = accuracy(&classfied_argmax,
										&ts);
			println!("accuracy {}", accuracy);
			sum_accuracy += accuracy * batch_size as f64;
		}
		let avg_accuracy = sum_accuracy/test_datasets_loader.get_num_of_samples() as f64;
		println!("test result: avg_accuracy {avg_accuracy}");
	}

	Ok(())
}
