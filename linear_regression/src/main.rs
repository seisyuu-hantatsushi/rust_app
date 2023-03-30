/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

extern crate rand;
use rand::SeedableRng;
use rand::distributions::{Uniform, Distribution};
use rand_xorshift::XorShiftRng;

use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;

use plotters::prelude::full_palette::*;
use plotters::prelude::{Circle,BitMapBackend,LineSeries,ChartBuilder,PathElement};
use plotters::prelude::{SeriesLabelPosition};
use plotters::drawing::IntoDrawingArea;
use plotters::style::{IntoFont,Color};

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

#[cfg(not(feature="prototype"))]
fn linear_regression() -> Result<(),Box<dyn std::error::Error>> {
	fn f_y(x:f64) -> f64 {
		2.0*x + 5.0
	}

	let mut rng = XorShiftRng::from_entropy();
	let uniform_dist = Uniform::new(0.0,1.0);
	let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
	let uniform_dist = Uniform::new(-0.5,0.5);
	let ys:Vec<f64> = xs.iter().map(|&x| f_y(x)+uniform_dist.sample(&mut rng)).collect();

	let learning_rate:f64 = 0.1;

	let mut nn = NeuralNetwork::<f64>::new();
	let x = nn.create_constant("x", Tensor::<f64>::from_vector(vec![100,1], xs));
	let y = nn.create_constant("y", Tensor::<f64>::from_vector(vec![100,1], ys));

	let w = nn.create_neuron("W", Tensor::<f64>::zero(&[1,1]));
	let b = nn.create_neuron("b", Tensor::<f64>::zero(&[1,1]));
/*
	let xw = nn.matrix_product(Rc::clone(&x),Rc::clone(&w));
	let y_pred = nn.add(xw,Rc::clone(&b));
	 */
	let y_pred = nn.affine(Rc::clone(&x),Rc::clone(&w),Rc::clone(&b));
	let loss = nn.mean_square_error(Rc::clone(&y),y_pred);

	//println!("loss {}", loss.borrow());

	// create computational graph of 1st order differential.
	nn.backward_propagating(0)?;

	for _ in 0..100 {

		println!("w:{},b:{},loss:{}",
				 w.borrow().ref_signal()[vec![0,0]],
				 b.borrow().ref_signal()[vec![0,0]],
				 loss.borrow().ref_signal()[vec![0,0]]);

		nn.forward_propagating(0)?;
		nn.forward_propagating(1)?;

		//println!("loss {}\n", loss.borrow());

		let w_feedback = if let Some(ref gw) = w.borrow().ref_grad() {
			//println!("gw {}", gw.borrow().ref_signal());
			gw.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("w does not have grad".to_string())));
		};

		let updated_w = w.borrow().ref_signal() - w_feedback;
		//println!("updated_w = {}\n", updated_w);

		let b_feedback = if let Some(ref gb) = b.borrow().ref_grad() {
			//println!("gb {}", gb.borrow().ref_signal());
			gb.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("b does not have grad".to_string())));
		};

		let updated_b = b.borrow().ref_signal() - b_feedback;
		//println!("updated_b = {}\n", updated_b);

		w.borrow_mut().assign(updated_w);
		b.borrow_mut().assign(updated_b);
	}

	{
		let estimated_w = w.borrow().ref_signal()[vec![0,0]];
		let estimated_b = b.borrow().ref_signal()[vec![0,0]];
		let borrowed_x = x.borrow();
		let borrowed_y = y.borrow();
		let data_points:Vec<(&f64,&f64)> =
			borrowed_x.ref_signal().buffer().iter().zip(borrowed_y.ref_signal().buffer().iter()).collect();
		let render_backend = BitMapBackend::new("liner_regression.png", (800,600)).into_drawing_area();
		render_backend.fill(&WHITE)?;
		let mut data_points_chart_builder = ChartBuilder::on(&render_backend)
			.caption("linear regression", ("sans-serif", 50).into_font())
			.margin(10)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(0.0f32..1.0f32, 4.5f32..7.5f32)?;
		data_points_chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
		data_points_chart_builder.draw_series(data_points.iter().map(|(x,y)| Circle::new((**x as f32,**y as f32), 2, GREEN.filled())))?;
		data_points_chart_builder
			.draw_series(LineSeries::new([(0.0,estimated_b as f32),(1.0,(estimated_w+estimated_b) as f32)],&RED))?
			.label(format!("y = {} x + {}", estimated_w, estimated_b))
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], &RED));
		data_points_chart_builder
			.configure_series_labels()
			.position(SeriesLabelPosition::UpperLeft)
			.background_style(&WHITE.mix(0.8))
			.border_style(&BLACK)
			.draw()?;
		render_backend.present()?;
	}

	Ok(())
}

#[cfg(feature="prototype")]
fn linear_regression() -> Result<(),Box<dyn std::error::Error>> {
	fn f_y(x:f64) -> f64 {
		2.0*x + 5.0
	}

	let mut rng = XorShiftRng::from_entropy();
	let uniform_dist = Uniform::new(0.0,1.0);
	let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
	let uniform_dist = Uniform::new(-0.5,0.5);
	let ys:Vec<f64> = xs.iter().map(|&x| f_y(x)+uniform_dist.sample(&mut rng)).collect();

	let learning_rate:f64 = 0.1;

	let mut nn = NeuralNetwork::<f64>::new();
	let x = nn.create_constant("x", Tensor::<f64>::from_vector(vec![100,1], xs));
	let y = nn.create_constant("y", Tensor::<f64>::from_vector(vec![100,1], ys));

	let w = nn.create_neuron("W", Tensor::<f64>::zero(&[1,1]));
	let b = nn.create_neuron("b", Tensor::<f64>::zero(&[1,1]));
	let xw = nn.matrix_product(Rc::clone(&x),Rc::clone(&w));
	let y_pred = nn.add(xw,Rc::clone(&b));

	let diff = nn.sub(Rc::clone(&y),Rc::clone(&y_pred));
	let two = nn.create_constant("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
	let sample_size = nn.create_constant("sample_size", Tensor::<f64>::from_array(&[1,1],&[100.0]));
	let diff_squared = nn.pow(diff,two);
	let sum_diff_sqaure = nn.sum_to(diff_squared,vec![1,1]);
	let loss = nn.hadamard_division(sum_diff_sqaure,sample_size);

	for _ in 0..100 {
		nn.clear_grads(0)?;
		nn.forward_propagating(0)?;
		nn.backward_propagating(0)?;

		//println!("loss {}\n", loss.borrow());

		let w_feedback = if let Some(ref gw) = w.borrow().ref_grad() {
			//println!("gw {}", gw.borrow().ref_signal());
			gw.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("w does not have grad".to_string())));
		};

		let updated_w = w.borrow().ref_signal() - w_feedback;
		//println!("updated_w = {}\n", updated_w);

		let b_feedback = if let Some(ref gb) = b.borrow().ref_grad() {
			//println!("gb {}", gb.borrow().ref_signal());
			gb.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("b does not have grad".to_string())));
		};

		let updated_b = b.borrow().ref_signal() - b_feedback;
		//println!("updated_b = {}\n", updated_b);

		w.borrow_mut().assign(updated_w);
		b.borrow_mut().assign(updated_b);

		println!("w:{},b:{},loss:{}",
				 w.borrow().ref_signal()[vec![0,0]],
				 b.borrow().ref_signal()[vec![0,0]],
				 loss.borrow().ref_signal()[vec![0,0]]);
	}

	{
		let estimated_w = w.borrow().ref_signal()[vec![0,0]];
		let estimated_b = b.borrow().ref_signal()[vec![0,0]];
		let borrowed_x = x.borrow();
		let borrowed_y = y.borrow();
		let data_points:Vec<(&f64,&f64)> =
			borrowed_x.ref_signal().buffer().iter().zip(borrowed_y.ref_signal().buffer().iter()).collect();
		let render_backend = BitMapBackend::new("liner_regression.png", (800,600)).into_drawing_area();
		render_backend.fill(&WHITE)?;
		let mut data_points_chart_builder = ChartBuilder::on(&render_backend)
			.caption("linear regression", ("sans-serif", 50).into_font())
			.margin(10)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(0.0f32..1.0f32, 4.5f32..7.5f32)?;
		data_points_chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
		data_points_chart_builder.draw_series(data_points.iter().map(|(x,y)| Circle::new((**x as f32,**y as f32), 2, GREEN.filled())))?;
		data_points_chart_builder.draw_series(LineSeries::new([(0.0,estimated_b as f32),(1.0,(estimated_w+estimated_b) as f32)],&RED))?;
		render_backend.present()?;
	}

	Ok(())
}

fn main() -> Result<(),Box<dyn std::error::Error>> {
	linear_regression()
}
