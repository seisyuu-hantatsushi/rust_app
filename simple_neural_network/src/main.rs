/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use rand::SeedableRng;
use rand_distr::{Normal, Uniform, Distribution};
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

fn main() -> Result<(),Box<dyn std::error::Error>>{
    let mut rng = XorShiftRng::from_entropy();
    let uniform_dist = Uniform::new(0.0,1.0);
    let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
    let uniform_dist = Uniform::new(0.0,1.0);
    let ys:Vec<f64> = xs.iter().map(|&x| (2.0*std::f64::consts::PI*x+uniform_dist.sample(&mut rng)).sin()).collect();

    let normal_dist = Normal::new(0.0, 1.0)?;
    let mut nn = NeuralNetwork::<f64>::new();
    let x = nn.create_constant("x", Tensor::<f64>::from_vector(vec![100,1],xs));
    let y = nn.create_constant("y", Tensor::<f64>::from_vector(vec![100,1],ys));

    let w1 = nn.create_neuron("W1", Tensor::<f64>::from_vector(vec![1,10],
							       (0..10).map(|_| 0.01 * normal_dist.sample(&mut rng)).collect()));
    let b1 = nn.create_neuron("b1", Tensor::<f64>::zero(&[1,1]));
    let w2 = nn.create_neuron("W2", Tensor::<f64>::from_vector(vec![10,1],
							       (0..10).map(|_| 0.01 * normal_dist.sample(&mut rng)).collect()));
    let b2 = nn.create_neuron("b2", Tensor::<f64>::zero(&[1,1]));

    let term1  = nn.affine(Rc::clone(&x), Rc::clone(&w1), Rc::clone(&b1));
	term1.borrow_mut().rename("affine1");
	let term2  = nn.sigmoid(Rc::clone(&term1));
	let pred_y = nn.affine(Rc::clone(&term2), Rc::clone(&w2), Rc::clone(&b2));
	pred_y.borrow_mut().rename("pred_y");
	let loss   = nn.mean_square_error(Rc::clone(&y), Rc::clone(&pred_y));
	let learning_rate = 0.2;

	println!("w1 {}\n", w1.borrow());
	println!("b1 {}\n", b1.borrow());
	println!("w2 {}\n", w2.borrow());
	println!("b2 {}\n", b2.borrow());
	println!("loss {}", loss.borrow());

	nn.backward_propagating(0)?;

	if let Err(e) = nn.make_dot_graph(0,"nn_order0.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(1,"nn_order1.dot") {
		println!("{}",e);
		assert!(false)
	}

	for i in 0..10000 {

		let w_feedback = if let Some(ref gw1) = w1.borrow().ref_grad() {
			gw1.borrow().ref_signal().scale(learning_rate)
		}
		else {
		return Err(Box::new(MyError::StringMsg("w1 does not have grad".to_string())));
		};
		//println!("w_feedback {}\n", w_feedback);
		let updated_w = w1.borrow().ref_signal() - w_feedback;
		w1.borrow_mut().assign(updated_w);

		let b_feedback = if let Some(ref gb1) = b1.borrow().ref_grad() {
			gb1.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("b1 does not have grad".to_string())));
		};

		let updated_b = b1.borrow().ref_signal() - b_feedback;
		b1.borrow_mut().assign(updated_b);

		let w_feedback = if let Some(ref gw2) = w2.borrow().ref_grad() {
			gw2.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("w2 does not have grad".to_string())));
		};

		let updated_w = w2.borrow().ref_signal() - w_feedback;
		w2.borrow_mut().assign(updated_w);

		let b_feedback = if let Some(ref gb2) = b2.borrow().ref_grad() {
			gb2.borrow().ref_signal().scale(learning_rate)
		}
		else {
			return Err(Box::new(MyError::StringMsg("b2 does not have grad".to_string())));
		};

		let updated_b = b2.borrow().ref_signal() - b_feedback;
		b2.borrow_mut().assign(updated_b);

		if i % 1000 == 0 {
			println!("{}", loss.borrow());
		}

		nn.forward_propagating(0)?;
		nn.forward_propagating(1)?;

		/*
		let result = nn.clear_grads(0);
		if let Err(s) = result {
			return Err(Box::new(MyError::StringMsg(s)));
		}
		nn.forward_propagating(0)?;
		nn.backward_propagating(0)?;
*/
	}

    {
		let borrowed_x = x.borrow();
		let borrowed_y = y.borrow();
		let data_points: Vec<(&f64, &f64)> =
			borrowed_x.ref_signal().buffer().iter().zip(borrowed_y.ref_signal().buffer().iter()).collect();
		let borrowed_w1 = w1.borrow();
		let borrowed_b1 = b1.borrow();
		let borrowed_w2 = w2.borrow();
		let borrowed_b2 = b2.borrow();

		let w1 = borrowed_w1.ref_signal();
		let b1 = borrowed_b1.ref_signal();
		let w2 = borrowed_w2.ref_signal();
		let b2 = borrowed_b2.ref_signal();

		let expand_b1 = b1.broadcast(w1.shape());

		let pred_y = |x:f64| -> f64 {
			let xt = Tensor::<f64>::from_array(&[1,1],&[x]);
			let y  = Tensor::<f64>::affine(&xt, w1, &expand_b1);
			let y  = y.sigmoid();
			let y  = Tensor::<f64>::affine(&y, w2, b2);
			y[vec![0,0]]
		};

		let pred_points: Vec<(f32, f32)> =
			(0..100).map(|i| { let x = (i as f64)/100.0;
							   (x as f32, pred_y(x) as f32) }).collect();
		let render_backend = BitMapBackend::new("simple_nn.png", (800,600)).into_drawing_area();
		render_backend.fill(&WHITE)?;
		let mut data_points_chart_builder = ChartBuilder::on(&render_backend)
			.caption("simple neural network", ("sans-serif", 50).into_font())
			.margin(10)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(0.0f32..1.0f32, -2.0f32..2.0f32)?;
		data_points_chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
		data_points_chart_builder.draw_series(data_points.iter().map(|(x,y)| Circle::new((**x as f32,**y as f32), 2, GREEN.filled())))?;
		data_points_chart_builder
			.draw_series(LineSeries::new(pred_points,&RED))?
			.label(format!("prediction"))
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
