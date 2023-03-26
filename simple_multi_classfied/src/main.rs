/* -*- tab-width:4 -*- */
use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;
use std::ops::Range;
use linear_transform::Tensor;

use deep_learning::neural_network::NeuralNetwork;
use deep_learning::datasets::*;
use deep_learning::neural_network::model::MLPActivator;
use deep_learning::neural_network::optimizer::{SGD,MomentumSDG,Optimizer,NNOptimizer};

use rand::prelude::SliceRandom;

use plotters::prelude::{BitMapBackend,ChartBuilder};
use plotters::prelude::{Circle,Cross,Rectangle,TriangleMarker,EmptyElement,LineSeries,
						SeriesLabelPosition};
use plotters::prelude::full_palette::*;
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

fn main() -> Result<(),Box<dyn std::error::Error>> {
	let max_epoch = 300;
	let batch_size = 30;
	let num_of_class = 3;
	let num_of_data = 100;
	let num_of_alldata = num_of_class * num_of_data;
	let (xs,ts) = spiral::get_2d_dataset::<f64>(num_of_class,num_of_data);

	{
		let render_backend = BitMapBackend::new("spiral_graph.png", (640, 480)).into_drawing_area();
		render_backend.fill(&WHITE)?;

		let mut chart_builder = ChartBuilder::on(&render_backend)
			.caption("spiral", ("sans-serif", 40).into_font())
			.margin(5)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(-1.56f32..1.56f32, -1.2f32..1.2f32)?;
		chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

		let class_iter =
			xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 0.0 as f64);
		chart_builder.draw_series(class_iter.map(|(xst,t)| {
			let x = xst[vec![0,0]] as f32;
			let y = xst[vec![0,1]] as f32;
			EmptyElement::at((x,y)) + Circle::new((0,0), 2, GREEN.filled())
		}))?;
		let class_iter =
			xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 1.0 as f64);
		chart_builder.draw_series(class_iter.map(|(xst,t)| {
			let x = xst[vec![0,0]] as f32;
			let y = xst[vec![0,1]] as f32;
			EmptyElement::at((x,y)) + TriangleMarker::new((0,0), 2, RED.filled())
		}))?;
		let class_iter =
			xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 2.0 as f64);
		chart_builder.draw_series(class_iter.map(|(xst,t)| {
			let x = xst[vec![0,0]] as f32;
			let y = xst[vec![0,1]] as f32;
			EmptyElement::at((x,y)) + Cross::new((0,0), 2, CYAN.filled())
		}))?;

		render_backend.present()?;
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let input_x = nn.create_constant("input_x", Tensor::<f64>::zero(&[batch_size,2]));
		let teacher_label = nn.create_constant("teacher", Tensor::<f64>::zero(&[batch_size,1]));

		let mut mlp_model = nn.create_mlp_model("MLP1", &[10,num_of_class], MLPActivator::Sigmoid);
		let classfied_result = nn.model_set_inputs(&mut mlp_model, vec![Rc::clone(&input_x)]);
		let mut optimizer = NNOptimizer::<f64>::new(Optimizer::SGD(SGD::new(1.0)),
													&mlp_model);
		let loss = nn.softmax_cross_entropy_error(Rc::clone(&classfied_result[0]), Rc::clone(&teacher_label));
		let max_iter = (num_of_data*num_of_class)/batch_size;
		let mut avgloss_at_epoch:Vec<(f64,f64)> = vec!();
		nn.backward_propagating(0)?;

		//nn.make_dot_graph(0,"order0.dot")?;
		//nn.make_dot_graph(1,"order1.dot")?;
		{
			for epoch in 0..max_epoch {
				let mut sum_loss:f64 = 0.0;
				let perm_table = {
					let mut v = (0..(num_of_data*num_of_class)).collect::<Vec<usize>>();
					v.shuffle(&mut nn.get_rng());
					v
				};
				for i in 0..max_iter {
					// make mini batch
					let batch_index = &perm_table[i*batch_size..(i+1)*batch_size];
					let batch_x = xs.selector(batch_index);
					let batch_t = ts.selector(batch_index);

					input_x.borrow_mut().assign(batch_x);
					teacher_label.borrow_mut().assign(batch_t);

					//println!("{}",x.borrow().ref_signal());
					//println!("{}",t.borrow().ref_signal());
					nn.forward_propagating(0)?;
					nn.forward_propagating(1)?;
					optimizer.update()?;

					println!("loss {} {}", loss.borrow().ref_signal()[vec![0,0]], batch_size);
					sum_loss += loss.borrow().ref_signal()[vec![0,0]] * (batch_size as f64);

				}
				//println!("sum_loss {} {}",sum_loss,num_of_alldata);
				let avg_of_loss = sum_loss / (num_of_alldata as f64);
				println!("epoch {}, avg loss {}", epoch, avg_of_loss);
				avgloss_at_epoch.push((epoch as f64, avg_of_loss));
			}

			let render_backend = BitMapBackend::new("epoch_loss.png", (640, 480)).into_drawing_area();
			render_backend.fill(&WHITE)?;

			let mut chart_builder = ChartBuilder::on(&render_backend)
				.caption("epoch loss", ("sans-serif", 40).into_font())
				.margin(5)
				.x_label_area_size(30)
				.y_label_area_size(30)
				.build_cartesian_2d(0.0f32..300.0f32, 0.0f32..1.4f32)?;
			chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

			chart_builder.draw_series(LineSeries::new(avgloss_at_epoch.iter().map(|(x,y)| { (*x as f32, *y as f32) }),&BLUE))?;
			render_backend.present()?;
		}

		let mut predict = |x:f64, y:f64| -> usize {
			let input = Tensor::<f64>::from_array(&[1,2],&[x,y]);
			input_x.borrow_mut().assign(input);
			let _ = nn.forward_propagating(0);
			let (cls_predict,_) = classfied_result[0].borrow().ref_signal().max_element_index();
			cls_predict[1]
		};

		fn predict_plot_pos(x_range:Range<f32>, y_range:Range<f32>, pixel_samples:(usize,usize))
						-> impl Iterator<Item = (f32, f32)> {
			let (step_x,step_y) = (
				(x_range.end-x_range.start)/(pixel_samples.0 as f32),
				(y_range.end-y_range.start)/(pixel_samples.1 as f32)
			);
			(0..(pixel_samples.0*pixel_samples.1)).map(move |k| {
				(x_range.start + step_x * (k % pixel_samples.0) as f32,
				 y_range.start + step_y * (k / pixel_samples.0) as f32)
			})
		}

		{
			let render_backend = BitMapBackend::new("classified_spiral_graph.png", (640, 480)).into_drawing_area();
			render_backend.fill(&WHITE)?;

			let mut chart_builder = ChartBuilder::on(&render_backend)
				.caption("sprial classification result", ("sans-serif", 40).into_font())
				.margin(5)
				.x_label_area_size(30)
				.y_label_area_size(30)
				.build_cartesian_2d(-1.56f32..1.56f32, -1.2f32..1.2f32)?;
			chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

			let plotting_area = chart_builder.plotting_area();
			let range = plotting_area.get_pixel_range();

			let (pw,ph) = (range.0.end - range.0.start, range.1.end - range.1.start);
			let (xr,yr) = (chart_builder.x_range(),chart_builder.y_range());

			for (x,y) in predict_plot_pos(xr,yr,(pw as usize,ph as usize)){
				match predict(x as f64, y as f64) {
					0 => { plotting_area.draw_pixel((x,y),&AMBER)?; },
					1 => { plotting_area.draw_pixel((x,y),&BLUE)?; },
					2 => { plotting_area.draw_pixel((x,y),&LIME)?; },
					_ => { plotting_area.draw_pixel((x,y),&BLACK)?; }
				}
			}

			let class_iter =
				xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 0.0 as f64);
			chart_builder.draw_series(class_iter.map(|(xst,t)| {
				let x = xst[vec![0,0]] as f32;
				let y = xst[vec![0,1]] as f32;
				EmptyElement::at((x,y)) + Circle::new((0,0), 2, GREEN.filled())
			}))?;
			let class_iter =
				xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 1.0 as f64);
			chart_builder.draw_series(class_iter.map(|(xst,t)| {
				let x = xst[vec![0,0]] as f32;
				let y = xst[vec![0,1]] as f32;
				EmptyElement::at((x,y)) + TriangleMarker::new((0,0), 2, RED.filled())
			}))?;
			let class_iter =
				xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 2.0 as f64);
			chart_builder.draw_series(class_iter.map(|(xst,t)| {
				let x = xst[vec![0,0]] as f32;
				let y = xst[vec![0,1]] as f32;
				EmptyElement::at((x,y)) + Cross::new((0,0), 2, CYAN.filled())
			}))?;

			render_backend.present()?;
		}
	}

	Ok(())
}
