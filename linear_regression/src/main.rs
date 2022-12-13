/* -*- tab-width:4 -*- */

extern crate rand;
use rand::SeedableRng;
use rand::distributions::{Uniform, Distribution};
use rand_xorshift::XorShiftRng;

use plotters::prelude::full_palette::*;
use plotters::prelude::{Circle,BitMapBackend,LineSeries,ChartBuilder,PathElement};
use plotters::drawing::IntoDrawingArea;
use plotters::style::{IntoFont,Color};

fn main() -> Result<(),Box<dyn std::error::Error>> {

	fn f_y(x:f64) -> f64 {
		2.0*x + 5.0
	}

	let mut rng = XorShiftRng::from_entropy();
	let uniform_dist = Uniform::new(0.0,1.0);
	let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
	let uniform_dist = Uniform::new(-0.5,0.5);
	let data_points:Vec<(f64,f64)> = xs.iter().map(|&x| (x,f_y(x)+uniform_dist.sample(&mut rng))).collect();
	
	println!("{:?}",xs);
	println!("{:?}", data_points);

	let render_backend = BitMapBackend::new("liner_regression.png", (800,600)).into_drawing_area();
	render_backend.fill(&WHITE)?;
	let mut data_points_chart_builder = ChartBuilder::on(&render_backend)
		.margin(10)
		.x_label_area_size(30)
		.y_label_area_size(30)
		.build_cartesian_2d(0.0f32..1.0f32, 5.0f32..7.5f32)?;
	data_points_chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
	data_points_chart_builder.draw_series(data_points.iter().map(|(x,y)| Circle::new((*x as f32,*y as f32), 1, GREEN.filled())))?;
	render_backend.present()?;
	Ok(())
}
