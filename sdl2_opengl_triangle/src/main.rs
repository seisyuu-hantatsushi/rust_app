extern crate gl;
extern crate sdl2;

use std::thread;

use gl::types::{GLfloat, GLenum, GLuint, GLint, GLchar, GLsizeiptr, GLboolean};

use sdl2::video::{GLProfile};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;

fn main() -> Result<(),String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
	.window("rust-sdl2 OpenGL", 800, 600)
	.position_centered()
	.opengl()
	.build()
	.map_err(|e| e.to_string())?;
    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    canvas.set_draw_color(Color::RGB(255,0,0));
    canvas.clear();
    canvas.present();

    let mut event_pump = sdl_context.event_pump()?;

    'running: loop {
	for event in event_pump.poll_iter(){
	    match event {
		Event::Quit { .. }
		| Event::KeyDown {
		    keycode: Some(Keycode::Escape),
		    ..
		} => break 'running,
		_ => {}
	    }
	}

	canvas.clear();
	canvas.present();

	thread::yield_now();
    }

    Ok(())
}
