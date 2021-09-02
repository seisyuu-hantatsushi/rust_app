extern crate gl;
extern crate sdl2;

use std::ffi::CString;
use std::{ptr,str,mem};

use gl::types::{GLfloat, GLenum, GLuint, GLint, GLchar, GLsizeiptr, GLboolean};

use sdl2::video::{GLProfile};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

static VERTEX_DATA: [GLfloat; 6] = [
    0.0,    0.5,
    0.5,   -0.5,
   -0.5,   -0.5
];

static FRAGMENT_COLOR_DATA: [GLfloat; 9] = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
];

static VERTEX_SHADER_CODE: &'static str =
    r#"#version 150
    in vec2 position;
    in vec3 color;
    out vec3 frag_color;
    void main() {
       gl_Position = vec4(position, 0.0, 1.0);
       frag_color = color;
    }"#;

static FRAGMENT_SHADER_CODE: &'static str =
    r#"#version 150
    in vec3 frag_color;
    out vec4 out_color;
    void main() {
       out_color = vec4(frag_color, 1.0);
    }"#;

fn compile_shader(shader_code: &str, shader_type: GLenum) -> GLuint {
    let shader;
    unsafe {
        shader = gl::CreateShader(shader_type);
        // Attempt to compile the shader
        let c_str = CString::new(shader_code.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        // Get the compile status
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len - 1) as usize); // -1 removes null terminator
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            println!("{}", str::from_utf8(mem::transmute(buf.as_slice())).ok().expect("Failed to transmute shader error string!"));
        }
    }
    shader
}

fn link_program(vertex_shader: GLuint, fragment_shader: GLuint) -> GLuint {
    let program;
    unsafe {
        program = gl::CreateProgram();
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);

        // Get the link status
        let mut status = gl::FALSE as GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len - 1) as usize); // -1 removes null terminator
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            println!("{}", str::from_utf8(mem::transmute(buf.as_slice())).ok().expect("Failed to transmute program error string!"));
        }
    }
    program
}

fn main() {
    let sdl_context = sdl2::init().expect("Failed to create SDL2 context!");
    let video_subsystem = sdl_context.video().expect("Failed to get SDL2 video subsystem!");

    let gl_attr = video_subsystem.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_context_version(3, 3);

    let window = video_subsystem.window("Rust SDL2", 1024, 768).opengl().build().unwrap();

    let _context = window.gl_create_context().unwrap();
    gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as *const std::os::raw::c_void);
    video_subsystem.gl_set_swap_interval(1).unwrap_or_else(|err| panic!("err = {}", err));

    let vs = compile_shader(VERTEX_SHADER_CODE, gl::VERTEX_SHADER);
    let fs = compile_shader(FRAGMENT_SHADER_CODE, gl::FRAGMENT_SHADER);
    let program = link_program(vs, fs);

    let mut vao = 0;
    let mut vbos : [GLuint;2] = [0,0];

    unsafe {
	let vertex_vbo_index = 0;
	let color_vbo_index  = 1;
        gl::GenVertexArrays(1, &mut vao);
	gl::BindVertexArray(vao);

        gl::GenBuffers(2, &mut vbos[0]);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbos[vertex_vbo_index]);
        gl::BufferData(gl::ARRAY_BUFFER,
                       (VERTEX_DATA.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       mem::transmute(&VERTEX_DATA[0]),
                       gl::STATIC_DRAW);


	gl::BindBuffer(gl::ARRAY_BUFFER, vbos[color_vbo_index]);
	gl::BufferData(gl::ARRAY_BUFFER,
                       (FRAGMENT_COLOR_DATA.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       mem::transmute(&FRAGMENT_COLOR_DATA[0]),
                       gl::STATIC_DRAW);


	gl::UseProgram(program);

	let out_color_str = CString::new("out_color").unwrap_or_else(|_| panic!("failed to allocate string space"));
	let out_color_str_ptr = out_color_str.as_ptr();
	gl::BindFragDataLocation(program, 0, out_color_str_ptr);

	let position_str = CString::new("position").unwrap_or_else(|_| panic!("failed to allocate string space"));
	let position_str_ptr = position_str.as_ptr();
	{
	    let location : GLuint  = gl::GetAttribLocation(program, position_str_ptr) as GLuint;
	    println!("location {}", location);
	    gl::EnableVertexAttribArray(location);
	    gl::BindBuffer(gl::ARRAY_BUFFER, vbos[vertex_vbo_index]);
	    gl::VertexAttribPointer(location, 2, gl::FLOAT, gl::FALSE as GLboolean, 0, ptr::null());
	}


	let in_color_str = CString::new("color").unwrap_or_else(|_| panic!("failed to allocate string space"));
	let in_color_str_ptr = in_color_str.as_ptr();
	{
	    let location : GLuint  = gl::GetAttribLocation(program, in_color_str_ptr) as GLuint;
	    println!("location {}", location as GLint);
	    gl::EnableVertexAttribArray(location);
	    gl::BindBuffer(gl::ARRAY_BUFFER, vbos[color_vbo_index]);
	    gl::VertexAttribPointer(location, 3, gl::FLOAT, gl::FALSE as GLboolean, 0, ptr::null());
	}

    }

    // this should only fail if you already have a event pump
    let mut ep = sdl_context.event_pump().expect("Failed to get SDL2 event pump!");

    'mainloop: loop {
        loop {
            match ep.poll_event().unwrap_or(Event::Unknown{ timestamp: 0, type_: 0}) {
                Event::Unknown{ .. } => break,
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'mainloop,
                Event::Quit { .. } => break 'mainloop,
                Event::Window { win_event, .. } => {
                    match win_event {
                        sdl2::event::WindowEvent::Resized (w, h) => unsafe { gl::Viewport(0, 0, w, h) },
                        _ => (),
                    }
                },
                _ => (),
            }
        }

        unsafe {
            gl::ClearColor(0.3, 0.3, 0.3, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

	    gl::BindVertexArray(vao);
            // draw trangle
            gl::DrawArrays(gl::TRIANGLES, 0, 3);
	    gl::BindVertexArray(0);
        }

        window.gl_swap_window();
    }
}
