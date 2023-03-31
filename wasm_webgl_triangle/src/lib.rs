extern crate wasm_bindgen;

use std::f64;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{console, WebGlProgram, WebGl2RenderingContext, WebGlShader};

const vertex_shader_code: &'static str = r##"
     versioin 300 es
     attribute vec3 position;
     attribute vec4 color;
     varying vec4 v_color;
"##;

pub fn compile_shader(context: &WebGl2RenderingContext,
		      shader_type: u32,
		      source: &str) -> Result<WebGlShader, String>{
    let shader = context
	.create_shader(shader_type)
	.ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
	.get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
	.as_bool()
	.unwrap_or(false)
    {
	Ok(shader)
    }
    else {
	Err(context
	    .get_shader_info_log(&shader)
	    .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

#[wasm_bindgen]
pub fn start() -> Result<(),JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement =
	canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let context = canvas
	.get_context("webgl2")?
	.unwrap()
	.dyn_into::<WebGl2RenderingContext>()?;

    console::log_1(&JsValue::from_str("start webgl"));
    let vert_shader = compile_shader(
	&context,
	WebGl2RenderingContext::VERTEX_SHADER,
	vertex_shader_code);


    Ok(())
}
