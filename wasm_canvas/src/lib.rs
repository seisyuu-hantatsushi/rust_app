extern crate wasm_bindgen;

use std::f64;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

#[wasm_bindgen]
pub fn start(){
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas   = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
	.dyn_into::<web_sys::HtmlCanvasElement>()
	.map_err(|_| ())
	.unwrap();
    let context = canvas
	.get_context("2d")
	.unwrap()
	.unwrap()
	.dyn_into::<web_sys::CanvasRenderingContext2d>()
	.unwrap();
    console::log_1(&JsValue::from_str("start draw canvas"));
    context.begin_path();
    // Draw the outer circle.
    context
        .arc(75.0, 75.0, 50.0, 0.0, f64::consts::PI * 2.0)
        .unwrap();
    context.stroke();
}
