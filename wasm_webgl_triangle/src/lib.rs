extern crate wasm_bindgen;

use std::f64;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

#[wasm_bindgen]
pub fn start(){
    console::log_1(&JsValue::from_str("start webgl"));
}
