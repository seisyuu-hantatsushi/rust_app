extern crate wasm_bindgen;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() -> String {
    "from wasm".to_string()
}

#[wasm_bindgen]
pub fn greet_alert(name: &str){
    alert(&format!("Hello, {}!", name));
}
