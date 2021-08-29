
all: release

release: release_pkg/wasm_webgl_triangle_bg.wasm

target/wasm32-unknown-unknown/release/wasm_webgl_triangle.wasm: src/lib.rs
	cargo build --target wasm32-unknown-unknown --release


release_pkg/wasm_webgl_triangle_bg.wasm: target/wasm32-unknown-unknown/release/wasm_webgl_triangle.wasm
	wasm-bindgen target/wasm32-unknown-unknown/release/wasm_webgl_triangle.wasm --out-dir ./release_pkg --target web

clean:
	rm -rf ./release_pkg
	cargo clean
