# webpackを使わない単純なweb assemblyのload
## 参考URL
* https://lealog.hateblo.jp/entry/2019/12/27/105424
* https://rustwasm.github.io/docs/wasm-bindgen/examples/without-a-bundler.html
## `wasm-bindgen`を追加
`cargo install wasm-bindgen-cli`
## RustによるWeb Assemblyプロジェクトを作成.
* `cargo new --lib <project_name>`
* `Cargo.toml`に以下を追加
```
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2.63"
```
*`src/lib.rs`にweb assemblyのコードを書く.

## Build
Release版の場合,`cargo build --target wasm32-unknown-unknown --release`

## ES Modulesにする
* [ES Modulesとは](https://codegrid.net/articles/2017-es-modules-1/)
