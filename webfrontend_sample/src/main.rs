
extern crate clap;

use serde::{Deserialize, Serialize};
use actix_web::{get, post, web, client, HttpRequest, HttpResponse, HttpServer, Responder};
use actix_files::{NamedFile, Files};

use std::{env,io,fs};
use std::time::SystemTime;
use std::path::PathBuf;

mod greet;

struct AppContext {
    document_root: String,
    restfulapi_server_address: String
}

async fn index(data: web::Data<AppContext>, _req: HttpRequest) -> actix_web::Result<NamedFile> {
    let path = format!("{}{}", data.document_root, "/index.html");
    let path_buf:PathBuf = path.parse().unwrap();
    Ok(NamedFile::open(path_buf)?)
}

async fn webasm_greet(data: web::Data<AppContext>, _req: HttpRequest) -> actix_web::Result<NamedFile> {
    let path = format!("{}{}", data.document_root, "/webasm_greet.html");
    let path_buf:PathBuf = path.parse().unwrap();
    Ok(NamedFile::open(path_buf)?)
}

async fn webasm_canvas(data: web::Data<AppContext>, _req: HttpRequest) -> actix_web::Result<NamedFile> {
    let path = format!("{}{}", data.document_root, "/webasm_canvas.html");
    let path_buf:PathBuf = path.parse().unwrap();
    Ok(NamedFile::open(path_buf)?)
}

async fn webasm_webgl_triangle(data: web::Data<AppContext>, _req: HttpRequest) -> actix_web::Result<NamedFile> {
    let path = format!("{}{}", data.document_root, "/webasm_webgl_triangle.html");
    let path_buf:PathBuf = path.parse().unwrap();
    Ok(NamedFile::open(path_buf)?)
}

#[actix_web::main]
async fn main() -> io::Result<()> {
    const NAME: &'static str = env!("CARGO_PKG_NAME");
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUTHORS: &'static str = env!("CARGO_PKG_AUTHORS");

    env::set_var("RUST_LOG", "actix_web=debug,actix_server=info");
    env_logger::init();

    let arg_matches = clap::App::new(NAME)
	.version(VERSION)
	.author(AUTHORS)
	.about("webui frontend")
	.arg(clap::Arg::with_name("bind")
	     .short("b")
	     .long("bind")
	     .value_name("ADDRESS_AND_PORT")
	     .help("specify to bind address and port. for example 127.0.0.1:8080")
	     .takes_value(true))
	.arg(clap::Arg::with_name("document root")
	     .short("d")
	     .long("document_root")
	     .value_name("DOCUMENT_ROOT")
	     .help("specify document root path")
	     .takes_value(true))
	.arg(clap::Arg::with_name("restful api server")
	     .short("r")
	     .long("restfulapiserver")
	     .value_name("RESTFULAPISERVER")
	     .help("specify restful api server address and port, for example 127.0.0.1:8081")
	     .takes_value(true))
	.get_matches();

    let bind_address = arg_matches.value_of("bind").unwrap_or("0.0.0.0:80");
    println!("bind address: {}", bind_address);

    let document_root = arg_matches.value_of("document root").unwrap_or("/var/www").to_string();
    println!("document_root: {}", document_root);

    let restfulapiserver_address =
	arg_matches.value_of("restful api server").unwrap_or("127.0.0.1:8081").to_string();
    println!("restfulapiserver_address: {}", restfulapiserver_address);

    let appcontext_ref = web::Data::new(AppContext{
	document_root: document_root.to_string().clone(),
	restfulapi_server_address: restfulapiserver_address.clone()
    });

    HttpServer::new(move || {
	actix_web::App::new()
	    .app_data(appcontext_ref.clone())
	    .service(web::resource("/").route(web::get().to(index)))
	    .service(web::resource("/index.html").route(web::get().to(index)))
	    .service(web::resource("/webasm_greet.html").route(web::get().to(webasm_greet)))
	    .service(actix_files::Files::new("/simple_wasm", format!("{}/simple_wasm", document_root)))
	    .service(web::resource("/webasm_canvas.html").route(web::get().to(webasm_canvas)))
	    .service(actix_files::Files::new("/wasm_canvas", format!("{}/wasm_canvas", document_root)))
	    .service(web::resource("/webasm_webgl_triangle.html").route(web::get().to(webasm_webgl_triangle)))
	    .service(actix_files::Files::new("/wasm_webgl_triangle", format!("{}/wasm_webgl_triangle", document_root)))
	    .service(web::resource("/greet/{name}").route(web::get().to(greet::greet)))
	    .default_service(web::route().to(|| HttpResponse::NotFound())) })
	.bind(bind_address)?
	.run()
	.await

}
