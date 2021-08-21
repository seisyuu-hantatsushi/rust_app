
use actix_web::HttpRequest;
use actix_web::HttpResponse;
use actix_web::error::InternalError;
use actix_web::http::StatusCode;
use sailfish::TemplateOnce;

#[derive(TemplateOnce)]
#[template(path = "greet.stpl")]
struct Greet<'a> {
    name: &'a str,
    messages: Vec<String>
}

pub async fn greet(req: HttpRequest) -> actix_web::Result<HttpResponse>{
    let name = req.match_info().get("name").unwrap_or("World");
    let messages = vec![String::from("Message 1"), String::from("<Message 2>")];
    let body = Greet{ name:name, messages:messages }.render_once()
	.map_err(|e| InternalError::new(e, StatusCode::INTERNAL_SERVER_ERROR))?;

    Ok(HttpResponse::Ok()
       .content_type("text/html; charset=utf-8")
       .body(body))
}
