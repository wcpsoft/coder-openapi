use coder_openapi::utils::init;
use coder_openapi::utils::run;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize application
    init().await?;

    // Run server
    run().await?;

    Ok(())
}
