// main.rs — standalone claw-vector-server binary entry point.
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let addr: std::net::SocketAddr = std::env::var("CLAW_GRPC_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;

    tracing::info!(%addr, "starting claw-vector gRPC server");
    claw_vector::grpc::server::serve(addr).await?;
    Ok(())
}
