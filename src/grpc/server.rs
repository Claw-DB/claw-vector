// grpc/server.rs — tonic gRPC server implementing the EmbeddingService stub.
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::grpc::proto::{
    embedding_service_server::{EmbeddingService, EmbeddingServiceServer},
    EmbedRequest, EmbedResponse, HealthRequest, HealthResponse, ModelInfoRequest,
    ModelInfoResponse,
};

/// Minimal pass-through gRPC servicer (real inference lives in Python).
pub struct EmbeddingServiceImpl;

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    /// Embed a batch of texts — this stub returns Unimplemented.
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        Err(Status::unimplemented(
            "Embed is handled by the Python embedding service",
        ))
    }

    /// Health check — always reports not ready from the Rust stub.
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            ready: false,
            model_name: String::new(),
            model_load_time_ms: 0,
        }))
    }

    /// Return model info — this stub returns Unimplemented.
    async fn model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Err(Status::unimplemented(
            "ModelInfo is handled by the Python embedding service",
        ))
    }

    type EmbedStreamStream = ReceiverStream<Result<EmbedResponse, Status>>;

    /// Streaming embed — this stub returns Unimplemented.
    async fn embed_stream(
        &self,
        _request: Request<tonic::Streaming<EmbedRequest>>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        Err(Status::unimplemented(
            "EmbedStream is handled by the Python embedding service",
        ))
    }
}

/// Start the standalone gRPC server and block until termination.
pub async fn serve(addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let svc = EmbeddingServiceServer::new(EmbeddingServiceImpl);
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr)
        .await?;
    Ok(())
}
