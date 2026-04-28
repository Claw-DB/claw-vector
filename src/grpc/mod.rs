// grpc/mod.rs — public re-exports for the gRPC sub-module plus generated proto bindings.
/// Standalone gRPC server implementation.
pub mod server;

/// Auto-generated tonic/prost code compiled from `src/proto/vector.proto`.
pub mod proto {
    #![allow(missing_docs)] // generated code does not carry doc comments
    tonic::include_proto!("clawvector.v1");
}
