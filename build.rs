// build.rs — compiles proto/vector.proto into Rust gRPC bindings via tonic-build.
//
// To regenerate Python stubs run:
//   python -m grpc_tools.protoc -I src/proto \
//     --python_out=python/proto \
//     --grpc_python_out=python/proto \
//     src/proto/vector.proto

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["src/proto/vector.proto"], &["src/proto"])?;

    println!("cargo:rerun-if-changed=src/proto/vector.proto");
    println!("cargo:rerun-if-changed=src/proto");
    Ok(())
}
