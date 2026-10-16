fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/aries/agent/v1/agent.proto");
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()?;
        std::env::set_var("PROTOC", protoc);
    }
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .build_transport(false)
        .compile_protos(&["../proto/aries/agent/v1/agent.proto"], &["../proto"])?;
    Ok(())
}
