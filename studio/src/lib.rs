// tonic::Status is the error type of every gRPC call; boxing it buys nothing here.
#[allow(clippy::result_large_err)]
pub mod proto {
    tonic::include_proto!("aries.agent.v1");
}

pub mod api;
pub mod auth;
pub mod cloud;
pub mod config;
pub mod db;
pub mod grpc;
pub mod legal;
pub mod metrics;
pub mod scheduler;
pub mod scraper;
pub mod storage;
