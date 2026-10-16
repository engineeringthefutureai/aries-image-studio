pub mod proto {
    tonic::include_proto!("aries.agent.v1");
}

pub mod api;
pub mod config;
pub mod db;
pub mod grpc;
pub mod metrics;
pub mod scheduler;
pub mod scraper;
pub mod storage;
