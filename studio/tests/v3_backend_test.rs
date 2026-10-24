use aries_image_studio::api::build_http_router;
use aries_image_studio::cloud::{Api, CloudConfig, ProviderConfig};
use aries_image_studio::config::BackendConfig;
use aries_image_studio::db::{
    now_s, AttemptFailure, Database, ExecutorState, JobStatus, TaskStatus,
};
use aries_image_studio::grpc::AgentGatewayService;
use aries_image_studio::metrics::{AgentLiveMetrics, ExecutorLiveMetrics};
use aries_image_studio::proto;
use aries_image_studio::proto::agent_gateway_client::AgentGatewayClient;
use aries_image_studio::proto::agent_gateway_server::AgentGatewayServer;
use aries_image_studio::scheduler::{run_scheduler_tick, AppState};
use aries_image_studio::storage::{read_png_text_chunk, BlobStore};
use axum::http::HeaderMap;
use image::{ImageBuffer, ImageFormat, Rgba};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tonic::metadata::MetadataValue;
use tonic::transport::Server as GrpcServer;
use tonic::Request;

struct TestHarness {
    pub state: Arc<AppState>,
    pub http_url: String,
    pub grpc_url: String,
    pub token: String,
    pub temp_dir: std::path::PathBuf,
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.temp_dir);
    }
}

async fn start_test_harness() -> TestHarness {
    start_test_harness_with(Default::default()).await
}

async fn start_test_harness_with(cloud: CloudConfig) -> TestHarness {
    let temp_dir = std::env::temp_dir().join(format!("aries_v3_test_{}", ulid::Ulid::new()));
    std::fs::create_dir_all(temp_dir.join("static")).unwrap();
    std::fs::write(
        temp_dir.join("static/index.html"),
        "<!doctype html><title>app</title>",
    )
    .unwrap();

    let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr: SocketAddr = http_listener.local_addr().unwrap();
    let grpc_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_addr: SocketAddr = grpc_listener.local_addr().unwrap();

    let token = "test-secret-token".to_string();
    let config = BackendConfig {
        http_addr: http_addr.to_string(),
        grpc_addr: grpc_addr.to_string(),
        data_dir: temp_dir.clone(),
        static_dir: temp_dir.join("static"),
        agent_token: token.clone(),
        artifact_base_url: format!("http://{http_addr}"),
        max_wait_s: 600.0,
        idle_grace_s: 0.05,
        scrape_interval_ms: 1000,
        scrape_stale_s: 5.0,
        reconnect_grace_s: 0.2,
        disk_watermark_pct: 80.0,
        reference_res: 1024,
        cloud,
    };

    let db = Database::connect(temp_dir.join("studio.db")).await.unwrap();
    let blobs = BlobStore::new(&temp_dir).unwrap();
    let state = AppState::new(config, db, blobs);

    let router = build_http_router(state.clone());
    tokio::spawn(async move {
        let _ = axum::serve(http_listener, router).await;
    });

    let grpc_svc = AgentGatewayServer::new(AgentGatewayService::new(state.clone()));
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(grpc_listener);
    tokio::spawn(async move {
        let _ = GrpcServer::builder()
            .add_service(grpc_svc)
            .serve_with_incoming(incoming)
            .await;
    });

    TestHarness {
        state,
        http_url: format!("http://{http_addr}"),
        grpc_url: format!("http://{grpc_addr}"),
        token,
        temp_dir,
    }
}

fn sample_register_msg(agent_id: &str) -> proto::Register {
    proto::Register {
        protocol: 1,
        agent: agent_id.to_string(),
        agent_version: "0.3.0".to_string(),
        backend_kind: "xpu".to_string(),
        metrics_url: String::new(),
        host_ram_total_bytes: 32 * 1024 * 1024 * 1024,
        models: vec![
            proto::ModelRef {
                id: "Qwen/Qwen-Image-2.1".to_string(),
                rev: "rev123".to_string(),
                dtype: "bf16".to_string(),
            },
            proto::ModelRef {
                id: "Qwen/Qwen-Image-2.1-PE-T2I".to_string(),
                rev: "revpe456".to_string(),
                dtype: "bf16".to_string(),
            },
        ],
        executors: vec![
            proto::Executor {
                id: "xpu0".to_string(),
                device: "0000:04:00.0".to_string(),
                mem_total_bytes: 24 * 1024 * 1024 * 1024,
                stages: vec![
                    proto::StageCapability {
                        stage: proto::Stage::Enhance as i32,
                        models: vec!["Qwen/Qwen-Image-2.1-PE-T2I".to_string()],
                        max_px: 0,
                        modes: vec![],
                        max_references: 0,
                        reference_models: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Encode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 0,
                        modes: vec![],
                        max_references: 4,
                        reference_models: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Denoise as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![],
                        max_references: 4,
                        reference_models: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Decode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![
                            proto::DecodeMode::Untiled as i32,
                            proto::DecodeMode::Tiled as i32,
                        ],
                        max_references: 0,
                        reference_models: vec![],
                    },
                ],
            },
            proto::Executor {
                id: "xpu1".to_string(),
                device: "0000:08:00.0".to_string(),
                mem_total_bytes: 24 * 1024 * 1024 * 1024,
                stages: vec![
                    proto::StageCapability {
                        stage: proto::Stage::Encode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 0,
                        modes: vec![],
                        max_references: 4,
                        reference_models: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Denoise as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![],
                        max_references: 4,
                        reference_models: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Decode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![
                            proto::DecodeMode::Untiled as i32,
                            proto::DecodeMode::Tiled as i32,
                        ],
                        max_references: 0,
                        reference_models: vec![],
                    },
                ],
            },
        ],
        limits: vec![proto::Limit {
            stage: proto::Stage::Decode as i32,
            above_px: 1024 * 1024,
            above_references: None,
            decode_mode: None,
            requires: vec![
                proto::Requirement {
                    kind: Some(proto::requirement::Kind::HostRamAvailableBytes(
                        6 * 1024 * 1024 * 1024,
                    )),
                },
                proto::Requirement {
                    kind: Some(proto::requirement::Kind::MaxConcurrentPerAgent(1)),
                },
            ],
        }],
        running: vec![],
    }
}

fn build_safetensors_bytes(
    tensors: &[(&str, &str, Vec<usize>, &[u8])],
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let mut header_map = serde_json::Map::new();
    let mut meta_map = serde_json::Map::new();
    for (k, v) in metadata {
        meta_map.insert((*k).to_string(), Value::String((*v).to_string()));
    }
    header_map.insert("__metadata__".to_string(), Value::Object(meta_map));

    let mut data_buf = Vec::new();
    for (name, dtype, shape, raw) in tensors {
        let start = data_buf.len();
        data_buf.extend_from_slice(raw);
        let end = data_buf.len();
        header_map.insert(
            (*name).to_string(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_bytes = serde_json::to_vec(&Value::Object(header_map)).unwrap();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data_buf);
    out
}

fn build_png_bytes(width: u32, height: u32) -> Vec<u8> {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(width, height, Rgba([42, 100, 200, 255]));
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png).unwrap();
    buf.into_inner()
}

#[tokio::test]
async fn test_unknown_requirement_kind_rejected() {
    let harness = start_test_harness().await;
    let channel = tonic::transport::Endpoint::from_shared(harness.grpc_url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = AgentGatewayClient::new(channel);

    let (tx, rx) = mpsc::unbounded_channel();
    let mut reg = sample_register_msg("aries");
    // Simulate an unknown requirement kind from a newer agent (kind = None)
    reg.limits.push(proto::Limit {
        stage: proto::Stage::Decode as i32,
        above_px: 0,
        above_references: None,
        decode_mode: None,
        requires: vec![proto::Requirement { kind: None }],
    });
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(reg)),
    })
    .unwrap();

    let mut req = Request::new(UnboundedReceiverStream::new(rx));
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );

    let res = client.connect(req).await;
    assert!(
        res.is_err(),
        "Registration with unknown Requirement.kind must fail"
    );
    let status = res.err().unwrap();
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
}

#[tokio::test]
async fn test_submit_feasibility_and_e2e_pipeline() {
    let harness = start_test_harness().await;
    let http = reqwest::Client::new();

    // 1. Before any agent registers, submitting a job is rejected (422)
    let resp = http
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&json!({
            "prompt": "a mountain lake at dawn",
            "width": 1024,
            "height": 1024,
            "steps": 10,
            "enhance": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 422);

    // 2. Connect agent and register
    let channel = tonic::transport::Endpoint::from_shared(harness.grpc_url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut grpc_client = AgentGatewayClient::new(channel);
    let (tx, rx) = mpsc::unbounded_channel();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(sample_register_msg(
            "aries",
        ))),
    })
    .unwrap();

    let mut req = Request::new(UnboundedReceiverStream::new(rx));
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );
    let mut inbound = grpc_client.connect(req).await.unwrap().into_inner();

    // Expect Welcome
    let welcome = inbound.next().await.unwrap().unwrap();
    assert!(matches!(
        welcome.msg,
        Some(proto::backend_message::Msg::Welcome(_))
    ));

    // Populate live metrics so >1MP limits also have fresh metrics if needed
    {
        let mut lm = harness.state.live_metrics.write().unwrap();
        let mut exec_map = HashMap::new();
        exec_map.insert(
            "xpu0".to_string(),
            ExecutorLiveMetrics {
                mem_used_bytes: 0,
                mem_total_bytes: 24 * 1024 * 1024 * 1024,
                mem_available_bytes: 24 * 1024 * 1024 * 1024,
                overflow_host_bytes: 0,
            },
        );
        lm.insert(
            "aries".to_string(),
            AgentLiveMetrics {
                host_ram_available_bytes: 16 * 1024 * 1024 * 1024,
                executors: exec_map,
                scraped_at: now_s(),
                raw: String::new(),
            },
        );
    }

    // 3. Submitting 4096x4096 must be rejected at submit with 422
    let resp_4096 = http
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&json!({
            "prompt": "too big",
            "width": 4096,
            "height": 4096,
            "steps": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp_4096.status(), 422);
    let body_4096: Value = resp_4096.json().await.unwrap();
    assert!(body_4096["rejected"].as_str().unwrap().contains("4096"));

    // 4. Submit a valid 1024x1024 job with enhance=true
    let resp_ok = http
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&json!({
            "prompt": "a serene mountain lake",
            "negative_prompt": "blurry",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 12345,
            "true_cfg_scale": 4.0,
            "enhance": true,
            "tiling": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp_ok.status(), 201);
    let created_job: Value = resp_ok.json().await.unwrap();
    let job_id = created_job["id"].as_str().unwrap().to_string();
    assert_eq!(created_job["tiling"], true);

    // Drive all 4 stages: enhance -> encode -> denoise -> decode
    for expected_stage in [
        proto::Stage::Enhance,
        proto::Stage::Encode,
        proto::Stage::Denoise,
        proto::Stage::Decode,
    ] {
        // Wait for StartWorker (skipping any Release messages from idle_grace)
        let start_worker = loop {
            let msg = inbound.next().await.unwrap().unwrap();
            match msg.msg {
                Some(proto::backend_message::Msg::StartWorker(sw)) => break sw,
                Some(proto::backend_message::Msg::Release(rel)) => {
                    tx.send(proto::AgentMessage {
                        msg: Some(proto::agent_message::Msg::ExecutorState(
                            proto::ExecutorState {
                                executor: rel.executor,
                                state: proto::ExecutorStateKind::ExecutorStateIdle as i32,
                                stage: proto::Stage::Unspecified as i32,
                                pid: 0,
                                load_s: 0.0,
                                detail: String::new(),
                            },
                        )),
                    })
                    .unwrap();
                }
                other => panic!("Unexpected message while waiting for StartWorker: {other:?}"),
            }
        };
        assert_eq!(
            proto::Stage::try_from(start_worker.stage).unwrap(),
            expected_stage
        );

        // Agent reports READY
        tx.send(proto::AgentMessage {
            msg: Some(proto::agent_message::Msg::ExecutorState(
                proto::ExecutorState {
                    executor: start_worker.executor.clone(),
                    state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                    stage: start_worker.stage,
                    pid: 4242,
                    load_s: 1.25,
                    detail: String::new(),
                },
            )),
        })
        .unwrap();

        // Expect Assign
        let assign = loop {
            let msg = inbound.next().await.unwrap().unwrap();
            match msg.msg {
                Some(proto::backend_message::Msg::Assign(a)) => break a,
                Some(proto::backend_message::Msg::Release(rel)) => {
                    tx.send(proto::AgentMessage {
                        msg: Some(proto::agent_message::Msg::ExecutorState(
                            proto::ExecutorState {
                                executor: rel.executor,
                                state: proto::ExecutorStateKind::ExecutorStateIdle as i32,
                                stage: proto::Stage::Unspecified as i32,
                                pid: 0,
                                load_s: 0.0,
                                detail: String::new(),
                            },
                        )),
                    })
                    .unwrap();
                }
                other => panic!("Unexpected message while waiting for Assign: {other:?}"),
            }
        };
        assert_eq!(
            proto::Stage::try_from(assign.stage).unwrap(),
            expected_stage
        );

        // Verify inputs can be downloaded when present
        for inp in &assign.inputs {
            let dl = http
                .get(&inp.url)
                .header("Authorization", format!("Bearer {}", harness.token))
                .send()
                .await
                .unwrap();
            assert_eq!(dl.status(), 200);
            let dl_bytes = dl.bytes().await.unwrap();
            let mut h = Sha256::new();
            h.update(&dl_bytes);
            assert_eq!(hex::encode(h.finalize()), inp.sha256);
        }

        // Prepare artifact bytes for this stage
        let upload_ref = &assign.uploads[0];
        let payload: Vec<u8> = match expected_stage {
            proto::Stage::Enhance => serde_json::to_vec(&json!({
                "rewritten_prompt": "a serene alpine mountain lake at golden dawn, crystal clear reflections",
                "wh_ratio": "1:1"
            }))
            .unwrap(),
            proto::Stage::Encode => {
                // Confirm backend resolved the enhanced prompt!
                let Some(proto::assign::Params::Encode(enc_p)) = &assign.params else {
                    panic!("expected EncodeParams");
                };
                assert_eq!(enc_p.prompt_source, "enhanced");
                assert!(enc_p.prompt.contains("alpine mountain lake"));
                build_safetensors_bytes(
                    &[
                        ("prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2048]),
                        ("prompt_embeds_mask", "I64", vec![1, 16], &[1u8; 128]),
                        ("negative_prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2048]),
                        ("negative_prompt_embeds_mask", "I64", vec![1, 16], &[1u8; 128]),
                    ],
                    &[
                        ("prompt", &enc_p.prompt),
                        ("prompt_source", &enc_p.prompt_source),
                        ("model", "Qwen/Qwen-Image-2.1"),
                        ("rev", "rev123"),
                        ("dtype", "bf16"),
                    ],
                )
            }
            proto::Stage::Denoise => {
                tx.send(proto::AgentMessage {
                    msg: Some(proto::agent_message::Msg::Progress(proto::Progress {
                        task: assign.task.clone(),
                        attempt: assign.attempt,
                        step: 2,
                        steps: 4,
                        s_per_step: 0.5,
                        ..Default::default()
                    })),
                })
                .unwrap();
                build_safetensors_bytes(
                    &[("latent", "BF16", vec![1, 16, 128, 128], &[0u8; 512])],
                    &[
                        ("model", "Qwen/Qwen-Image-2.1"),
                        ("rev", "rev123"),
                        ("dtype", "bf16"),
                        ("width", "1024"),
                        ("height", "1024"),
                        ("seed", "12345"),
                        ("steps", "4"),
                    ],
                )
            }
            proto::Stage::Decode => {
                // The job asked for tiling, so the decode assignment says tiled.
                let Some(proto::assign::Params::Decode(dec_p)) = &assign.params else {
                    panic!("expected DecodeParams");
                };
                assert_eq!(dec_p.mode, proto::DecodeMode::Tiled as i32);
                build_png_bytes(1024, 1024)
            }
            proto::Stage::Unspecified => unreachable!(),
        };

        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let sha = hex::encode(hasher.finalize());

        let up_resp = http
            .put(&upload_ref.url)
            .header("Authorization", format!("Bearer {}", harness.token))
            .header("x-content-sha256", &sha)
            .body(payload.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(up_resp.status(), 201);

        let model_ref = start_worker.model.unwrap();
        let mut timings = HashMap::new();
        timings.insert("stage_s".to_string(), 1.2);

        tx.send(proto::AgentMessage {
            msg: Some(proto::agent_message::Msg::Done(proto::Done {
                task: assign.task.clone(),
                attempt: assign.attempt,
                artifacts: vec![proto::ArtifactRef {
                    name: upload_ref.name.clone(),
                    url: upload_ref.url.clone(),
                    sha256: sha,
                    bytes: payload.len() as u64,
                }],
                timings,
                peak_mem_bytes: 12 * 1024 * 1024 * 1024,
                provenance: Some(proto::Provenance {
                    agent: "aries".to_string(),
                    executor: assign.executor.clone(),
                    device: "0000:04:00.0".to_string(),
                    backend_kind: "xpu".to_string(),
                    model: Some(model_ref),
                    torch_version: "2.6.0+xpu".to_string(),
                    worker_version: "0.3.0".to_string(),
                }),
            })),
        })
        .unwrap();
    }

    // Wait briefly for Done processing & PNG metadata finalization
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let job_resp: Value = http
        .get(format!("{}/api/jobs/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(job_resp["status"], "done");
    assert!(job_resp["enhanced_prompt"]
        .as_str()
        .unwrap()
        .contains("alpine mountain lake"));
    assert!(job_resp["provenance"]["encode"].is_object());
    assert!(job_resp["provenance"]["denoise"].is_object());
    assert!(job_resp["provenance"]["decode"].is_object());

    // Fetch final image and verify embedded PNG tEXt chunk "aries-image-studio"
    let cache_control = |r: &reqwest::Response| {
        r.headers()
            .get("cache-control")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
    };
    let img_resp = http
        .get(format!("{}/api/images/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap();
    assert!(cache_control(&img_resp).contains("immutable"));
    let img_bytes = img_resp.bytes().await.unwrap();
    let chunk_text = read_png_text_chunk(&img_bytes, "aries-image-studio")
        .unwrap()
        .expect("PNG must contain aries-image-studio tEXt chunk");
    let chunk_json: Value = serde_json::from_str(&chunk_text).unwrap();
    assert_eq!(chunk_json["job_id"], job_id);
    assert_eq!(
        chunk_json["provenance"]["decode"]["torch_version"],
        "2.6.0+xpu"
    );
    // Before the provenance chunk is in (the job is already done), the decoder's PNG is
    // served but not cached.
    sqlx::query("UPDATE artifacts SET metadata_json = NULL WHERE name = 'image.png'")
        .execute(harness.state.db.pool())
        .await
        .unwrap();
    let raw = http
        .get(format!("{}/api/images/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap();
    assert_eq!(cache_control(&raw), "no-store");

    // Thumbnails: JPEG at both sizes (written in the background, or on request), cacheable
    for (query, size) in [
        ("", 1024u32),
        ("?size=1024", 1024),
        ("?size=768", 768),
        ("?size=512", 512),
    ] {
        let thumb_resp = http
            .get(format!(
                "{}/api/images/{job_id}/thumb{query}",
                harness.http_url
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(thumb_resp.status(), 200);
        assert_eq!(
            thumb_resp.headers().get("content-type").unwrap(),
            "image/jpeg"
        );
        assert!(thumb_resp
            .headers()
            .get("cache-control")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("immutable"));
        let thumb = image::load_from_memory(&thumb_resp.bytes().await.unwrap()).unwrap();
        // The decoded test image is 1024², so 1024 keeps it and the others shrink it.
        assert_eq!((thumb.width(), thumb.height()), (size, size));
    }
    let bad = http
        .get(format!(
            "{}/api/images/{job_id}/thumb?size=300",
            harness.http_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    // Retrying a finished job is refused, and the job stays done with its image.
    let retry = http
        .post(format!("{}/api/jobs/{job_id}/retry", harness.http_url))
        .send()
        .await
        .unwrap();
    assert_eq!(retry.status(), 409);
    assert_eq!(
        harness
            .state
            .db
            .get_job(&job_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        JobStatus::Done
    );

    // Verify intermediate artifacts were garbage collected while final artifact remains
    let (inter_bytes, final_bytes) = harness.state.db.artifact_bytes_by_kind().await.unwrap();
    assert_eq!(
        inter_bytes, 0,
        "Intermediate artifacts must be deleted on job completion"
    );
    assert!(final_bytes > 0, "Final PNG artifact must be retained");
}

#[tokio::test]
async fn test_live_limits_stale_scrape_and_retry_on_crash() {
    let harness = start_test_harness().await;
    let http = reqwest::Client::new();

    // Register agent
    let channel = tonic::transport::Endpoint::from_shared(harness.grpc_url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut grpc_client = AgentGatewayClient::new(channel);
    let (tx, rx) = mpsc::unbounded_channel();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(sample_register_msg(
            "aries",
        ))),
    })
    .unwrap();
    let mut req = Request::new(UnboundedReceiverStream::new(rx));
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );
    let mut inbound = grpc_client.connect(req).await.unwrap().into_inner();
    let _welcome = inbound.next().await.unwrap().unwrap();

    // Set host RAM available to ONLY 2 GiB (< 6 GiB limit for 2048x2048 decode)
    {
        let mut lm = harness.state.live_metrics.write().unwrap();
        lm.insert(
            "aries".to_string(),
            AgentLiveMetrics {
                host_ram_available_bytes: 2 * 1024 * 1024 * 1024,
                executors: HashMap::new(),
                scraped_at: now_s(),
                raw: String::new(),
            },
        );
    }

    // Submit 2048x2048 job (encode & denoise have no host RAM limit, only decode > 1MP does)
    let resp = http
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&json!({
            "prompt": "high res test",
            "width": 2048,
            "height": 2048,
            "steps": 2,
            "seed": 99,
            "enhance": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let job: Value = resp.json().await.unwrap();
    let job_id = job["id"].as_str().unwrap().to_string();

    // 1. Encode starts on xpu0; simulate a worker crash (attempt 1 fails with oom_kill)
    let sw_encode_1 = match inbound.next().await.unwrap().unwrap().msg.unwrap() {
        proto::backend_message::Msg::StartWorker(sw) => sw,
        other => panic!("expected StartWorker, got {other:?}"),
    };
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::ExecutorState(
            proto::ExecutorState {
                executor: sw_encode_1.executor.clone(),
                state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                stage: sw_encode_1.stage,
                pid: 1001,
                load_s: 0.5,
                detail: String::new(),
            },
        )),
    })
    .unwrap();

    let assign_encode_1 = match inbound.next().await.unwrap().unwrap().msg.unwrap() {
        proto::backend_message::Msg::Assign(a) => a,
        other => panic!("expected Assign, got {other:?}"),
    };
    assert_eq!(assign_encode_1.attempt, 1);

    // Report Failed (oom_kill) on attempt 1 and executor returning to IDLE
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Failed(proto::Failed {
            task: assign_encode_1.task.clone(),
            attempt: 1,
            failure_class: proto::FailureClass::FailureOomKill as i32,
            detail: "SIGKILL oom".to_string(),
            exit_code: -9,
            signal: 9,
            kernel_log_tail: "Out of memory: Killed process 1001".to_string(),
        })),
    })
    .unwrap();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::ExecutorState(
            proto::ExecutorState {
                executor: sw_encode_1.executor.clone(),
                state: proto::ExecutorStateKind::ExecutorStateIdle as i32,
                stage: proto::Stage::Unspecified as i32,
                pid: 0,
                load_s: 0.0,
                detail: String::new(),
            },
        )),
    })
    .unwrap();

    // Backend must automatically retry encode as attempt 2!
    let sw_encode_2 = loop {
        let msg = inbound.next().await.unwrap().unwrap();
        if let Some(proto::backend_message::Msg::StartWorker(sw)) = msg.msg {
            break sw;
        }
    };
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::ExecutorState(
            proto::ExecutorState {
                executor: sw_encode_2.executor.clone(),
                state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                stage: sw_encode_2.stage,
                pid: 1002,
                load_s: 0.5,
                detail: String::new(),
            },
        )),
    })
    .unwrap();

    let assign_encode_2 = loop {
        let msg = inbound.next().await.unwrap().unwrap();
        if let Some(proto::backend_message::Msg::Assign(a)) = msg.msg {
            break a;
        }
    };
    assert_eq!(assign_encode_2.attempt, 2);

    // Verify attempt fencing: uploading to attempt 1 URL must now return 409 Conflict!
    let stale_url = assign_encode_1.uploads[0].url.clone();
    let stale_resp = http
        .put(&stale_url)
        .header("Authorization", format!("Bearer {}", harness.token))
        .body(vec![1, 2, 3])
        .send()
        .await
        .unwrap();
    assert_eq!(stale_resp.status(), 409);

    // Complete encode attempt 2 and denoise attempt 1 so job reaches decode (2048x2048)
    let embeds_bytes = build_safetensors_bytes(
        &[
            ("prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2048]),
            ("prompt_embeds_mask", "I64", vec![1, 16], &[1u8; 128]),
            (
                "negative_prompt_embeds",
                "BF16",
                vec![1, 16, 64],
                &[0u8; 2048],
            ),
            (
                "negative_prompt_embeds_mask",
                "I64",
                vec![1, 16],
                &[1u8; 128],
            ),
        ],
        &[
            ("prompt", "high res test"),
            ("prompt_source", "original"),
            ("model", "Qwen/Qwen-Image-2.1"),
            ("rev", "rev123"),
            ("dtype", "bf16"),
        ],
    );
    let mut h = Sha256::new();
    h.update(&embeds_bytes);
    let embeds_sha = hex::encode(h.finalize());
    http.put(&assign_encode_2.uploads[0].url)
        .header("Authorization", format!("Bearer {}", harness.token))
        .header("x-content-sha256", &embeds_sha)
        .body(embeds_bytes.clone())
        .send()
        .await
        .unwrap();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Done(proto::Done {
            task: assign_encode_2.task.clone(),
            attempt: 2,
            artifacts: vec![proto::ArtifactRef {
                name: "embeds.safetensors".to_string(),
                url: assign_encode_2.uploads[0].url.clone(),
                sha256: embeds_sha,
                bytes: embeds_bytes.len() as u64,
            }],
            timings: HashMap::new(),
            peak_mem_bytes: 1024,
            provenance: Some(proto::Provenance {
                agent: "aries".to_string(),
                executor: assign_encode_2.executor.clone(),
                device: "0000:04:00.0".to_string(),
                backend_kind: "xpu".to_string(),
                model: sw_encode_2.model.clone(),
                torch_version: "2.6.0+xpu".to_string(),
                worker_version: "0.3.0".to_string(),
            }),
        })),
    })
    .unwrap();

    // Denoise stage
    let sw_denoise = loop {
        let msg = inbound.next().await.unwrap().unwrap();
        if let Some(proto::backend_message::Msg::StartWorker(sw)) = msg.msg {
            break sw;
        }
    };
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::ExecutorState(
            proto::ExecutorState {
                executor: sw_denoise.executor.clone(),
                state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                stage: sw_denoise.stage,
                pid: 1003,
                load_s: 0.5,
                detail: String::new(),
            },
        )),
    })
    .unwrap();
    let assign_denoise = loop {
        let msg = inbound.next().await.unwrap().unwrap();
        if let Some(proto::backend_message::Msg::Assign(a)) = msg.msg {
            break a;
        }
    };
    let latent_bytes = build_safetensors_bytes(
        &[("latent", "BF16", vec![1, 16, 256, 256], &[0u8; 512])],
        &[
            ("model", "Qwen/Qwen-Image-2.1"),
            ("rev", "rev123"),
            ("dtype", "bf16"),
            ("width", "2048"),
            ("height", "2048"),
            ("seed", "99"),
            ("steps", "2"),
        ],
    );
    let mut h2 = Sha256::new();
    h2.update(&latent_bytes);
    let latent_sha = hex::encode(h2.finalize());
    http.put(&assign_denoise.uploads[0].url)
        .header("Authorization", format!("Bearer {}", harness.token))
        .header("x-content-sha256", &latent_sha)
        .body(latent_bytes.clone())
        .send()
        .await
        .unwrap();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Done(proto::Done {
            task: assign_denoise.task.clone(),
            attempt: 1,
            artifacts: vec![proto::ArtifactRef {
                name: "latent.safetensors".to_string(),
                url: assign_denoise.uploads[0].url.clone(),
                sha256: latent_sha,
                bytes: latent_bytes.len() as u64,
            }],
            timings: HashMap::new(),
            peak_mem_bytes: 1024,
            provenance: Some(proto::Provenance {
                agent: "aries".to_string(),
                executor: assign_denoise.executor.clone(),
                device: "0000:08:00.0".to_string(),
                backend_kind: "xpu".to_string(),
                model: sw_denoise.model.clone(),
                torch_version: "2.6.0+xpu".to_string(),
                worker_version: "0.3.0".to_string(),
            }),
        })),
    })
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    // Now the job is at decode (2048x2048), which requires 6 GiB host RAM, but only 2 GiB is available!
    let blocked_job: Value = http
        .get(format!("{}/api/jobs/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let blocked_reason = blocked_job["blocked_reason"].as_str().unwrap_or("");
    assert!(
        blocked_reason.contains("waiting for host RAM"),
        "Expected 'waiting for host RAM', got '{blocked_reason}'"
    );

    // Now simulate stale scrape (> 5s old) even with 16 GiB RAM
    {
        let mut lm = harness.state.live_metrics.write().unwrap();
        lm.insert(
            "aries".to_string(),
            AgentLiveMetrics {
                host_ram_available_bytes: 16 * 1024 * 1024 * 1024,
                executors: HashMap::new(),
                scraped_at: now_s() - 10.0,
                raw: String::new(),
            },
        );
    }
    run_scheduler_tick(&harness.state).await.unwrap();

    let stale_job: Value = http
        .get(format!("{}/api/jobs/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let stale_reason = stale_job["blocked_reason"].as_str().unwrap_or("");
    assert!(
        stale_reason.contains("stale"),
        "Expected stale scrape blocked reason, got '{stale_reason}'"
    );
}

#[tokio::test]
async fn test_lending_disabled_persistence_and_reconnect_reclaim() {
    let harness = start_test_harness().await;
    let http = reqwest::Client::new();

    // Register agent "aries"
    let channel = tonic::transport::Endpoint::from_shared(harness.grpc_url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut grpc_client = AgentGatewayClient::new(channel.clone());
    let (tx, rx) = mpsc::unbounded_channel();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(sample_register_msg(
            "aries",
        ))),
    })
    .unwrap();
    let mut req = Request::new(UnboundedReceiverStream::new(rx));
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );
    let mut inbound = grpc_client.connect(req).await.unwrap().into_inner();
    let _welcome = inbound.next().await.unwrap().unwrap();

    // 1. Lend executor xpu1 via operator API
    let lend_resp: Value = http
        .post(format!(
            "{}/api/executors/aries:xpu1/lend",
            harness.http_url
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(lend_resp["state"], "lent");

    let return_resp: Value = http
        .post(format!(
            "{}/api/executors/aries:xpu1/return",
            harness.http_url
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(return_resp["state"], "idle");

    // 2. Submit a job, start encode on xpu0, and report FailureWedged -> xpu0 becomes disabled
    let job_resp: Value = http
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&json!({
            "prompt": "test wedged",
            "width": 1024,
            "height": 1024,
            "steps": 2,
            "seed": 11,
            "enhance": false
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let job_id = job_resp["id"].as_str().unwrap().to_string();

    let sw = match inbound.next().await.unwrap().unwrap().msg.unwrap() {
        proto::backend_message::Msg::StartWorker(sw) => sw,
        other => panic!("expected StartWorker, got {other:?}"),
    };
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::ExecutorState(
            proto::ExecutorState {
                executor: sw.executor.clone(),
                state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                stage: sw.stage,
                pid: 2001,
                load_s: 0.2,
                detail: String::new(),
            },
        )),
    })
    .unwrap();

    let assign = match inbound.next().await.unwrap().unwrap().msg.unwrap() {
        proto::backend_message::Msg::Assign(a) => a,
        other => panic!("expected Assign, got {other:?}"),
    };
    let wedged_ex_key = format!("aries:{}", assign.executor);

    // Report FailureWedged on this executor
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Failed(proto::Failed {
            task: assign.task.clone(),
            attempt: assign.attempt,
            failure_class: proto::FailureClass::FailureWedged as i32,
            detail: "worker did not exit after SIGKILL".to_string(),
            exit_code: -9,
            signal: 9,
            kernel_log_tail: String::new(),
        })),
    })
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    // Check that the wedged executor is now disabled in /api/system
    let sys1: Value = http
        .get(format!("{}/api/system", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ex_list = sys1["agents"][0]["executors"].as_array().unwrap();
    let wedged_ex = ex_list
        .iter()
        .find(|e| e["key"].as_str() == Some(&wedged_ex_key))
        .unwrap();
    assert_eq!(wedged_ex["state"], "disabled");

    // 3. Reconnect agent "aries" and verify `disabled` persists across reconnects!
    drop(tx);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let (tx2, rx2) = mpsc::unbounded_channel();
    tx2.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(sample_register_msg(
            "aries",
        ))),
    })
    .unwrap();
    let mut req2 = Request::new(UnboundedReceiverStream::new(rx2));
    req2.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );
    let mut inbound2 = grpc_client.connect(req2).await.unwrap().into_inner();
    let _welcome2 = inbound2.next().await.unwrap().unwrap();

    let sys2: Value = http
        .get(format!("{}/api/system", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ex_list2 = sys2["agents"][0]["executors"].as_array().unwrap();
    let wedged_ex2 = ex_list2
        .iter()
        .find(|e| e["key"].as_str() == Some(&wedged_ex_key))
        .unwrap();
    assert_eq!(
        wedged_ex2["state"], "disabled",
        "disabled state must persist across agent reconnects"
    );

    // Enable the disabled executor via operator API
    let enable_resp: Value = http
        .post(format!(
            "{}/api/executors/{wedged_ex_key}/enable",
            harness.http_url
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(enable_resp["state"], "idle");

    // Clean up job
    let _ = http
        .post(format!("{}/api/jobs/{job_id}/cancel", harness.http_url))
        .send()
        .await;
}

// ---------- regression tests for the V3 review fixes ----------

type AgentLink = (
    mpsc::UnboundedSender<proto::AgentMessage>,
    tonic::Streaming<proto::BackendMessage>,
);

async fn connect_agent(harness: &TestHarness, reg: proto::Register) -> AgentLink {
    let channel = tonic::transport::Endpoint::from_shared(harness.grpc_url.clone())
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = AgentGatewayClient::new(channel);
    let (tx, rx) = mpsc::unbounded_channel();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Register(reg)),
    })
    .unwrap();
    let mut req = Request::new(UnboundedReceiverStream::new(rx));
    req.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {}", harness.token)).unwrap(),
    );
    let mut inbound = client.connect(req).await.unwrap().into_inner();
    let welcome = inbound.next().await.unwrap().unwrap();
    assert!(matches!(
        welcome.msg,
        Some(proto::backend_message::Msg::Welcome(_))
    ));
    (tx, inbound)
}

async fn submit(harness: &TestHarness, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{}/api/jobs", harness.http_url))
        .json(&body)
        .send()
        .await
        .unwrap()
}

/// Waits until `cond` holds for the task, polling the DB.
async fn wait_task(
    harness: &TestHarness,
    task_id: &str,
    cond: impl Fn(&aries_image_studio::db::TaskRecord) -> bool,
) -> aries_image_studio::db::TaskRecord {
    for _ in 0..200 {
        if let Some(t) = harness.state.db.get_task(task_id).await.unwrap() {
            if cond(&t) {
                return t;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("task {task_id} never reached the expected state");
}

async fn executor(harness: &TestHarness, key: &str) -> aries_image_studio::db::ExecutorRecord {
    harness.state.db.find_executor(key).await.unwrap().unwrap()
}

#[tokio::test]
async fn test_client_job_id_is_used_and_idempotent() {
    let harness = start_test_harness().await;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;

    let body = json!({ "id": "01JCLIENTULID0000000000000", "prompt": "a lake", "width": 1024, "height": 1024 });
    let first = submit(&harness, body.clone()).await;
    assert_eq!(first.status(), 201);
    let job: Value = first.json().await.unwrap();
    assert_eq!(job["id"], "01JCLIENTULID0000000000000");
    assert_eq!(job["true_cfg_scale"], 1.0, "default must be no CFG");

    let again = submit(&harness, body).await;
    assert_eq!(again.status(), 200);
    assert_eq!(harness.state.db.list_jobs(None).await.unwrap().len(), 1);

    let bad = submit(
        &harness,
        json!({ "id": "../etc", "prompt": "x", "width": 1024, "height": 1024 }),
    )
    .await;
    assert_eq!(bad.status(), 400);
}

#[tokio::test]
async fn test_all_composer_presets_are_feasible() {
    let harness = start_test_harness().await;
    let mut reg = sample_register_msg("aries");
    for ex in &mut reg.executors {
        for st in &mut ex.stages {
            if st.max_px > 0 {
                st.max_px = 2400 * 1792; // the agent's default
            }
        }
    }
    let _agent = connect_agent(&harness, reg).await;
    let full = [
        (2048, 2048),
        (2400, 1792),
        (1792, 2400),
        (2528, 1696),
        (1696, 2528),
        (2752, 1536),
        (1536, 2752),
    ];
    // ½ and ¼ presets, rounded like web/src/types.ts calculateDimensions.
    let scaled = |v: u32, f: f64| ((v as f64 * f / 32.0).round() as u32) * 32;
    for (fw, fh) in full {
        for f in [1.0, 0.5, 0.25] {
            let (w, h) = (scaled(fw, f), scaled(fh, f));
            let resp = submit(&harness, json!({ "prompt": "p", "width": w, "height": h })).await;
            assert_eq!(resp.status(), 201, "{w}x{h} rejected");
        }
    }
    // Qwen-Image-2.1 floors sizes to multiples of 32 (1264 came back 1248): refuse at submit.
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1264, "height": 848 }),
    )
    .await;
    assert_eq!(resp.status(), 422);
    // Steps: the presets (12, 20, 40) and any custom count from 1 to 100.
    for (steps, status) in [
        (1, 201),
        (12, 201),
        (20, 201),
        (40, 201),
        (100, 201),
        (0, 422),
        (101, 422),
    ] {
        let resp = submit(
            &harness,
            json!({ "prompt": "p", "width": 1024, "height": 1024, "steps": steps }),
        )
        .await;
        assert_eq!(resp.status(), status, "steps {steps}");
    }
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    let job: Value = resp.json().await.unwrap();
    assert_eq!(job["steps"], 20, "default is the Standard preset");
    let caps = aries_image_studio::scheduler::compute_capabilities(&harness.state.db)
        .await
        .unwrap();
    assert!(
        caps.sizes.iter().all(|s| s.servable_ever),
        "{:?}",
        caps.sizes
    );
}

/// The agent sends READY right after Done; by then the backend may have assigned the next
/// task. That READY must not reopen the executor for a second assignment.
#[tokio::test]
async fn test_ready_after_done_does_not_double_assign() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    for i in 0..3 {
        let resp = submit(
            &harness,
            json!({ "prompt": format!("p{i}"), "width": 1024, "height": 1024 }),
        )
        .await;
        assert_eq!(resp.status(), 201);
    }
    let encode_tasks: Vec<_> = db.list_queued_tasks().await.unwrap();
    assert_eq!(encode_tasks.len(), 3);

    // Pretend an encode worker on xpu0 is ready and the first task got assigned to it.
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("encode"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    let first = &encode_tasks[0];
    db.assign_task(&first.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        executor(&harness, "aries:xpu0").await.state,
        ExecutorState::Busy
    );

    // Late READY (the one that followed the previous Done) arrives: still busy.
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("encode"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    let ex = executor(&harness, "aries:xpu0").await;
    assert_eq!(ex.state, ExecutorState::Busy);
    assert_eq!(ex.current_task_id.as_deref(), Some(first.id.as_str()));
}

#[tokio::test]
async fn test_stale_rejection_is_ignored() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);

    db.assign_task(&task.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    db.fail_task_attempt(&task.id, 1, AttemptFailure::studio("python_error", "boom"))
        .await
        .unwrap();
    // A rejection for attempt 1 arrives after the task was requeued: ignore it.
    db.handle_rejected_assignment(&task.id, 1, "late")
        .await
        .unwrap();

    let t = db.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!((t.status, t.attempts), (TaskStatus::Queued, 1));
    assert!(executor(&harness, "aries:xpu0")
        .await
        .disabled_reason
        .is_none());
}

/// An agent that restarts within reconnect_grace registers without the task it had: the
/// backend requeues it instead of waiting forever. A task it still runs stays assigned.
#[tokio::test]
async fn test_reconnect_requeues_tasks_the_agent_no_longer_has() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let (tx, inbound) = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "a", "width": 1024, "height": 1024 }),
    )
    .await;
    submit(
        &harness,
        json!({ "prompt": "b", "width": 1024, "height": 1024 }),
    )
    .await;
    let queued = db.list_queued_tasks().await.unwrap();
    let (lost, kept) = (&queued[0], &queued[1]);
    db.assign_task(&lost.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    db.assign_task(&kept.id, "aries", "xpu1")
        .await
        .unwrap()
        .unwrap();

    drop(tx);
    drop(inbound);
    let mut reg = sample_register_msg("aries");
    reg.running = vec![proto::RunningTask {
        task: kept.id.clone(),
        attempt: 1,
        executor: "xpu1".into(),
    }];
    let _agent = connect_agent(&harness, reg).await;

    let t = wait_task(&harness, &lost.id, |t| t.status == TaskStatus::Queued).await;
    assert_eq!(t.attempts, 1, "the lost attempt counts");
    let k = db.get_task(&kept.id).await.unwrap().unwrap();
    assert_eq!(k.status, TaskStatus::Assigned);
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!(
        (ex1.state, ex1.current_task_id.as_deref()),
        (ExecutorState::Busy, Some(kept.id.as_str()))
    );
}

/// The reconnect-grace timer of an earlier session does nothing: after disconnect,
/// reconnect, disconnect, the first timer reclaimed the tasks moments after the second
/// disconnect instead of a full grace later.
#[tokio::test]
async fn test_grace_timer_of_an_earlier_session_reclaims_nothing() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "a", "width": 1024, "height": 1024 }),
    )
    .await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);
    db.assign_task(&task.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    let first: String = sqlx::query_scalar("SELECT session FROM agents WHERE id = 'aries'")
        .fetch_one(db.pool())
        .await
        .unwrap();
    // The agent came back on a new session and dropped again.
    sqlx::query("UPDATE agents SET online = 0, session = 'sess_second' WHERE id = 'aries'")
        .execute(db.pool())
        .await
        .unwrap();

    let affected = db
        .expire_disconnected_agent_tasks("aries", &first)
        .await
        .unwrap();
    assert!(affected.is_empty());
    let t = db.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(t.status, TaskStatus::Assigned);

    let affected = db
        .expire_disconnected_agent_tasks("aries", "sess_second")
        .await
        .unwrap();
    assert_eq!(affected.len(), 1, "the second session's own timer reclaims");
}

#[tokio::test]
async fn test_enhance_progress_keeps_sub_stage_and_stats() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let (tx, _inbound) = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024, "enhance": true }),
    )
    .await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);
    assert_eq!(task.stage, "enhance");
    db.assign_task(&task.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();

    let stats: HashMap<String, f64> = [
        ("generated_tokens".to_string(), 512.0),
        ("max_new_tokens".to_string(), 2048.0),
    ]
    .into();
    tx.send(proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Progress(proto::Progress {
            task: task.id.clone(),
            attempt: 1,
            step: 512,
            steps: 2048,
            s_per_step: 0.05,
            sub_stage: "thinking".into(),
            stats: stats.clone(),
        })),
    })
    .unwrap();
    let t = wait_task(&harness, &task.id, |t| t.sub_stage.is_some()).await;
    assert_eq!(t.sub_stage.as_deref(), Some("thinking"));
    assert_eq!(t.stats, Some(stats));
}

/// A denoise backlog gets a second worker on the idle card; a short stage does not.
#[tokio::test]
async fn test_denoise_backlog_uses_both_executors() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    for i in 0..2 {
        submit(
            &harness,
            json!({ "prompt": format!("p{i}"), "width": 1024, "height": 1024 }),
        )
        .await;
    }
    // Fast-forward both jobs to "encode done, denoise queued".
    for q in [
        "UPDATE jobs SET model_id = 'Qwen/Qwen-Image-2.1', model_rev = 'rev123'",
        "UPDATE tasks SET status = 'done' WHERE stage = 'encode'",
        "UPDATE tasks SET status = 'queued', queued_at = 1 WHERE stage = 'denoise'",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("denoise"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    db.update_executor_state("aries", "xpu1", ExecutorState::Idle, None, None, None)
        .await
        .unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    assert_eq!(
        executor(&harness, "aries:xpu0").await.state,
        ExecutorState::Busy
    );
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!(
        (ex1.state, ex1.current_stage.as_deref()),
        (ExecutorState::Starting, Some("denoise"))
    );
}

/// A retried (or rejected) task gets a fresh attempt number: reusing one collided with the
/// first attempt's row and stalled the scheduler.
#[tokio::test]
async fn test_retry_and_rejection_use_new_attempt_numbers() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let job: Value = submit(
        &harness,
        json!({ "id": "job-retry", "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await
    .json()
    .await
    .unwrap();
    let task_id = job["tasks"][0]["id"].as_str().unwrap().to_string();

    for attempt in 1..=2 {
        db.assign_task(&task_id, "aries", "xpu0")
            .await
            .unwrap()
            .unwrap();
        db.fail_task_attempt(
            &task_id,
            attempt,
            AttemptFailure::studio("python_error", "boom"),
        )
        .await
        .unwrap();
    }
    assert_eq!(
        db.get_job("job-retry").await.unwrap().unwrap().status,
        JobStatus::Failed
    );

    db.retry_job("job-retry").await.unwrap();
    let (t, _, attempt) = db
        .assign_task(&task_id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (attempt, t.attempts),
        (3, 1),
        "new attempt number, fresh retry budget"
    );

    db.handle_rejected_assignment(&task_id, 3, "nope")
        .await
        .unwrap();
    db.enable_executor("aries:xpu0").await.unwrap();
    let (t, _, attempt) = db
        .assign_task(&task_id, "aries", "xpu1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (attempt, t.attempts),
        (4, 1),
        "a rejection gives the retry back"
    );
}

/// A retry runs again the finished stages whose output the intermediate cleanup deleted:
/// otherwise the decode went out without its latent and failed again.
#[tokio::test]
async fn test_retry_reruns_stages_whose_output_was_collected() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "id": "job-gc", "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    let job = db.get_job("job-gc").await.unwrap().unwrap();
    let task = |stage: &str| {
        job.tasks
            .iter()
            .find(|t| t.stage == stage)
            .unwrap()
            .id
            .clone()
    };
    let (encode, denoise) = (task("encode"), task("denoise"));
    for q in [
        "UPDATE tasks SET status = 'done', current_attempt = 1 WHERE stage IN ('encode', 'denoise')",
        "UPDATE tasks SET status = 'failed', current_attempt = 2 WHERE stage = 'decode'",
        "UPDATE jobs SET status = 'failed', finished_at = 1",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    async fn artifact(db: &Database, task_id: String, name: &str) {
        sqlx::query(
            "INSERT INTO artifacts (id, job_id, task_id, attempt, name, sha256, bytes, kind, created_at)
             VALUES (?, 'job-gc', ?, 1, ?, 'x', 1, 'intermediate', 1)",
        )
        .bind(format!("art-{name}"))
        .bind(task_id)
        .bind(name)
        .execute(db.pool())
        .await
        .unwrap();
    }
    let statuses = || async {
        let job = db.get_job("job-gc").await.unwrap().unwrap();
        job.tasks
            .iter()
            .map(|t| (t.stage.clone(), t.status))
            .collect::<Vec<_>>()
    };

    // The latent is gone, the embeds are still there: denoise runs again, then decode.
    artifact(db, encode, "embeds.safetensors").await;
    db.retry_job("job-gc").await.unwrap();
    let s = statuses().await;
    assert!(s.contains(&("encode".into(), TaskStatus::Done)), "{s:?}");
    assert!(s.contains(&("denoise".into(), TaskStatus::Queued)), "{s:?}");
    assert!(s.contains(&("decode".into(), TaskStatus::Waiting)), "{s:?}");

    // With the latent kept, only the decode runs again.
    for q in [
        "UPDATE tasks SET status = 'done' WHERE stage = 'denoise'",
        "UPDATE tasks SET status = 'failed' WHERE stage = 'decode'",
        "UPDATE jobs SET status = 'failed'",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    artifact(db, denoise, "latent.safetensors").await;
    db.retry_job("job-gc").await.unwrap();
    let s = statuses().await;
    assert!(s.contains(&("denoise".into(), TaskStatus::Done)), "{s:?}");
    assert!(s.contains(&("decode".into(), TaskStatus::Queued)), "{s:?}");
}

/// After a reconnect an executor re-announces READY(enhance) without saying which enhancer
/// is loaded. It must not get an enhance on trust (a text enhancer given a job with images
/// rejects it and gets disabled): it starts again with a known model.
#[tokio::test]
async fn test_enhancer_of_unknown_model_is_restarted_not_trusted() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "id": "enh", "prompt": "e", "width": 1024, "height": 1024, "enhance": true }),
    )
    .await;
    db.disable_executor("aries:xpu1", "test").await.unwrap();
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("enhance"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    // What a re-registration leaves (`upsert_agent_registration` clears the model).
    sqlx::query("UPDATE executors SET current_model_id = NULL, current_model_rev = NULL WHERE key = 'aries:xpu0'")
        .execute(db.pool())
        .await
        .unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    let ex0 = executor(&harness, "aries:xpu0").await;
    assert_ne!(
        ex0.state,
        ExecutorState::Busy,
        "no task on an unknown model"
    );
    run_scheduler_tick(&harness.state).await.unwrap();
    let ex0 = executor(&harness, "aries:xpu0").await;
    assert_eq!(
        (ex0.state, ex0.current_stage.as_deref()),
        (ExecutorState::Starting, Some("enhance"))
    );
    assert!(ex0.current_model_id.is_some(), "started with a known model");
}

/// Seen live: xpu1 finished its encodes and sat idle on encode for a minute and a half,
/// because a third job was still enhancing on xpu0, while two denoise tasks waited. A
/// worker stays only while its own stage has queued work.
#[tokio::test]
async fn test_idle_worker_switches_to_waiting_stage() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "id": "enh", "prompt": "e", "width": 1024, "height": 1024, "enhance": true }),
    )
    .await;
    for id in ["d1", "d2"] {
        submit(
            &harness,
            json!({ "id": id, "prompt": id, "width": 1024, "height": 1024 }),
        )
        .await;
    }
    for q in [
        "UPDATE jobs SET model_id = 'Qwen/Qwen-Image-2.1', model_rev = 'rev123' WHERE id IN ('d1', 'd2')",
        "UPDATE tasks SET status = 'done' WHERE stage = 'encode' AND job_id IN ('d1', 'd2')",
        "UPDATE tasks SET status = 'queued', queued_at = (julianday('now') - 2440587.5) * 86400.0 WHERE stage = 'denoise' AND job_id IN ('d1', 'd2')",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    let enhance = db
        .list_queued_tasks()
        .await
        .unwrap()
        .into_iter()
        .find(|t| t.stage == "enhance")
        .unwrap();
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("enhance"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    db.assign_task(&enhance.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    db.update_executor_state(
        "aries",
        "xpu1",
        ExecutorState::Ready,
        Some("encode"),
        Some(2),
        None,
    )
    .await
    .unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!(
        (ex1.state, ex1.current_stage.as_deref()),
        (ExecutorState::Starting, Some("denoise"))
    );
}

/// No switch when the other stage's queue is already covered by a free worker.
#[tokio::test]
async fn test_idle_worker_does_not_switch_to_covered_stage() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "id": "d1", "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    for q in [
        "UPDATE jobs SET model_id = 'Qwen/Qwen-Image-2.1', model_rev = 'rev123'",
        "UPDATE tasks SET status = 'done' WHERE stage = 'encode'",
        "UPDATE tasks SET status = 'queued', queued_at = (julianday('now') - 2440587.5) * 86400.0 WHERE stage = 'denoise'",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    let model = aries_image_studio::db::ModelDecl {
        id: "Qwen/Qwen-Image-2.1".into(),
        rev: "rev123".into(),
        dtype: "bf16".into(),
    };
    db.mark_executor_starting("aries", "xpu0", "denoise", &model)
        .await
        .unwrap();
    db.update_executor_state(
        "aries",
        "xpu1",
        ExecutorState::Ready,
        Some("encode"),
        Some(2),
        None,
    )
    .await
    .unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    let ex0 = executor(&harness, "aries:xpu0").await;
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_ne!(
        ex1.current_stage.as_deref(),
        Some("denoise"),
        "xpu0={ex0:?}"
    );
}

/// The SPA keeps no local state: favorites, UI metadata, deletions and preferences all
/// round-trip through Studio.
#[tokio::test]
async fn test_studio_owns_ui_state() {
    let harness = start_test_harness().await;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let http = reqwest::Client::new();
    let url = |p: &str| format!("{}{p}", harness.http_url);

    let meta = json!({ "batch_id": "turn-1", "aspect": "1:1", "quality": "draft" });
    let job: Value = submit(
        &harness,
        json!({ "id": "j1", "prompt": "p", "width": 1024, "height": 1024, "meta": meta }),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(job["meta"], meta);
    assert_eq!(job["favorite"], false);
    let big = json!({ "x": "y".repeat(5000) });
    assert_eq!(
        submit(
            &harness,
            json!({ "id": "j2", "prompt": "p", "width": 1024, "height": 1024, "meta": big })
        )
        .await
        .status(),
        400
    );

    let listed: Value = http
        .get(url("/api/jobs"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let v0 = listed["version"].as_i64().unwrap();

    let fav: Value = http
        .patch(url("/api/jobs/j1"))
        .json(&json!({ "favorite": true }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(fav["favorite"], true);
    let changed: Value = http
        .get(url(&format!("/api/jobs?changed_since={v0}")))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(changed["jobs"][0]["id"], "j1");

    let v1 = changed["version"].as_i64().unwrap();
    assert_eq!(
        http.delete(url("/api/jobs/j1"))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
    let after: Value = http
        .get(url(&format!("/api/jobs?changed_since={v1}")))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(after["deleted"], json!(["j1"]));
    assert_eq!(after["jobs"], json!([]));

    let empty: Value = http
        .get(url("/api/preferences"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(empty, json!({}));
    let prefs = json!({ "theme": "dark", "steps": 20 });
    assert_eq!(
        http.put(url("/api/preferences"))
            .json(&prefs)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let got: Value = http
        .get(url("/api/preferences"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(got, prefs);
    assert_eq!(
        http.put(url("/api/preferences"))
            .json(&json!([1, 2]))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
}

#[test]
fn test_enhance_request_states_the_job_ratio() {
    use aries_image_studio::scheduler::{aspect_label, enhance_request};
    for (w, h, label) in [
        (2048, 2048, "1:1"),
        (2400, 1792, "4:3"),
        (1792, 2400, "3:4"),
        (2528, 1696, "3:2"),
        (848, 1264, "2:3"),
        (416, 640, "2:3"),
        (2752, 1536, "16:9"),
        (768, 1376, "9:16"),
    ] {
        assert_eq!(aspect_label(w, h), label, "{w}x{h}");
    }
    assert_eq!(
        enhance_request("a sorcerer \n", 848, 1264),
        "a sorcerer\n\nAspect ratio: 2:3"
    );
}

/// Tiling is per job: refused at submit when no executor declares tiled decode.
#[tokio::test]
async fn test_tiling_needs_an_executor_that_declares_it() {
    let harness = start_test_harness().await;
    let mut reg = sample_register_msg("aries");
    for ex in &mut reg.executors {
        for cap in &mut ex.stages {
            if cap.stage == proto::Stage::Decode as i32 {
                cap.modes = vec![proto::DecodeMode::Untiled as i32];
            }
        }
    }
    let _agent = connect_agent(&harness, reg).await;
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024, "tiling": true }),
    )
    .await;
    assert_eq!(resp.status(), 422);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["rejected"].as_str().unwrap().contains("tiled"),
        "{body}"
    );
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    assert_eq!(resp.status(), 201);
    let job: Value = resp.json().await.unwrap();
    assert_eq!(job["tiling"], false, "off by default");
}

/// The next Assign, answering StartWorker with READY and Release with IDLE on the way.
async fn next_assign(link: &mut AgentLink) -> proto::Assign {
    let (tx, inbound) = link;
    loop {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), inbound.next())
            .await
            .expect("no Assign from the backend")
            .unwrap()
            .unwrap();
        let reply =
            |executor: String, state: proto::ExecutorStateKind, stage: i32| proto::AgentMessage {
                msg: Some(proto::agent_message::Msg::ExecutorState(
                    proto::ExecutorState {
                        executor,
                        state: state as i32,
                        stage,
                        pid: 1,
                        load_s: 1.0,
                        detail: String::new(),
                    },
                )),
            };
        match msg.msg {
            Some(proto::backend_message::Msg::StartWorker(sw)) => tx
                .send(reply(
                    sw.executor,
                    proto::ExecutorStateKind::ExecutorStateReady,
                    sw.stage,
                ))
                .unwrap(),
            Some(proto::backend_message::Msg::Release(rel)) => tx
                .send(reply(
                    rel.executor,
                    proto::ExecutorStateKind::ExecutorStateIdle,
                    0,
                ))
                .unwrap(),
            Some(proto::backend_message::Msg::Assign(a)) => return a,
            _ => {}
        }
    }
}

/// PUTs `payload` to an upload URL of an assignment, as the agent does.
async fn put_artifact(
    harness: &TestHarness,
    url: &str,
    payload: &[u8],
) -> (String, reqwest::StatusCode) {
    let sha = hex::encode(Sha256::digest(payload));
    let status = reqwest::Client::new()
        .put(url)
        .header("Authorization", format!("Bearer {}", harness.token))
        .header("x-content-sha256", &sha)
        .body(payload.to_vec())
        .send()
        .await
        .unwrap()
        .status();
    (sha, status)
}

fn done_msg(assign: &proto::Assign) -> proto::AgentMessage {
    proto::AgentMessage {
        msg: Some(proto::agent_message::Msg::Done(proto::Done {
            task: assign.task.clone(),
            attempt: assign.attempt,
            artifacts: vec![],
            timings: HashMap::from([("encode_s".to_string(), 1.0)]),
            peak_mem_bytes: 1,
            provenance: Some(proto::Provenance {
                agent: "aries".to_string(),
                executor: assign.executor.clone(),
                device: "0000:04:00.0".to_string(),
                backend_kind: "xpu".to_string(),
                model: Some(proto::ModelRef {
                    id: "Qwen/Qwen-Image-2.1".to_string(),
                    rev: "rev123".to_string(),
                    dtype: "bf16".to_string(),
                }),
                torch_version: "2.14.1+xpu".to_string(),
                worker_version: "0.4.0".to_string(),
            }),
        })),
    }
}

/// docs/REFERENCE_IMAGES.md: uploads, references on jobs, one encode per send, latents
/// encoded once and reused, and denoise fed the stored latents.
#[tokio::test]
async fn test_reference_images_flow() {
    let harness = start_test_harness().await;
    let http = reqwest::Client::new();
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;

    // Uploads: stored by hash, the same image twice is one upload; not an image is refused.
    let upload = |bytes: Vec<u8>| {
        let http = http.clone();
        let url = format!("{}/api/uploads", harness.http_url);
        async move { http.post(url).body(bytes).send().await.unwrap() }
    };
    let a: Value = upload(build_png_bytes(640, 480))
        .await
        .json()
        .await
        .unwrap();
    let b: Value = upload(build_png_bytes(300, 300))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(
        (a["width"].as_u64(), a["height"].as_u64()),
        (Some(640), Some(480))
    );
    let again = upload(build_png_bytes(640, 480)).await;
    assert_eq!(again.json::<Value>().await.unwrap()["id"], a["id"]);
    assert_eq!(upload(b"not an image".to_vec()).await.status(), 422);
    let (a, b) = (
        a["id"].as_str().unwrap().to_string(),
        b["id"].as_str().unwrap().to_string(),
    );
    let got = http
        .get(format!("{}/api/uploads/{a}", harness.http_url))
        .send()
        .await
        .unwrap();
    assert_eq!(got.headers()["content-type"], "image/png");
    let thumb = http
        .get(format!(
            "{}/api/uploads/{a}/thumb?size=512",
            harness.http_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(
        (
            thumb.status(),
            thumb.headers()["content-type"].to_str().unwrap()
        ),
        (reqwest::StatusCode::OK, "image/jpeg")
    );

    // Refused at submit: enhance with references, more than the agent declares, unknown uploads.
    let refs = json!([{ "upload": a }, { "upload": b }]);
    let job = |prompt: &str, seed: u64, refs: &Value| json!({ "prompt": prompt, "width": 1024, "height": 1024, "steps": 2, "seed": seed, "references": refs });
    let mut enhanced = job("p", 1, &refs);
    enhanced["enhance"] = json!(true);
    assert_eq!(submit(&harness, enhanced).await.status(), 422);
    let five = json!([{ "upload": a }, { "upload": a }, { "upload": a }, { "upload": a }, { "upload": a }]);
    let resp = submit(&harness, job("p", 1, &five)).await;
    assert_eq!(resp.status(), 422);
    assert!(resp.json::<Value>().await.unwrap()["rejected"]
        .as_str()
        .unwrap()
        .contains("at most 4"));
    assert_eq!(
        submit(
            &harness,
            job("p", 1, &json!([{ "upload": "f".repeat(64) }]))
        )
        .await
        .status(),
        422
    );
    let caps: Value = http
        .get(format!("{}/api/capabilities", harness.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(caps["max_references_ever"], 4);

    // Two images of one send: same prompt and references, other seeds.
    let j1: Value = submit(&harness, job("these two in one room", 1, &refs))
        .await
        .json()
        .await
        .unwrap();
    let j2: Value = submit(&harness, job("these two in one room", 2, &refs))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(j1["references"][1]["sha256"], b.as_str());
    let enc = next_assign(&mut link).await;
    assert_eq!(
        proto::Stage::try_from(enc.stage).unwrap(),
        proto::Stage::Encode
    );
    let Some(proto::assign::Params::Encode(p)) = &enc.params else {
        panic!("expected EncodeParams")
    };
    assert_eq!((p.references.len(), p.reference_res), (2, 1024));
    let input_names: Vec<_> = enc.inputs.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(input_names, ["ref0.image", "ref1.image"]);
    for inp in &enc.inputs {
        let bytes = http
            .get(&inp.url)
            .header("Authorization", format!("Bearer {}", harness.token))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(hex::encode(Sha256::digest(&bytes)), inp.sha256);
    }
    let mut upload_names: Vec<_> = enc.uploads.iter().map(|u| u.name.as_str()).collect();
    upload_names.sort();
    assert_eq!(
        upload_names,
        ["embeds.safetensors", "ref0.safetensors", "ref1.safetensors"]
    );

    // The worker's outputs: latents per reference, embeds with the mask and the sizes.
    let sizes = [(1184u32, 896u32), (1024, 1024)];
    for (i, (w, h)) in sizes.iter().enumerate() {
        let tokens = (w / 16 * (h / 16)) as usize;
        let latents = build_safetensors_bytes(
            &[("latents", "BF16", vec![1, tokens, 64], &[0u8; 2])],
            &[
                ("width", &w.to_string()),
                ("height", &h.to_string()),
                ("reference_res", "1024"),
                ("source_sha256", if i == 0 { &a } else { &b }),
                ("model", "Qwen/Qwen-Image-2.1"),
                ("rev", "rev123"),
                ("dtype", "bf16"),
            ],
        );
        let url = &enc
            .uploads
            .iter()
            .find(|u| u.name == format!("ref{i}.safetensors"))
            .unwrap()
            .url;
        assert_eq!(put_artifact(&harness, url, &latents).await.1, 201);
    }
    // Latents of the wrong image are refused.
    let wrong = build_safetensors_bytes(
        &[("latents", "BF16", vec![1, 4144, 64], &[0u8; 2])],
        &[
            ("width", "1184"),
            ("height", "896"),
            ("reference_res", "1024"),
            ("source_sha256", &b),
            ("model", "m"),
            ("rev", "r"),
            ("dtype", "bf16"),
        ],
    );
    let url0 = &enc
        .uploads
        .iter()
        .find(|u| u.name == "ref0.safetensors")
        .unwrap()
        .url;
    assert_eq!(put_artifact(&harness, url0, &wrong).await.1, 422);
    let embeds = build_safetensors_bytes(
        &[
            ("prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2]),
            ("image_pad_mask", "BOOL", vec![1, 16], &[0u8; 2]),
        ],
        &[
            ("prompt", "these two in one room"),
            ("prompt_source", "original"),
            ("model", "Qwen/Qwen-Image-2.1"),
            ("rev", "rev123"),
            ("dtype", "bf16"),
            ("reference_sizes", "[[1184, 896], [1024, 1024]]"),
        ],
    );
    let url = &enc
        .uploads
        .iter()
        .find(|u| u.name == "embeds.safetensors")
        .unwrap()
        .url;
    let (embeds_sha, status) = put_artifact(&harness, url, &embeds).await;
    assert_eq!(status, 201);
    link.0.send(done_msg(&enc)).unwrap();

    // The other image of the send takes the same embeds instead of encoding again.
    let job2 = harness
        .state
        .db
        .get_job(j2["id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    let enc2 = job2.tasks.iter().find(|t| t.stage == "encode").unwrap();
    wait_task(&harness, &enc2.id, |t| t.status == TaskStatus::Done).await;
    let shared = harness
        .state
        .db
        .get_job_artifact_by_name(&job2.id, "embeds.safetensors")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(shared.sha256, embeds_sha);
    let job2 = harness.state.db.get_job(&job2.id).await.unwrap().unwrap();
    // It records the encode it took: where it ran, its time and memory, marked shared.
    let enc_prov = &job2.provenance["encode"];
    assert_eq!(enc_prov.timings.get("shared"), Some(&1.0));
    assert_eq!(enc_prov.timings.get("encode_s"), Some(&1.0));
    assert_eq!(enc_prov.peak_mem_bytes, 1);
    assert_eq!(job2.model_rev.as_deref(), Some("rev123"));

    // Denoise reads the stored latents, never the pixels.
    let den = next_assign(&mut link).await;
    assert_eq!(
        proto::Stage::try_from(den.stage).unwrap(),
        proto::Stage::Denoise
    );
    let Some(proto::assign::Params::Denoise(dp)) = &den.params else {
        panic!("expected DenoiseParams")
    };
    assert_eq!(dp.references, 2);
    let mut names: Vec<_> = den.inputs.iter().map(|i| i.name.as_str()).collect();
    names.sort();
    assert_eq!(
        names,
        ["embeds.safetensors", "ref0.safetensors", "ref1.safetensors"]
    );
    for inp in den.inputs.iter().filter(|i| i.name.starts_with("ref")) {
        let resp = http
            .get(&inp.url)
            .header("Authorization", format!("Bearer {}", harness.token))
            .send()
            .await
            .unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(resp.bytes().await.unwrap())),
            inp.sha256
        );
    }
    let anon = http
        .get(
            &den.inputs
                .iter()
                .find(|i| i.name == "ref0.safetensors")
                .unwrap()
                .url,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(anon.status(), 401);

    // Free both cards: cancel the send, and the agent reports its denoise workers ready.
    for j in [&j1, &j2] {
        http.post(format!(
            "{}/api/jobs/{}/cancel",
            harness.http_url,
            j["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap();
    }
    for ex in ["xpu0", "xpu1"] {
        link.0
            .send(proto::AgentMessage {
                msg: Some(proto::agent_message::Msg::ExecutorState(
                    proto::ExecutorState {
                        executor: ex.to_string(),
                        state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                        stage: proto::Stage::Denoise as i32,
                        pid: 1,
                        load_s: 1.0,
                        detail: String::new(),
                    },
                )),
            })
            .unwrap();
    }

    // A later job with one of these references and another prompt: its own encode, but the
    // reference's latents are reused, so nothing to upload but the embeds.
    let j3: Value = submit(
        &harness,
        job("the same, at night", 3, &json!([{ "upload": b }])),
    )
    .await
    .json()
    .await
    .unwrap();
    let job3 = harness
        .state
        .db
        .get_job(j3["id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    let enc3_id = job3
        .tasks
        .iter()
        .find(|t| t.stage == "encode")
        .unwrap()
        .id
        .clone();
    let enc3 = loop {
        let a = next_assign(&mut link).await;
        if a.task == enc3_id {
            break a;
        }
    };
    assert_eq!(
        enc3.inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["ref0.image"]
    );
    assert_eq!(
        enc3.uploads
            .iter()
            .map(|u| u.name.as_str())
            .collect::<Vec<_>>(),
        ["embeds.safetensors"]
    );
}

#[test]
fn test_reference_limit_applies_only_above_its_count() {
    use aries_image_studio::db::{LimitDecl, RequirementDecl, TaskShape};
    let shape = |px, refs| TaskShape {
        px,
        refs,
        tiled: false,
    };
    let refs_limit = LimitDecl {
        stage: "denoise".to_string(),
        above_px: 0,
        requires: vec![RequirementDecl::MaxConcurrentPerAgent(1)],
        above_references: Some(3),
        decode_mode: None,
    };
    assert!(!refs_limit.applies("denoise", shape(4_194_304, 3)));
    assert!(refs_limit.applies("denoise", shape(262_144, 4)));
    assert!(!refs_limit.applies("decode", shape(4_194_304, 4)));
    // A limit without a reference condition (full-size decode) ignores references.
    let decode = LimitDecl {
        above_references: None,
        stage: "decode".to_string(),
        above_px: 3_500_000,
        ..refs_limit
    };
    assert!(decode.applies("decode", shape(4_194_304, 0)));
    assert!(!decode.applies("decode", shape(1_048_576, 4)));
    // One for untiled decodes leaves tiled ones alone (they stay on the card).
    let untiled = LimitDecl {
        decode_mode: Some("untiled".to_string()),
        ..decode
    };
    assert!(untiled.applies("decode", shape(4_194_304, 0)));
    assert!(!untiled.applies(
        "decode",
        TaskShape {
            px: 4_194_304,
            refs: 0,
            tiled: true
        }
    ));
}

/// Uploads nobody refers to go a day after upload; ones a job uses stay, with their blob.
#[tokio::test]
async fn test_upload_gc_keeps_referenced_images() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let old = now_s() - 2.0 * 86400.0;
    let mut shas = Vec::new();
    for size in [64, 96] {
        let (sha, bytes) = harness
            .state
            .blobs
            .put_blob(&build_png_bytes(size, size))
            .unwrap();
        db.insert_upload(&aries_image_studio::db::UploadRecord {
            sha256: sha.clone(),
            mime: "image/png".to_string(),
            width: size,
            height: size,
            bytes,
            created_at: old,
        })
        .await
        .unwrap();
        shas.push(sha);
    }
    let resp = submit(&harness, json!({ "prompt": "p", "width": 1024, "height": 1024, "references": [{ "upload": shas[0] }] })).await;
    assert_eq!(resp.status(), 201);

    let orphans = db.gc_uploads(86400.0).await.unwrap();
    assert_eq!(orphans, vec![shas[1].clone()]);
    assert!(db.get_upload(&shas[0]).await.unwrap().is_some());
    assert!(db.get_upload(&shas[1]).await.unwrap().is_none());
}

/// Advanced "Enhancer tokens": the job's limit reaches the enhance assignment; outside
/// 256–24576 it is refused, and without Enhance it is not kept.
#[tokio::test]
async fn test_enhance_max_tokens_reaches_the_worker() {
    let harness = start_test_harness().await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    let job = |tokens: Value, enhance: bool| json!({ "prompt": "a detailed infographic", "width": 1024, "height": 1024, "enhance": enhance, "enhance_max_tokens": tokens });
    for bad in [100, 24577] {
        assert_eq!(
            submit(&harness, job(json!(bad), true)).await.status(),
            422,
            "{bad}"
        );
    }
    let j: Value = submit(&harness, job(json!(4096), true))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(j["enhance_max_tokens"], 4096);
    let enhance_task = j["tasks"][0]["id"].as_str().unwrap().to_string();
    let a = loop {
        let a = next_assign(&mut link).await;
        if a.task == enhance_task {
            break a;
        }
    };
    let Some(proto::assign::Params::Enhance(p)) = &a.params else {
        panic!("expected EnhanceParams")
    };
    assert_eq!(p.max_new_tokens, 4096);

    let plain: Value = submit(&harness, job(json!(4096), false))
        .await
        .json()
        .await
        .unwrap();
    assert!(plain["enhance_max_tokens"].is_null());
    assert_eq!(
        submit(&harness, job(json!(24576), true)).await.status(),
        201
    );
}

const T2I: &str = "Qwen/Qwen-Image-2.1-PE-T2I";
const I2I: &str = "Qwen/Qwen-Image-2.1-PE-I2I";

/// The sample agent with the image-aware enhancer on xpu0 (docs/ENHANCER_I2I.md).
fn i2i_register_msg(agent_id: &str) -> proto::Register {
    let mut reg = sample_register_msg(agent_id);
    reg.models.push(proto::ModelRef {
        id: I2I.to_string(),
        rev: "revi2i".to_string(),
        dtype: "int4".to_string(),
    });
    let enhance = reg.executors[0]
        .stages
        .iter_mut()
        .find(|s| s.stage == proto::Stage::Enhance as i32)
        .unwrap();
    enhance.models = vec![T2I.to_string(), I2I.to_string()];
    enhance.reference_models = vec![I2I.to_string()];
    enhance.max_references = 4;
    reg
}

async fn upload_png(harness: &TestHarness, w: u32, h: u32) -> String {
    let resp: Value = reqwest::Client::new()
        .post(format!("{}/api/uploads", harness.http_url))
        .body(build_png_bytes(w, h))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    resp["id"].as_str().unwrap().to_string()
}

/// The StartWorker models seen until the next Assign (answered READY), and that Assign.
async fn next_assign_with_starts(link: &mut AgentLink) -> (Vec<String>, proto::Assign) {
    let (tx, inbound) = link;
    let mut started = Vec::new();
    loop {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), inbound.next())
            .await
            .expect("no Assign from the backend")
            .unwrap()
            .unwrap();
        match msg.msg {
            Some(proto::backend_message::Msg::StartWorker(sw)) => {
                started.push(sw.model.map(|m| m.id).unwrap_or_default());
                tx.send(proto::AgentMessage {
                    msg: Some(proto::agent_message::Msg::ExecutorState(
                        proto::ExecutorState {
                            executor: sw.executor,
                            state: proto::ExecutorStateKind::ExecutorStateReady as i32,
                            stage: sw.stage,
                            pid: 1,
                            load_s: 1.0,
                            detail: String::new(),
                        },
                    )),
                })
                .unwrap();
            }
            Some(proto::backend_message::Msg::Assign(a)) => return (started, a),
            _ => {}
        }
    }
}

/// Enhance with references runs on PE-I2I: accepted only when an agent declares it, the
/// worker loads that model and gets the images; `ratio_follow` is kept on the job.
#[tokio::test]
async fn test_image_enhancer_takes_jobs_with_references() {
    let harness = start_test_harness().await;
    let http = reqwest::Client::new();
    let caps_url = format!("{}/api/capabilities", harness.http_url);

    // A text-only enhancer: Enhance with references is refused.
    let plain = start_test_harness().await;
    let _plain_agent = connect_agent(&plain, sample_register_msg("aries")).await;
    let sha = upload_png(&plain, 640, 480).await;
    let body = json!({ "prompt": "add a hat", "width": 1024, "height": 1024, "enhance": true, "references": [{ "upload": sha }] });
    let resp = submit(&plain, body).await;
    assert_eq!(resp.status(), 422);
    assert!(resp.json::<Value>().await.unwrap()["rejected"]
        .as_str()
        .unwrap()
        .contains("enhance with 1 reference"));
    let caps: Value = http
        .get(format!("{}/api/capabilities", plain.http_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(caps["enhance_max_references_ever"], 0);

    let mut link = connect_agent(&harness, i2i_register_msg("aries")).await;
    let caps: Value = http
        .get(&caps_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (
            caps["enhance_max_references_ever"].as_u64(),
            caps["enhance_max_references_now"].as_u64()
        ),
        (Some(4), Some(4))
    );
    let (a, b) = (
        upload_png(&harness, 640, 480).await,
        upload_png(&harness, 300, 300).await,
    );
    let five = json!([{ "upload": a }, { "upload": a }, { "upload": a }, { "upload": a }, { "upload": a }]);
    let body = |refs: Value| json!({ "prompt": "put image 1 into image 2", "width": 1024, "height": 1024, "enhance": true, "references": refs });
    assert_eq!(submit(&harness, body(five)).await.status(), 422);

    let j: Value = submit(&harness, body(json!([{ "upload": a }, { "upload": b }])))
        .await
        .json()
        .await
        .unwrap();
    let (started, enh) = next_assign_with_starts(&mut link).await;
    assert_eq!(started, [I2I]);
    assert_eq!(
        proto::Stage::try_from(enh.stage).unwrap(),
        proto::Stage::Enhance
    );
    let Some(proto::assign::Params::Enhance(p)) = &enh.params else {
        panic!("expected EnhanceParams")
    };
    assert_eq!(
        p.references
            .iter()
            .map(|r| r.sha256.as_str())
            .collect::<Vec<_>>(),
        [a.as_str(), b.as_str()]
    );
    assert_eq!(
        enh.inputs
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["ref0.image", "ref1.image"]
    );
    assert_eq!(enh.inputs[1].sha256, b);

    let enhanced = json!({ "rewritten_prompt": "a goblin in the cafe", "wh_ratio": "1:1", "ratio_follow": "<image2>" });
    let url = &enh
        .uploads
        .iter()
        .find(|u| u.name == "enhanced.json")
        .unwrap()
        .url;
    assert_eq!(
        put_artifact(&harness, url, &serde_json::to_vec(&enhanced).unwrap())
            .await
            .1,
        201
    );
    link.0.send(done_msg(&enh)).unwrap();
    wait_task(&harness, &enh.task, |t| t.status == TaskStatus::Done).await;
    let job = harness
        .state
        .db
        .get_job(j["id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(job.enhanced_prompt.as_deref(), Some("a goblin in the cafe"));
    assert_eq!(job.ratio_follow.as_deref(), Some("<image2>"));
    let api: Value = http
        .get(format!("{}/api/jobs/{}", harness.http_url, job.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(api["ratio_follow"], "<image2>");
}

/// A text job's enhance loads PE-T2I; a ready PE-T2I worker never gets a job with images:
/// it switches to PE-I2I instead.
#[tokio::test]
async fn test_enhance_worker_matches_the_jobs_enhancer() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let mut link = connect_agent(&harness, i2i_register_msg("aries")).await;
    let text: Value = submit(
        &harness,
        json!({ "prompt": "a lighthouse", "width": 1024, "height": 1024, "enhance": true }),
    )
    .await
    .json()
    .await
    .unwrap();
    let (started, enh) = next_assign_with_starts(&mut link).await;
    assert_eq!(
        (started.as_slice(), enh.task.as_str()),
        (
            [T2I.to_string()].as_slice(),
            text["tasks"][0]["id"].as_str().unwrap()
        )
    );

    // The text job's enhance is done; its worker is ready on PE-T2I.
    http_cancel(&harness, text["id"].as_str().unwrap()).await;
    db.update_executor_state(
        "aries",
        "xpu0",
        ExecutorState::Ready,
        Some("enhance"),
        Some(1),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        executor(&harness, "aries:xpu0")
            .await
            .current_model_id
            .as_deref(),
        Some(T2I)
    );

    let sha = upload_png(&harness, 640, 480).await;
    let refs: Value = submit(&harness, json!({ "prompt": "add a hat", "width": 1024, "height": 1024, "enhance": true, "references": [{ "upload": sha }] }))
        .await
        .json()
        .await
        .unwrap();
    run_scheduler_tick(&harness.state).await.unwrap();
    let ex0 = executor(&harness, "aries:xpu0").await;
    assert_eq!(
        (ex0.state, ex0.current_model_id.as_deref()),
        (ExecutorState::Starting, Some(I2I)),
        "{ex0:?}"
    );
    let (started, enh) = next_assign_with_starts(&mut link).await;
    assert_eq!(started.last().map(String::as_str), Some(I2I));
    assert_eq!(enh.task, refs["tasks"][0]["id"].as_str().unwrap());
}

async fn http_cancel(harness: &TestHarness, job_id: &str) {
    reqwest::Client::new()
        .post(format!("{}/api/jobs/{job_id}/cancel", harness.http_url))
        .send()
        .await
        .unwrap();
}

/// The tasks that spill into host RAM share one slot per agent, whatever their stage: a
/// full-size decode and a denoise with many references never run side by side (seen live:
/// together they ended in a global OOM that took the desktop session with it).
#[test]
fn test_memory_heavy_tasks_take_turns_across_stages() {
    use aries_image_studio::db::{LimitDecl, RequirementDecl, TaskShape};
    use aries_image_studio::scheduler::check_limits_for_task;
    let slot = vec![RequirementDecl::MaxConcurrentPerAgent(1)];
    let mut agent: aries_image_studio::db::AgentRecord = serde_json::from_value(json!({
        "id": "aries", "agent_version": "t", "backend_kind": "xpu", "metrics_url": "", "online": true,
        "host_ram_total_bytes": 0, "models": [], "limits": [], "last_seen_at": 0.0
    }))
    .unwrap();
    agent.limits = vec![
        LimitDecl {
            stage: "decode".into(),
            above_px: 3_500_000,
            requires: slot.clone(),
            above_references: None,
            decode_mode: None,
        },
        LimitDecl {
            stage: "denoise".into(),
            above_px: 0,
            requires: slot,
            above_references: Some(2),
            decode_mode: None,
        },
    ];
    let ex: aries_image_studio::db::ExecutorRecord = serde_json::from_value(json!({
        "key": "aries:xpu1", "agent_id": "aries", "id": "xpu1", "device": "", "mem_total_bytes": 0,
        "stages": [], "state": "idle", "lent": false, "pending_lend": false, "last_seen_at": 0.0
    }))
    .unwrap();
    let mut decode: aries_image_studio::db::TaskRecord = serde_json::from_value(json!({
        "id": "t1", "job_id": "j1", "seq": 3, "stage": "decode", "status": "running", "attempts": 1,
        "current_attempt": 1, "step": 0, "steps": 0, "s_per_step": 0.0
    }))
    .unwrap();
    decode.assigned_agent = Some("aries".into());
    let full = 2752 * 1536;
    let shape = |px, refs| TaskShape {
        px,
        refs,
        tiled: false,
    };
    let running = vec![(decode, shape(full, 0))];
    let check = |stage: &str,
                 px: u64,
                 refs: u32,
                 active: &[(aries_image_studio::db::TaskRecord, TaskShape)]| {
        check_limits_for_task(&agent, &ex, stage, shape(px, refs), active, None, 5.0, 0.0)
    };
    // A full-size decode is running: a denoise with 3 or 4 references waits, a plain one does not.
    let err = check("denoise", full, 3, &running).unwrap_err();
    assert!(err.contains("memory-heavy"), "{err}");
    assert!(check("denoise", full, 4, &running).is_err());
    assert!(check("denoise", full, 2, &running).is_ok());
    assert!(check("denoise", full, 0, &running).is_ok());
    // And the other way round: a heavy denoise holds back a full-size decode, not a ½-size one.
    let mut den = running[0].0.clone();
    den.stage = "denoise".into();
    let running = vec![(den, shape(full, 4))];
    assert!(check("decode", full, 0, &running).is_err());
    assert!(check("decode", 1376 * 768, 0, &running).is_ok());
    assert!(check("denoise", full, 4, &[]).is_ok());
}

/// The upload an agent sends for `assign`'s stage (1024², no enhance).
fn stage_payload(assign: &proto::Assign) -> Vec<u8> {
    match proto::Stage::try_from(assign.stage).unwrap() {
        proto::Stage::Encode => {
            let Some(proto::assign::Params::Encode(p)) = &assign.params else {
                panic!("expected EncodeParams");
            };
            build_safetensors_bytes(
                &[
                    ("prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2048]),
                    ("prompt_embeds_mask", "I64", vec![1, 16], &[1u8; 128]),
                    (
                        "negative_prompt_embeds",
                        "BF16",
                        vec![1, 16, 64],
                        &[0u8; 2048],
                    ),
                    (
                        "negative_prompt_embeds_mask",
                        "I64",
                        vec![1, 16],
                        &[1u8; 128],
                    ),
                ],
                &[
                    ("prompt", &p.prompt),
                    ("prompt_source", &p.prompt_source),
                    ("model", "Qwen/Qwen-Image-2.1"),
                    ("rev", "rev123"),
                    ("dtype", "bf16"),
                ],
            )
        }
        proto::Stage::Denoise => build_safetensors_bytes(
            &[("latent", "BF16", vec![1, 16, 128, 128], &[0u8; 512])],
            &[
                ("model", "Qwen/Qwen-Image-2.1"),
                ("rev", "rev123"),
                ("dtype", "bf16"),
                ("width", "1024"),
                ("height", "1024"),
                ("seed", "1"),
                ("steps", "4"),
            ],
        ),
        proto::Stage::Decode => build_png_bytes(1024, 1024),
        s => panic!("unexpected stage {s:?}"),
    }
}

/// Blobs are content-addressed: two jobs that decode to the same bytes share the raw PNG's
/// blob, and the first to finish must not delete it from under the second.
#[tokio::test]
async fn test_identical_images_share_their_blob_safely() {
    let harness = start_test_harness().await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    let mut jobs = Vec::new();
    for prompt in ["a", "b"] {
        let resp = submit(
            &harness,
            json!({ "prompt": prompt, "width": 1024, "height": 1024, "enhance": false }),
        )
        .await;
        assert_eq!(resp.status(), 201);
        jobs.push(
            resp.json::<Value>().await.unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    // Both PNGs uploaded before either job finishes.
    let mut decodes = Vec::new();
    while decodes.len() < 2 {
        let assign = next_assign(&mut link).await;
        let (_, status) =
            put_artifact(&harness, &assign.uploads[0].url, &stage_payload(&assign)).await;
        assert_eq!(status, 201);
        if assign.stage == proto::Stage::Decode as i32 {
            decodes.push(assign);
        } else {
            link.0.send(done_msg(&assign)).unwrap();
        }
    }
    for assign in &decodes {
        link.0.send(done_msg(assign)).unwrap();
        wait_task(&harness, &assign.task, |t| t.status == TaskStatus::Done).await;
    }
    for job_id in &jobs {
        let mut ok = false;
        for _ in 0..200 {
            let job = harness.state.db.get_job(job_id).await.unwrap().unwrap();
            let art = harness
                .state
                .db
                .get_job_artifact_by_name(job_id, "image.png")
                .await
                .unwrap()
                .unwrap();
            if job.status == JobStatus::Done && art.metadata_json.is_some() {
                ok = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(ok, "job {job_id} never got its tagged PNG");
        let img = reqwest::get(format!("{}/api/images/{job_id}", harness.http_url))
            .await
            .unwrap();
        assert_eq!(img.status(), 200, "job {job_id}");
    }
}

/// An agent that declares only tiled decode takes jobs that ask for tiling.
#[tokio::test]
async fn test_tiled_only_decoder_serves_tiled_jobs() {
    let harness = start_test_harness().await;
    let mut reg = sample_register_msg("aries");
    for ex in &mut reg.executors {
        for cap in &mut ex.stages {
            if cap.stage == proto::Stage::Decode as i32 {
                cap.modes = vec![proto::DecodeMode::Tiled as i32];
            }
        }
    }
    let mut link = connect_agent(&harness, reg).await;
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024, "enhance": false, "tiling": true }),
    )
    .await;
    assert_eq!(resp.status(), 201);
    loop {
        let assign = next_assign(&mut link).await;
        if assign.stage == proto::Stage::Decode as i32 {
            break;
        }
        let (_, status) =
            put_artifact(&harness, &assign.uploads[0].url, &stage_payload(&assign)).await;
        assert_eq!(status, 201);
        link.0.send(done_msg(&assign)).unwrap();
    }
    // And untiled jobs are refused up front.
    let resp = submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    assert_eq!(resp.status(), 422);
}

/// A Done that Studio cannot record fails the attempt instead of leaving the task running
/// and its executor busy (which would drop the executor's next READY).
#[tokio::test]
async fn test_done_that_cannot_be_recorded_fails_the_attempt() {
    let harness = start_test_harness().await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024, "enhance": true }),
    )
    .await;
    let assign = next_assign(&mut link).await;
    assert_eq!(assign.stage, proto::Stage::Enhance as i32);
    let payload =
        serde_json::to_vec(&json!({ "rewritten_prompt": "x", "wh_ratio": "1:1" })).unwrap();
    let (sha, status) = put_artifact(&harness, &assign.uploads[0].url, &payload).await;
    assert_eq!(status, 201);
    // The blob vanishes before Done arrives: reading enhanced.json fails.
    harness.state.blobs.delete_blob(&sha).unwrap();
    link.0.send(done_msg(&assign)).unwrap();
    let task = wait_task(&harness, &assign.task, |t| {
        t.current_attempt == 2 || t.status == TaskStatus::Queued
    })
    .await;
    assert_ne!(task.status, TaskStatus::Running);
    let ex = executor(&harness, &format!("aries:{}", assign.executor)).await;
    assert_ne!(ex.current_task_id.as_deref(), Some(assign.task.as_str()));
}

/// The scheduler loads its jobs in one batch: the same records as one by one.
#[tokio::test]
async fn test_get_jobs_matches_get_job() {
    let harness = start_test_harness().await;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let sha = upload_png(&harness, 64, 64).await;
    let mut ids = Vec::new();
    for body in [
        json!({ "prompt": "a", "width": 1024, "height": 1024 }),
        json!({ "prompt": "b", "width": 1024, "height": 1024, "references": [{ "upload": sha }] }),
    ] {
        let resp = submit(&harness, body).await;
        assert_eq!(resp.status(), 201);
        ids.push(
            resp.json::<Value>().await.unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    let db = &harness.state.db;
    let mut asked = ids.clone();
    asked.push("no-such-job".to_string());
    let batch = db.get_jobs(&asked).await.unwrap();
    assert_eq!(batch.len(), 2);
    for job in batch {
        let one = db.get_job(&job.id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&job).unwrap(),
            serde_json::to_value(&one).unwrap()
        );
    }
    assert!(db.get_jobs(&[]).await.unwrap().is_empty());
}

/// `GET /api/jobs?limit=`: pages newest first with a cursor that neither skips nor repeats,
/// gallery filters, facets, and lists without the enhancer's reasoning.
#[tokio::test]
async fn test_jobs_pages_filters_and_facets() {
    let harness = start_test_harness().await;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let mut ids = Vec::new();
    for (i, (w, h)) in [
        (1024, 1024),
        (1376, 768),
        (1024, 1024),
        (768, 1376),
        (1024, 1024),
    ]
    .into_iter()
    .enumerate()
    {
        let resp = submit(&harness, json!({ "prompt": format!("prompt {i}{}", if i == 0 { " 50%_off" } else { "" }), "width": w, "height": h, "seed": 4200 + i })).await;
        assert_eq!(resp.status(), 201);
        ids.push(
            resp.json::<Value>().await.unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    let pool = harness.state.db.pool();
    // All but the newest are done; the first two are favorites; one has reasoning.
    for (i, id) in ids.iter().enumerate().take(4) {
        sqlx::query("UPDATE jobs SET status = 'done', favorite = ?, thinking = ? WHERE id = ?")
            .bind(i < 2)
            .bind(if i == 0 { Some("long reasoning") } else { None })
            .bind(id)
            .execute(pool)
            .await
            .unwrap();
    }
    let get = |query: String| {
        let url = format!("{}/api/jobs?{query}", harness.http_url);
        async move {
            reqwest::get(url)
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };
    let names = |page: &Value| -> Vec<String> {
        page["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|j| j["id"].as_str().unwrap().to_string())
            .collect()
    };

    // Two at a time, newest first, until `next` is null.
    let mut seen = Vec::new();
    let mut query = "limit=2".to_string();
    loop {
        let page = get(query.clone()).await;
        seen.extend(names(&page));
        match page["next"].as_str() {
            Some(next) => query = format!("limit=2&before={}", urlencode(next)),
            None => break,
        }
    }
    let newest_first: Vec<String> = ids.iter().rev().cloned().collect();
    assert_eq!(seen, newest_first);

    let done = get("limit=10&status=done".into()).await;
    assert_eq!(names(&done), newest_first[1..]);
    let favorites = get("limit=10&status=done&favorite=true".into()).await;
    assert_eq!(names(&favorites), [ids[1].clone(), ids[0].clone()]);
    let square = get("limit=10&status=done&aspect=1:1".into()).await;
    assert_eq!(names(&square), [ids[2].clone(), ids[0].clone()]);
    assert_eq!(
        names(&get("limit=10&q=PROMPT%203".into()).await),
        [ids[3].clone()]
    );
    assert_eq!(
        names(&get("limit=10&q=4201".into()).await),
        [ids[1].clone()]
    );
    // LIKE wildcards in the search are literal.
    assert_eq!(
        names(&get("limit=10&q=50%25_o".into()).await),
        [ids[0].clone()]
    );
    assert!(
        names(&get("limit=10&q=pr%253".into()).await).is_empty(),
        "as a wildcard it would match prompt 3"
    );

    // Lists say there is reasoning; the job itself has it.
    let first = done["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == ids[0].as_str())
        .unwrap()
        .clone();
    assert_eq!(first["has_thinking"], true);
    assert!(first["thinking"].is_null());
    let full: Value = reqwest::get(format!("{}/api/jobs/{}", harness.http_url, ids[0]))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(full["thinking"], "long reasoning");

    let facets: Value = reqwest::get(format!("{}/api/gallery/facets", harness.http_url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (facets["total"].as_u64(), facets["favorites"].as_u64()),
        (Some(4), Some(2))
    );
    let mut sizes: Vec<(u64, u64, u64)> = facets["sizes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["width"].as_u64().unwrap(),
                s["height"].as_u64().unwrap(),
                s["count"].as_u64().unwrap(),
            )
        })
        .collect();
    sizes.sort();
    assert_eq!(sizes, [(768, 1376, 1), (1024, 1024, 2), (1376, 768, 1)]);
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// The web app's paths serve index.html with 200 (shared links open); unknown API paths 404.
#[tokio::test]
async fn test_app_paths_serve_the_app() {
    let harness = start_test_harness().await;
    for path in ["/", "/gallery?q=cat&fav=1", "/images/job_01abc", "/status"] {
        let resp = reqwest::get(format!("{}{path}", harness.http_url))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{path}");
        assert!(
            resp.text().await.unwrap().contains("<title>app</title>"),
            "{path}"
        );
    }
    for path in ["/api/nope", "/agent/v1/nope"] {
        let resp = reqwest::get(format!("{}{path}", harness.http_url))
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "{path}");
    }
}

/// A cancel from the agent's side (failure class `cancelled`, no user cancel before it) ends
/// the job and its remaining tasks instead of leaving it running with nothing to run.
#[tokio::test]
async fn test_agent_side_cancel_ends_the_job() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(
        &harness,
        json!({ "prompt": "p", "width": 1024, "height": 1024 }),
    )
    .await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);

    db.assign_task(&task.id, "aries", "xpu0")
        .await
        .unwrap()
        .unwrap();
    db.fail_task_attempt(
        &task.id,
        1,
        AttemptFailure::studio("cancelled", "cancelled"),
    )
    .await
    .unwrap();

    let job = db.get_job(&task.job_id).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Cancelled);
    assert!(job.finished_at.is_some());
    assert!(
        job.tasks.iter().all(|t| t.status == TaskStatus::Cancelled),
        "{:?}",
        job.tasks
    );
    assert!(db.list_queued_tasks().await.unwrap().is_empty());
    assert!(executor(&harness, "aries:xpu0")
        .await
        .current_task_id
        .is_none());
}

/// The thumbnail route builds a file path from the id: only job ids get that far.
#[tokio::test]
async fn test_thumbnail_id_cannot_leave_the_thumbs_dir() {
    let harness = start_test_harness().await;
    std::fs::write(harness.temp_dir.join("secret.512.jpg"), b"secret").unwrap();
    for id in ["..%2Fsecret", "..%2F..%2Fsecret", "%2E%2E%2Fsecret"] {
        let resp = reqwest::get(format!(
            "{}/api/images/{id}/thumb?size=512",
            harness.http_url
        ))
        .await
        .unwrap();
        assert_eq!(resp.status(), 404, "{id}");
        assert!(!resp.text().await.unwrap().contains("secret\""), "{id}");
    }
}

/// What the mock providers received, and how the Gemini one answers.
#[derive(Default)]
struct MockProviders {
    anthropic: Vec<(HeaderMap, Value)>,
    gemini: Vec<(HeaderMap, Value)>,
    /// Gemini answers 429 (retry after 1 s) this many times first.
    gemini_busy: u32,
    /// The text every provider replies with.
    reply: String,
}

type Mock = Arc<std::sync::Mutex<MockProviders>>;

/// A stand-in for api.anthropic.com and the Gemini API on a local port.
async fn start_mock_providers(reply: &str) -> (Mock, String) {
    use axum::extract::State as S;
    use axum::http::StatusCode as Code;
    use axum::response::IntoResponse;
    let mock: Mock = Arc::new(std::sync::Mutex::new(MockProviders {
        reply: reply.to_string(),
        ..Default::default()
    }));
    let anthropic = |S(m): S<Mock>, headers: HeaderMap, axum::Json(body): axum::Json<Value>| async move {
        let mut m = m.lock().unwrap();
        m.anthropic.push((headers, body));
        axum::Json(json!({
            "content": [{"type": "text", "text": m.reply}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "cache_read_input_tokens": 2400, "output_tokens": 300},
        }))
    };
    let gemini = |S(m): S<Mock>, headers: HeaderMap, axum::Json(body): axum::Json<Value>| async move {
        let mut m = m.lock().unwrap();
        m.gemini.push((headers, body));
        if m.gemini_busy > 0 {
            m.gemini_busy -= 1;
            return (
                Code::TOO_MANY_REQUESTS,
                [("retry-after", "1")],
                axum::Json(json!({"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}})),
            )
                .into_response();
        }
        axum::Json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "weighing the edit", "thought": true},
                    {"text": m.reply},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 3000, "candidatesTokenCount": 300, "thoughtsTokenCount": 900},
        }))
        .into_response()
    };
    let app = axum::Router::new()
        .route("/v1/messages", axum::routing::post(anthropic))
        .route("/v1beta/models/:call", axum::routing::post(gemini))
        .with_state(mock.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (mock, url)
}

fn mock_cloud(base_url: &str) -> CloudConfig {
    let provider = |provider, api, model: &str| ProviderConfig {
        id: model.to_string(),
        provider,
        api,
        api_key: format!("{provider}-test-key"),
        model: model.to_string(),
        base_url: base_url.to_string(),
        max_concurrent: 4,
        rpm: 0,
        tpm: 0,
        thinking_level: "high".to_string(),
    };
    CloudConfig {
        providers: vec![
            provider("anthropic", Api::Anthropic, "claude-sonnet-5-5"),
            provider("google", Api::Gemini, "gemini-3.5-flash-lite"),
        ],
        system_prompt: None,
    }
}

/// Polls until the job is past its enhance (or ended).
async fn wait_job(
    harness: &TestHarness,
    job_id: &str,
    cond: impl Fn(&aries_image_studio::db::JobRecord) -> bool,
) -> aries_image_studio::db::JobRecord {
    for _ in 0..500 {
        if let Some(j) = harness.state.db.get_job(job_id).await.unwrap() {
            if cond(&j) {
                return j;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("job {job_id} never reached the expected state");
}

/// docs/CLOUD_ENHANCERS.md: off by default; when on, capabilities list the providers and
/// a job naming one is rewritten by Studio itself, not by the agent's enhancer, and encode
/// gets the cloud's prompt.
#[tokio::test]
async fn test_cloud_enhancer_rewrites_in_studio() {
    // Off: an enhancer named on a job is refused, and none is listed.
    let off = start_test_harness().await;
    let _off_link = connect_agent(&off, sample_register_msg("aries")).await;
    let caps: Value = reqwest::get(format!("{}/api/capabilities", off.http_url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(caps["cloud_enhancers"], json!([]));
    let r = submit(&off, json!({"prompt": "a cat", "width": 1024, "height": 1024, "enhance": true, "enhancer": "claude-sonnet-5-5"})).await;
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("cloud enhancers are off"));

    let (mock, url) = start_mock_providers(
        r#"Here it is: {"rewritten_prompt": "A tabby cat on a sunlit sill", "wh_ratio": "3:2"}"#,
    )
    .await;
    let harness = start_test_harness_with(mock_cloud(&url)).await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    let caps: Value = reqwest::get(format!("{}/api/capabilities", harness.http_url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(caps["cloud_enhancers"][0]["label"], "Claude Sonnet 5.5");
    assert_eq!(caps["cloud_enhancers"][1]["label"], "Gemini 3.5 Flash Lite");
    assert_eq!(caps["cloud_enhancers"][1]["provider"], "google");
    assert!(!caps.to_string().contains("test-key"));
    assert_eq!(
        submit(&harness, json!({"prompt": "a cat", "width": 1024, "height": 1024, "enhance": true, "enhancer": "nope"})).await.status(),
        422
    );

    let j: Value = submit(&harness, json!({"prompt": "a cat", "width": 1024, "height": 1024, "enhance": true, "enhancer": "claude-sonnet-5-5"}))
        .await.json().await.unwrap();
    assert_eq!(j["enhancer"], "claude-sonnet-5-5");
    let job_id = j["id"].as_str().unwrap();
    // The agent's first assignment is encode, with the rewritten prompt.
    let a = next_assign(&mut link).await;
    let Some(proto::assign::Params::Encode(p)) = &a.params else {
        panic!("expected encode, got {:?}", a.params)
    };
    assert_eq!(p.prompt, "A tabby cat on a sunlit sill");
    assert_eq!(p.prompt_source, "enhanced");

    let job = harness.state.db.get_job(job_id).await.unwrap().unwrap();
    assert_eq!(job.wh_ratio.as_deref(), Some("3:2"));
    let prov = &job.provenance["enhance"];
    assert_eq!(
        (prov.agent.as_str(), prov.executor.as_str()),
        ("cloud", "claude-sonnet-5-5")
    );
    assert_eq!(prov.model.id, "claude-sonnet-5-5");
    assert_eq!(prov.timings["generated_tokens"], 300.0);
    assert_eq!(prov.timings["prompt_tokens"], 2450.0);

    let m = mock.lock().unwrap();
    let (headers, body) = &m.anthropic[0];
    assert_eq!(headers["x-api-key"], "anthropic-test-key");
    assert_eq!(body["model"], "claude-sonnet-5-5");
    assert!(body["system"][0]["text"]
        .as_str()
        .unwrap()
        .contains("Image Prompt Rewriting Expert"));
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    let user = body["messages"][0]["content"][0]["text"].as_str().unwrap();
    assert_eq!(user, "a cat\n\nAspect ratio: 1:1");
}

/// A busy provider (429) is waited out within the attempt; references go along as images
/// with the edit instructions, and the reasoning is kept.
#[tokio::test]
async fn test_cloud_enhancer_with_references_waits_out_a_rate_limit() {
    let (mock, url) = start_mock_providers(r#"{"rewritten_prompt": "Put the hat from image 1 on the dog", "ratio_follow": "<image1>"}"#).await;
    mock.lock().unwrap().gemini_busy = 1;
    let harness = start_test_harness_with(mock_cloud(&url)).await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    let upload = upload_png(&harness, 1600, 1200).await;
    let j: Value = submit(
        &harness,
        json!({
            "prompt": "put the hat on the dog", "width": 1024, "height": 1024, "enhance": true,
            "enhancer": "gemini-3.5-flash-lite", "references": [{"upload": upload}],
        }),
    )
    .await
    .json()
    .await
    .unwrap();
    let job_id = j["id"].as_str().unwrap().to_string();
    let a = next_assign(&mut link).await;
    assert_eq!(a.stage, proto::Stage::Encode as i32);

    let job = harness.state.db.get_job(&job_id).await.unwrap().unwrap();
    assert_eq!(
        job.enhanced_prompt.as_deref(),
        Some("Put the hat from image 1 on the dog")
    );
    assert_eq!(job.ratio_follow.as_deref(), Some("<image1>"));
    assert_eq!(job.thinking.as_deref(), Some("weighing the edit"));
    assert_eq!(job.provenance["enhance"].timings["thought_tokens"], 900.0);
    let enhance = job.tasks.iter().find(|t| t.stage == "enhance").unwrap();
    assert_eq!(
        enhance.current_attempt, 1,
        "a 429 does not use up the attempt"
    );

    let m = mock.lock().unwrap();
    assert_eq!(m.gemini.len(), 2);
    let (headers, body) = &m.gemini[1];
    assert_eq!(headers["x-goog-api-key"], "google-test-key");
    assert!(body["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .unwrap()
        .contains("When images are attached"));
    let parts = body["contents"][0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["inlineData"]["mimeType"], "image/jpeg");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "high"
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["includeThoughts"],
        true
    );
}

/// A reply without the JSON fails the attempt (retried once), then the job; a retry can
/// switch to the built-in enhancer, which runs on the agent.
#[tokio::test]
async fn test_cloud_enhancer_failure_is_visible_and_retry_can_go_builtin() {
    let (mock, url) = start_mock_providers("I would rather not.").await;
    let harness = start_test_harness_with(mock_cloud(&url)).await;
    let mut link = connect_agent(&harness, sample_register_msg("aries")).await;
    let j: Value = submit(&harness, json!({"prompt": "a cat", "width": 1024, "height": 1024, "enhance": true, "enhancer": "claude-sonnet-5-5"}))
        .await.json().await.unwrap();
    let job_id = j["id"].as_str().unwrap().to_string();
    let job = wait_job(&harness, &job_id, |j| j.status == JobStatus::Failed).await;
    assert_eq!(job.error_class.as_deref(), Some("bad_output"));
    assert_eq!(
        job.error_detail.as_deref(),
        Some("the reply had no JSON object")
    );
    assert_eq!(mock.lock().unwrap().anthropic.len(), 2, "tried twice");

    let r = reqwest::Client::new()
        .post(format!("{}/api/jobs/{job_id}/retry", harness.http_url))
        .json(&json!({"enhancer": "builtin"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let a = next_assign(&mut link).await;
    assert_eq!(
        a.stage,
        proto::Stage::Enhance as i32,
        "the agent's enhancer now"
    );
    let job = harness.state.db.get_job(&job_id).await.unwrap().unwrap();
    assert!(job.enhancer.is_none());
}
