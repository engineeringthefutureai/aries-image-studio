use aries_image_studio::api::build_http_router;
use aries_image_studio::config::BackendConfig;
use aries_image_studio::db::{now_s, Database};
use aries_image_studio::grpc::AgentGatewayService;
use aries_image_studio::metrics::{AgentLiveMetrics, ExecutorLiveMetrics};
use aries_image_studio::proto;
use aries_image_studio::proto::agent_gateway_client::AgentGatewayClient;
use aries_image_studio::proto::agent_gateway_server::AgentGatewayServer;
use aries_image_studio::scheduler::{run_scheduler_tick, AppState};
use aries_image_studio::storage::{read_png_text_chunk, BlobStore};
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
    let temp_dir = std::env::temp_dir().join(format!("aries_v3_test_{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&temp_dir).unwrap();

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
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Encode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 0,
                        modes: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Denoise as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Decode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![proto::DecodeMode::Untiled as i32],
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
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Denoise as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![],
                    },
                    proto::StageCapability {
                        stage: proto::Stage::Decode as i32,
                        models: vec!["Qwen/Qwen-Image-2.1".to_string()],
                        max_px: 2048 * 2048,
                        modes: vec![proto::DecodeMode::Untiled as i32],
                    },
                ],
            },
        ],
        limits: vec![proto::Limit {
            stage: proto::Stage::Decode as i32,
            above_px: 1024 * 1024,
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
    assert!(res.is_err(), "Registration with unknown Requirement.kind must fail");
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
            "enhance": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp_ok.status(), 201);
    let created_job: Value = resp_ok.json().await.unwrap();
    let job_id = created_job["id"].as_str().unwrap().to_string();

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
            proto::Stage::Decode => build_png_bytes(1024, 1024),
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
    let img_bytes = http
        .get(format!("{}/api/images/{job_id}", harness.http_url))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let chunk_text = read_png_text_chunk(&img_bytes, "aries-image-studio")
        .unwrap()
        .expect("PNG must contain aries-image-studio tEXt chunk");
    let chunk_json: Value = serde_json::from_str(&chunk_text).unwrap();
    assert_eq!(chunk_json["job_id"], job_id);
    assert_eq!(
        chunk_json["provenance"]["decode"]["torch_version"],
        "2.6.0+xpu"
    );

    // Fetch WebP thumbnail
    let thumb_resp = http
        .get(format!("{}/api/images/{job_id}/thumb", harness.http_url))
        .send()
        .await
        .unwrap();
    assert_eq!(thumb_resp.status(), 200);
    assert_eq!(
        thumb_resp.headers().get("content-type").unwrap(),
        "image/webp"
    );

    // Verify intermediate artifacts were garbage collected while final artifact remains
    let (inter_bytes, final_bytes) = harness.state.db.artifact_bytes_by_kind().await.unwrap();
    assert_eq!(inter_bytes, 0, "Intermediate artifacts must be deleted on job completion");
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
            ("negative_prompt_embeds", "BF16", vec![1, 16, 64], &[0u8; 2048]),
            ("negative_prompt_embeds_mask", "I64", vec![1, 16], &[1u8; 128]),
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
        .post(format!("{}/api/executors/aries:xpu1/lend", harness.http_url))
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
    assert!(matches!(welcome.msg, Some(proto::backend_message::Msg::Welcome(_))));
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

    let bad = submit(&harness, json!({ "id": "../etc", "prompt": "x", "width": 1024, "height": 1024 })).await;
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
    let full = [(2048, 2048), (2400, 1792), (1792, 2400), (2528, 1696), (1696, 2528), (2752, 1536), (1536, 2752)];
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
    let resp = submit(&harness, json!({ "prompt": "p", "width": 1264, "height": 848 })).await;
    assert_eq!(resp.status(), 422);
    let caps = aries_image_studio::scheduler::compute_capabilities(&harness.state.db).await.unwrap();
    assert!(caps.sizes.iter().all(|s| s.servable_ever), "{:?}", caps.sizes);
}

/// The agent sends READY right after Done; by then the backend may have assigned the next
/// task. That READY must not reopen the executor for a second assignment.
#[tokio::test]
async fn test_ready_after_done_does_not_double_assign() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    for i in 0..3 {
        let resp = submit(&harness, json!({ "prompt": format!("p{i}"), "width": 1024, "height": 1024 })).await;
        assert_eq!(resp.status(), 201);
    }
    let encode_tasks: Vec<_> = db.list_queued_tasks().await.unwrap();
    assert_eq!(encode_tasks.len(), 3);

    // Pretend an encode worker on xpu0 is ready and the first task got assigned to it.
    db.update_executor_state("aries", "xpu0", "ready", Some("encode"), Some(1), None).await.unwrap();
    let first = &encode_tasks[0];
    db.assign_task(&first.id, "aries", "xpu0").await.unwrap().unwrap();
    assert_eq!(executor(&harness, "aries:xpu0").await.state, "busy");

    // Late READY (the one that followed the previous Done) arrives: still busy.
    db.update_executor_state("aries", "xpu0", "ready", Some("encode"), Some(1), None).await.unwrap();
    let ex = executor(&harness, "aries:xpu0").await;
    assert_eq!(ex.state, "busy");
    assert_eq!(ex.current_task_id.as_deref(), Some(first.id.as_str()));
}

#[tokio::test]
async fn test_stale_rejection_is_ignored() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(&harness, json!({ "prompt": "p", "width": 1024, "height": 1024 })).await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);

    db.assign_task(&task.id, "aries", "xpu0").await.unwrap().unwrap();
    db.fail_task_attempt(&task.id, 1, "python_error", "boom", 1, 0, "").await.unwrap();
    // A rejection for attempt 1 arrives after the task was requeued: ignore it.
    db.handle_rejected_assignment(&task.id, 1, "late").await.unwrap();

    let t = db.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!((t.status.as_str(), t.attempts), ("queued", 1));
    assert!(executor(&harness, "aries:xpu0").await.disabled_reason.is_none());
}

/// An agent that restarts within reconnect_grace registers without the task it had: the
/// backend requeues it instead of waiting forever. A task it still runs stays assigned.
#[tokio::test]
async fn test_reconnect_requeues_tasks_the_agent_no_longer_has() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let (tx, inbound) = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(&harness, json!({ "prompt": "a", "width": 1024, "height": 1024 })).await;
    submit(&harness, json!({ "prompt": "b", "width": 1024, "height": 1024 })).await;
    let queued = db.list_queued_tasks().await.unwrap();
    let (lost, kept) = (&queued[0], &queued[1]);
    db.assign_task(&lost.id, "aries", "xpu0").await.unwrap().unwrap();
    db.assign_task(&kept.id, "aries", "xpu1").await.unwrap().unwrap();

    drop(tx);
    drop(inbound);
    let mut reg = sample_register_msg("aries");
    reg.running = vec![proto::RunningTask { task: kept.id.clone(), attempt: 1, executor: "xpu1".into() }];
    let _agent = connect_agent(&harness, reg).await;

    let t = wait_task(&harness, &lost.id, |t| t.status == "queued").await;
    assert_eq!(t.attempts, 1, "the lost attempt counts");
    let k = db.get_task(&kept.id).await.unwrap().unwrap();
    assert_eq!(k.status, "assigned");
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!((ex1.state.as_str(), ex1.current_task_id.as_deref()), ("busy", Some(kept.id.as_str())));
}

#[tokio::test]
async fn test_enhance_progress_keeps_sub_stage_and_stats() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let (tx, _inbound) = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(&harness, json!({ "prompt": "p", "width": 1024, "height": 1024, "enhance": true })).await;
    let task = db.list_queued_tasks().await.unwrap().remove(0);
    assert_eq!(task.stage, "enhance");
    db.assign_task(&task.id, "aries", "xpu0").await.unwrap().unwrap();

    let stats: HashMap<String, f64> =
        [("generated_tokens".to_string(), 512.0), ("max_new_tokens".to_string(), 2048.0)].into();
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
        submit(&harness, json!({ "prompt": format!("p{i}"), "width": 1024, "height": 1024 })).await;
    }
    // Fast-forward both jobs to "encode done, denoise queued".
    for q in [
        "UPDATE jobs SET model_id = 'Qwen/Qwen-Image-2.1', model_rev = 'rev123'",
        "UPDATE tasks SET status = 'done' WHERE stage = 'encode'",
        "UPDATE tasks SET status = 'queued', queued_at = 1 WHERE stage = 'denoise'",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    db.update_executor_state("aries", "xpu0", "ready", Some("denoise"), Some(1), None).await.unwrap();
    db.update_executor_state("aries", "xpu1", "idle", None, None, None).await.unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    assert_eq!(executor(&harness, "aries:xpu0").await.state, "busy");
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!((ex1.state.as_str(), ex1.current_stage.as_deref()), ("starting", Some("denoise")));
}

/// A retried (or rejected) task gets a fresh attempt number: reusing one collided with the
/// first attempt's row and stalled the scheduler.
#[tokio::test]
async fn test_retry_and_rejection_use_new_attempt_numbers() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    let job: Value = submit(&harness, json!({ "id": "job-retry", "prompt": "p", "width": 1024, "height": 1024 }))
        .await
        .json()
        .await
        .unwrap();
    let task_id = job["tasks"][0]["id"].as_str().unwrap().to_string();

    for attempt in 1..=2 {
        db.assign_task(&task_id, "aries", "xpu0").await.unwrap().unwrap();
        db.fail_task_attempt(&task_id, attempt, "python_error", "boom", 1, 0, "").await.unwrap();
    }
    assert_eq!(db.get_job("job-retry").await.unwrap().unwrap().status, "failed");

    db.retry_job("job-retry").await.unwrap();
    let (t, _, attempt) = db.assign_task(&task_id, "aries", "xpu0").await.unwrap().unwrap();
    assert_eq!((attempt, t.attempts), (3, 1), "new attempt number, fresh retry budget");

    db.handle_rejected_assignment(&task_id, 3, "nope").await.unwrap();
    db.enable_executor("aries:xpu0").await.unwrap();
    let (t, _, attempt) = db.assign_task(&task_id, "aries", "xpu1").await.unwrap().unwrap();
    assert_eq!((attempt, t.attempts), (4, 1), "a rejection gives the retry back");
}

/// Seen live: xpu1 finished its encodes and sat idle on encode for a minute and a half,
/// because a third job was still enhancing on xpu0, while two denoise tasks waited. A
/// worker stays only while its own stage has queued work.
#[tokio::test]
async fn test_idle_worker_switches_to_waiting_stage() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(&harness, json!({ "id": "enh", "prompt": "e", "width": 1024, "height": 1024, "enhance": true })).await;
    for id in ["d1", "d2"] {
        submit(&harness, json!({ "id": id, "prompt": id, "width": 1024, "height": 1024 })).await;
    }
    for q in [
        "UPDATE jobs SET model_id = 'Qwen/Qwen-Image-2.1', model_rev = 'rev123' WHERE id IN ('d1', 'd2')",
        "UPDATE tasks SET status = 'done' WHERE stage = 'encode' AND job_id IN ('d1', 'd2')",
        "UPDATE tasks SET status = 'queued', queued_at = (julianday('now') - 2440587.5) * 86400.0 WHERE stage = 'denoise' AND job_id IN ('d1', 'd2')",
    ] {
        sqlx::query(q).execute(db.pool()).await.unwrap();
    }
    let enhance = db.list_queued_tasks().await.unwrap().into_iter().find(|t| t.stage == "enhance").unwrap();
    db.update_executor_state("aries", "xpu0", "ready", Some("enhance"), Some(1), None).await.unwrap();
    db.assign_task(&enhance.id, "aries", "xpu0").await.unwrap().unwrap();
    db.update_executor_state("aries", "xpu1", "ready", Some("encode"), Some(2), None).await.unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_eq!((ex1.state.as_str(), ex1.current_stage.as_deref()), ("starting", Some("denoise")));
}

/// No switch when the other stage's queue is already covered by a free worker.
#[tokio::test]
async fn test_idle_worker_does_not_switch_to_covered_stage() {
    let harness = start_test_harness().await;
    let db = &harness.state.db;
    let _agent = connect_agent(&harness, sample_register_msg("aries")).await;
    submit(&harness, json!({ "id": "d1", "prompt": "p", "width": 1024, "height": 1024 })).await;
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
    db.mark_executor_starting("aries", "xpu0", "denoise", &model).await.unwrap();
    db.update_executor_state("aries", "xpu1", "ready", Some("encode"), Some(2), None).await.unwrap();

    run_scheduler_tick(&harness.state).await.unwrap();

    let ex0 = executor(&harness, "aries:xpu0").await;
    let ex1 = executor(&harness, "aries:xpu1").await;
    assert_ne!(ex1.current_stage.as_deref(), Some("denoise"), "xpu0={ex0:?}");
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
    let job: Value = submit(&harness, json!({ "id": "j1", "prompt": "p", "width": 1024, "height": 1024, "meta": meta }))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(job["meta"], meta);
    assert_eq!(job["favorite"], false);
    let big = json!({ "x": "y".repeat(5000) });
    assert_eq!(submit(&harness, json!({ "id": "j2", "prompt": "p", "width": 1024, "height": 1024, "meta": big })).await.status(), 400);

    let listed: Value = http.get(url("/api/jobs")).send().await.unwrap().json().await.unwrap();
    let v0 = listed["version"].as_i64().unwrap();

    let fav: Value = http.patch(url("/api/jobs/j1")).json(&json!({ "favorite": true })).send().await.unwrap().json().await.unwrap();
    assert_eq!(fav["favorite"], true);
    let changed: Value = http.get(url(&format!("/api/jobs?changed_since={v0}"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(changed["jobs"][0]["id"], "j1");

    let v1 = changed["version"].as_i64().unwrap();
    assert_eq!(http.delete(url("/api/jobs/j1")).send().await.unwrap().status(), 204);
    let after: Value = http.get(url(&format!("/api/jobs?changed_since={v1}"))).send().await.unwrap().json().await.unwrap();
    assert_eq!(after["deleted"], json!(["j1"]));
    assert_eq!(after["jobs"], json!([]));

    let empty: Value = http.get(url("/api/preferences")).send().await.unwrap().json().await.unwrap();
    assert_eq!(empty, json!({}));
    let prefs = json!({ "theme": "dark", "steps": 20 });
    assert_eq!(http.put(url("/api/preferences")).json(&prefs).send().await.unwrap().status(), 200);
    let got: Value = http.get(url("/api/preferences")).send().await.unwrap().json().await.unwrap();
    assert_eq!(got, prefs);
    assert_eq!(http.put(url("/api/preferences")).json(&json!([1, 2])).send().await.unwrap().status(), 400);
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
    assert_eq!(enhance_request("a sorcerer \n", 848, 1264), "a sorcerer\n\nAspect ratio: 2:3");
}
