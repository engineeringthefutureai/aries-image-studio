//! The backend's upload checks require exactly the stage artifact contract in
//! proto/aries/agent/v1/artifacts.json; backend/tests/test_contract.py checks the Python
//! workers write it.

use aries_image_studio::storage::{
    validate_embeds_safetensors, validate_enhanced_json, validate_latent_safetensors,
};
use serde_json::{json, Value};

fn contract() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../proto/aries/agent/v1/artifacts.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn names(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

/// A safetensors file with one 2-byte BF16 tensor per name and the given metadata.
fn safetensors(tensors: &[String], metadata: &[(String, String)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        Value::Object(
            metadata
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect(),
        ),
    );
    for (i, name) in tensors.iter().enumerate() {
        header.insert(
            name.clone(),
            json!({ "dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2] }),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.resize(out.len() + tensors.len() * 2, 0);
    out
}

fn meta_values(keys: &[String]) -> Vec<(String, String)> {
    keys.iter()
        .map(|k| {
            let v = match k.as_str() {
                "width" | "height" => "1024",
                "seed" => "7",
                "steps" => "4",
                _ => "x",
            };
            (k.clone(), v.to_string())
        })
        .collect()
}

#[test]
fn embeds_contract_is_exactly_what_the_backend_requires() {
    let spec = &contract()["embeds.safetensors"];
    let tensors = names(&spec["tensors"]);
    let cfg_tensors = names(&spec["tensors_when_cfg"]);
    let metadata = meta_values(&names(&spec["metadata"]));

    // Required tensors alone are enough; optional ones (masks) may be added.
    assert!(validate_embeds_safetensors(&safetensors(&tensors, &metadata), false, 0).is_ok());
    let with_optional: Vec<String> = tensors
        .iter()
        .chain(&names(&spec["optional_tensors"]))
        .cloned()
        .collect();
    assert!(validate_embeds_safetensors(&safetensors(&with_optional, &metadata), false, 0).is_ok());
    let with_cfg: Vec<String> = tensors.iter().chain(&cfg_tensors).cloned().collect();
    assert!(validate_embeds_safetensors(&safetensors(&with_cfg, &metadata), true, 0).is_ok());
    assert!(validate_embeds_safetensors(&safetensors(&tensors, &metadata), true, 0).is_err());

    for skip in &with_cfg {
        let fewer: Vec<String> = with_cfg.iter().filter(|t| *t != skip).cloned().collect();
        assert!(
            validate_embeds_safetensors(&safetensors(&fewer, &metadata), true, 0).is_err(),
            "missing tensor {skip} accepted"
        );
    }
    for skip in 0..metadata.len() {
        let mut fewer = metadata.clone();
        fewer.remove(skip);
        assert!(validate_embeds_safetensors(&safetensors(&tensors, &fewer), false, 0).is_err());
    }
}

#[test]
fn latent_contract_is_exactly_what_the_backend_requires() {
    let spec = &contract()["latent.safetensors"];
    let tensors = names(&spec["tensors"]);
    let metadata = meta_values(&names(&spec["metadata"]));

    assert!(
        validate_latent_safetensors(&safetensors(&tensors, &metadata), Some(1024), Some(1024))
            .is_ok()
    );
    assert!(validate_latent_safetensors(&safetensors(&[], &metadata), None, None).is_err());
    for skip in 0..metadata.len() {
        let mut fewer = metadata.clone();
        fewer.remove(skip);
        assert!(validate_latent_safetensors(&safetensors(&tensors, &fewer), None, None).is_err());
    }
}

#[test]
fn enhanced_json_contract_is_what_the_backend_reads() {
    let spec = &contract()["enhanced.json"];
    let mut doc = serde_json::Map::new();
    for f in names(&spec["fields"])
        .iter()
        .chain(&names(&spec["optional_fields"]))
    {
        doc.insert(f.clone(), Value::String("x".into()));
    }
    let parsed = validate_enhanced_json(&serde_json::to_vec(&doc).unwrap()).unwrap();
    assert_eq!(parsed.thinking, "x");
    // PE-I2I's `ratio_follow` names an input image; anything else is dropped.
    assert!(names(&spec["optional_fields"]).contains(&"ratio_follow".to_string()));
    assert_eq!(parsed.ratio_follow, "");
    doc.insert("ratio_follow".into(), Value::String("<image2>".into()));
    let parsed = validate_enhanced_json(&serde_json::to_vec(&doc).unwrap()).unwrap();
    assert_eq!(parsed.ratio_follow, "<image2>");
    for f in names(&spec["fields"]) {
        let mut fewer = doc.clone();
        fewer.remove(&f);
        assert!(validate_enhanced_json(&serde_json::to_vec(&fewer).unwrap()).is_err());
    }
}

#[test]
fn backend_parses_the_agent_metric_names() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../proto/aries/agent/v1/agent_metrics.json"
    );
    let m: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let name = |k: &str| m[k].as_str().unwrap().to_string();
    let label = name("executor_label");
    let text = format!(
        "# HELP x\n{} 2.0e+10\n{}{{{label}=\"xpu0\",device=\"0000:04:00.0\"}} 1.5e+10\n\
         {}{{{label}=\"xpu0\",device=\"0000:04:00.0\"}} 2.5e+10\n{}{{{label}=\"xpu0\",device=\"0000:04:00.0\"}} 2.6e+08\n",
        name("host_ram_available_bytes"),
        name("executor_mem_used_bytes"),
        name("executor_mem_total_bytes"),
        name("executor_mem_overflow_bytes"),
    );
    let parsed = aries_image_studio::metrics::parse_agent_prometheus_text(&text, 1.0);
    assert_eq!(parsed.host_ram_available_bytes, 20_000_000_000);
    let ex = &parsed.executors["xpu0"];
    assert_eq!(ex.mem_used_bytes, 15_000_000_000);
    assert_eq!(ex.mem_total_bytes, 25_000_000_000);
    assert_eq!(ex.mem_available_bytes, 10_000_000_000);
    assert_eq!(ex.overflow_host_bytes, 260_000_000);
}

#[test]
fn agent_metrics_are_reexported_once_per_family_with_an_agent_label() {
    let scrape = "# HELP aries_host_ram_available_bytes Available host RAM\n\
                  # TYPE aries_host_ram_available_bytes gauge\n\
                  aries_host_ram_available_bytes 2.0e+10\n\
                  # HELP aries_task_seconds Task duration\n\
                  # TYPE aries_task_seconds histogram\n\
                  aries_task_seconds_bucket{executor=\"xpu0\",le=\"1.0\"} 3.0\n\
                  aries_task_seconds_count{executor=\"xpu0\"} 3.0\n\
                  # HELP aries_agent_connected Connected\n\
                  # TYPE aries_agent_connected gauge\n\
                  aries_agent_connected 1.0\n";
    let out =
        aries_image_studio::metrics::reexport_agent_metrics(&[("a1", scrape), ("a2", scrape)]);
    assert_eq!(
        out.matches("# TYPE aries_host_ram_available_bytes gauge")
            .count(),
        1
    );
    assert!(out.contains("aries_host_ram_available_bytes{agent=\"a1\"} 2.0e+10"));
    assert!(out.contains("aries_host_ram_available_bytes{agent=\"a2\"} 2.0e+10"));
    assert!(
        out.contains("aries_task_seconds_bucket{agent=\"a2\",executor=\"xpu0\",le=\"1.0\"} 3.0")
    );
    assert!(
        !out.contains("aries_agent_connected"),
        "the backend exports its own"
    );
}

/// A latents file shaped like `ref<i>.safetensors`: one (1, tokens, 64) tensor.
fn latents_file(tensor: &str, tokens: usize, metadata: &[(String, String)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        Value::Object(
            metadata
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect(),
        ),
    );
    header.insert(
        tensor.into(),
        json!({ "dtype": "BF16", "shape": [1, tokens, 64], "data_offsets": [0, 2] }),
    );
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.resize(out.len() + 2, 0);
    out
}

#[test]
fn reference_latents_contract_is_exactly_what_the_backend_requires() {
    let spec = &contract()["ref<i>.safetensors"];
    let (tensor, keys) = (names(&spec["tensors"]).remove(0), names(&spec["metadata"]));
    let sha = "a".repeat(64);
    let meta = |skip: &str, overrides: &[(&str, &str)]| -> Vec<(String, String)> {
        keys.iter()
            .filter(|k| k.as_str() != skip)
            .map(|k| {
                let v = match k.as_str() {
                    "width" => "1184".to_string(),
                    "height" => "896".to_string(),
                    "reference_res" => "1024".to_string(),
                    "source_sha256" => sha.clone(),
                    _ => "x".to_string(),
                };
                let v = overrides
                    .iter()
                    .find(|(o, _)| o == k)
                    .map(|(_, v)| v.to_string())
                    .unwrap_or(v);
                (k.clone(), v)
            })
            .collect()
    };
    let tokens = (1184 / 16) * (896 / 16);
    use aries_image_studio::storage::validate_reference_latents;
    let (_, w, h) =
        validate_reference_latents(&latents_file(&tensor, tokens, &meta("", &[])), &sha, 1024)
            .unwrap();
    assert_eq!((w, h), (1184, 896));
    for k in &keys {
        assert!(
            validate_reference_latents(&latents_file(&tensor, tokens, &meta(k, &[])), &sha, 1024)
                .is_err(),
            "metadata '{k}' must be required"
        );
    }
    // Another image's latents, another resolution, or a size that does not match the tokens.
    let other = "b".repeat(64);
    assert!(validate_reference_latents(
        &latents_file(&tensor, tokens, &meta("", &[])),
        &other,
        1024
    )
    .is_err());
    assert!(
        validate_reference_latents(&latents_file(&tensor, tokens, &meta("", &[])), &sha, 512)
            .is_err()
    );
    assert!(validate_reference_latents(
        &latents_file(&tensor, tokens + 1, &meta("", &[])),
        &sha,
        1024
    )
    .is_err());
    assert!(
        validate_reference_latents(&latents_file("other", tokens, &meta("", &[])), &sha, 1024)
            .is_err()
    );
}

#[test]
fn embeds_with_references_need_the_mask_and_the_sizes() {
    let spec = &contract()["embeds.safetensors"];
    let mut tensors = names(&spec["tensors"]);
    tensors.extend(names(&spec["tensors_when_references"]));
    let mut metadata = meta_values(&names(&spec["metadata"]));
    for k in names(&spec["metadata_when_references"]) {
        assert_eq!(k, "reference_sizes");
        metadata.push((k, "[[1184, 896], [1024, 1024]]".to_string()));
    }
    assert!(validate_embeds_safetensors(&safetensors(&tensors, &metadata), false, 2).is_ok());
    assert!(validate_embeds_safetensors(&safetensors(&tensors, &metadata), false, 3).is_err());
    let no_mask: Vec<String> = tensors
        .iter()
        .filter(|t| *t != "image_pad_mask")
        .cloned()
        .collect();
    assert!(validate_embeds_safetensors(&safetensors(&no_mask, &metadata), false, 2).is_err());
    let no_sizes: Vec<_> = metadata
        .iter()
        .filter(|(k, _)| k != "reference_sizes")
        .cloned()
        .collect();
    assert!(validate_embeds_safetensors(&safetensors(&tensors, &no_sizes), false, 2).is_err());
    // Text-only jobs need neither.
    assert!(validate_embeds_safetensors(&safetensors(&no_mask, &no_sizes), false, 0).is_ok());
}
