use anyhow::{anyhow, bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::Cursor;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Thumbnail sizes (longer side, px). Grid tiles are 160–430 CSS px wide: 512 covers them
/// on ordinary screens, 768 and 1024 on high-density ones (1024 is also the viewer's
/// placeholder). The browser picks via `srcset`.
pub const THUMB_SIZES: [u32; 3] = [512, 768, 1024];
const THUMB_JPEG_QUALITY: u8 = 82;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedJson {
    pub rewritten_prompt: String,
    #[serde(default)]
    pub wh_ratio: String,
    /// The rewriter's reasoning, kept with the job for the Inspector.
    #[serde(default)]
    pub thinking: String,
}

#[derive(Debug, Clone)]
pub struct BlobStore {
    data_dir: PathBuf,
}

impl BlobStore {
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(data_dir.join("blobs"))?;
        fs::create_dir_all(data_dir.join("thumbs"))?;
        fs::create_dir_all(data_dir.join("tmp"))?;
        // WebP thumbnails from before 0.3.1; JPEGs replace them on first request.
        if let Ok(entries) = fs::read_dir(data_dir.join("thumbs")) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "webp") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        Ok(Self { data_dir })
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        let prefix = if sha256.len() >= 2 {
            &sha256[0..2]
        } else {
            "00"
        };
        self.data_dir.join("blobs").join(prefix).join(sha256)
    }

    /// Thumbnail of `job_id` whose longer side is `size` (one of [`THUMB_SIZES`]).
    pub fn thumb_path(&self, job_id: &str, size: u32) -> PathBuf {
        self.data_dir.join("thumbs").join(format!("{job_id}.{size}.jpg"))
    }

    /// Removes every thumbnail of `job_id`.
    pub fn delete_thumbnails(&self, job_id: &str) {
        for size in THUMB_SIZES {
            let _ = fs::remove_file(self.thumb_path(job_id, size));
        }
    }

    /// Computes SHA-256 hex digest of `bytes` and writes atomically to `blobs/<sha[0..2]>/<sha>`.
    pub fn put_blob(&self, bytes: &[u8]) -> Result<(String, u64)> {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let sha256 = hex::encode(hasher.finalize());
        let dest = self.blob_path(&sha256);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        if !dest.exists() {
            let tmp = self
                .data_dir
                .join("tmp")
                .join(format!("{}.tmp.{}", sha256, ulid::Ulid::new()));
            fs::write(&tmp, bytes)?;
            fs::rename(&tmp, &dest)?;
        }
        Ok((sha256, bytes.len() as u64))
    }

    pub fn get_blob(&self, sha256: &str) -> Result<Vec<u8>> {
        let path = self.blob_path(sha256);
        fs::read(&path).with_context(|| format!("failed to read blob {sha256}"))
    }

    pub fn delete_blob(&self, sha256: &str) -> Result<()> {
        let path = self.blob_path(sha256);
        if path.exists() {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    /// Writes a JPEG thumbnail at each of [`THUMB_SIZES`] (longer side, never upscaled).
    /// CPU-heavy (decodes the full PNG): call it from a blocking task.
    pub fn write_thumbnails(&self, job_id: &str, png_bytes: &[u8]) -> Result<()> {
        let img = image::load_from_memory_with_format(png_bytes, ImageFormat::Png)
            .context("failed to decode PNG for thumbnail")?;
        // Largest first, each made from the previous one: cheaper, and the same quality.
        let mut source = img;
        for size in THUMB_SIZES.iter().rev().copied() {
            let thumb = if source.width().max(source.height()) > size {
                source.resize(size, size, FilterType::Lanczos3)
            } else {
                source.clone()
            };
            let mut buf = Cursor::new(Vec::new());
            JpegEncoder::new_with_quality(&mut buf, THUMB_JPEG_QUALITY)
                .encode_image(&thumb.to_rgb8())
                .context("failed to encode JPEG thumbnail")?;
            let tmp = self
                .data_dir
                .join("tmp")
                .join(format!("thumb.{job_id}.{size}.{}.jpg", ulid::Ulid::new()));
            fs::write(&tmp, buf.into_inner())?;
            fs::rename(&tmp, self.thumb_path(job_id, size))?;
            source = thumb;
        }
        Ok(())
    }

    /// Returns disk usage percentage (0.0..=100.0) of the filesystem containing `data_dir`.
    pub fn disk_usage_pct(&self) -> Result<f64> {
        let c_path = CString::new(self.data_dir.as_os_str().as_bytes())
            .context("invalid data_dir path for statvfs")?;
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
                bail!("statvfs failed");
            }
            if stat.f_blocks == 0 {
                return Ok(0.0);
            }
            let total = stat.f_blocks as f64;
            let avail = stat.f_bavail as f64;
            let used = (total - avail).max(0.0);
            Ok((used / total) * 100.0)
        }
    }
}

/// Parses and validates a safetensors binary buffer without loading tensor data into tensors.
pub fn parse_safetensors_header(bytes: &[u8]) -> Result<SafetensorsHeader> {
    if bytes.len() < 8 {
        bail!("safetensors buffer too short (< 8 bytes)");
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if header_len == 0 || header_len > 10_000_000 {
        bail!("invalid safetensors header length: {header_len}");
    }
    if bytes.len() < 8 + header_len {
        bail!(
            "safetensors buffer truncated: header needs {} bytes, got {}",
            8 + header_len,
            bytes.len()
        );
    }
    let header_json: Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .context("invalid JSON in safetensors header")?;
    let obj = header_json
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header must be a JSON object"))?;

    let data_section_len = bytes.len() - 8 - header_len;
    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();

    for (key, val) in obj {
        if key == "__metadata__" {
            let meta_obj = val
                .as_object()
                .ok_or_else(|| anyhow!("__metadata__ must be an object"))?;
            for (mk, mv) in meta_obj {
                let s = mv
                    .as_str()
                    .ok_or_else(|| anyhow!("__metadata__.{mk} must be a string"))?;
                metadata.insert(mk.clone(), s.to_string());
            }
        } else {
            let info: TensorInfo = serde_json::from_value(val.clone())
                .with_context(|| format!("invalid tensor descriptor for {key}"))?;
            if info.data_offsets[0] > info.data_offsets[1]
                || info.data_offsets[1] > data_section_len
            {
                bail!(
                    "tensor {key} data_offsets {:?} out of bounds (data_section_len={data_section_len})",
                    info.data_offsets
                );
            }
            if info.dtype.is_empty() {
                bail!("tensor {key} has empty dtype");
            }
            tensors.insert(key.clone(), info);
        }
    }

    if tensors.is_empty() {
        bail!("safetensors file contains no tensors");
    }

    Ok(SafetensorsHeader { tensors, metadata })
}

pub fn validate_enhanced_json(bytes: &[u8]) -> Result<EnhancedJson> {
    let parsed: EnhancedJson =
        serde_json::from_slice(bytes).context("invalid enhanced.json payload")?;
    if parsed.rewritten_prompt.trim().is_empty() {
        bail!("enhanced.json rewritten_prompt cannot be empty");
    }
    Ok(parsed)
}

pub fn validate_embeds_safetensors(bytes: &[u8], require_negative: bool) -> Result<SafetensorsHeader> {
    let header = parse_safetensors_header(bytes)?;
    // Masks are optional: the pipeline returns None for them when nothing is padded.
    for req_tensor in ["prompt_embeds"] {
        let t = header
            .tensors
            .get(req_tensor)
            .ok_or_else(|| anyhow!("embeds.safetensors missing tensor '{req_tensor}'"))?;
        if t.shape.is_empty() {
            bail!("embeds.safetensors tensor '{req_tensor}' has empty shape");
        }
    }
    if require_negative {
        for req_tensor in ["negative_prompt_embeds"] {
            if !header.tensors.contains_key(req_tensor) {
                bail!(
                    "embeds.safetensors missing '{req_tensor}' required when true_cfg_scale > 1"
                );
            }
        }
    }
    for req_meta in ["prompt", "prompt_source", "model", "rev", "dtype"] {
        let v = header
            .metadata
            .get(req_meta)
            .ok_or_else(|| anyhow!("embeds.safetensors missing metadata '{req_meta}'"))?;
        if v.is_empty() {
            bail!("embeds.safetensors metadata '{req_meta}' is empty");
        }
    }
    Ok(header)
}

pub fn validate_latent_safetensors(
    bytes: &[u8],
    expected_width: Option<u32>,
    expected_height: Option<u32>,
) -> Result<SafetensorsHeader> {
    let header = parse_safetensors_header(bytes)?;
    let t = header
        .tensors
        .get("latent")
        .ok_or_else(|| anyhow!("latent.safetensors missing tensor 'latent'"))?;
    if t.shape.is_empty() {
        bail!("latent.safetensors tensor 'latent' has empty shape");
    }
    for req_meta in ["model", "rev", "dtype", "width", "height", "seed", "steps"] {
        let v = header
            .metadata
            .get(req_meta)
            .ok_or_else(|| anyhow!("latent.safetensors missing metadata '{req_meta}'"))?;
        if v.is_empty() {
            bail!("latent.safetensors metadata '{req_meta}' is empty");
        }
    }
    if let Some(w) = expected_width {
        if header.metadata.get("width").map(|s| s.as_str()) != Some(&w.to_string()) {
            bail!(
                "latent.safetensors width mismatch: expected {w}, got {:?}",
                header.metadata.get("width")
            );
        }
    }
    if let Some(h) = expected_height {
        if header.metadata.get("height").map(|s| s.as_str()) != Some(&h.to_string()) {
            bail!(
                "latent.safetensors height mismatch: expected {h}, got {:?}",
                header.metadata.get("height")
            );
        }
    }
    Ok(header)
}

pub fn validate_png_dimensions(
    bytes: &[u8],
    expected_width: Option<u32>,
    expected_height: Option<u32>,
) -> Result<(u32, u32)> {
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .context("invalid PNG file")?;
    let (w, h) = (img.width(), img.height());
    if let Some(ew) = expected_width {
        if w != ew {
            bail!("PNG width mismatch: expected {ew}, got {w}");
        }
    }
    if let Some(eh) = expected_height {
        if h != eh {
            bail!("PNG height mismatch: expected {eh}, got {h}");
        }
    }
    Ok((w, h))
}

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Injects or replaces a `tEXt` chunk with `keyword` and `text` before the `IEND` chunk in a PNG.
pub fn inject_png_text_chunk(png_bytes: &[u8], keyword: &str, text: &str) -> Result<Vec<u8>> {
    if png_bytes.len() < 8 || png_bytes[0..8] != PNG_SIG {
        bail!("invalid PNG signature");
    }

    let mut chunk_data = Vec::with_capacity(keyword.len() + 1 + text.len());
    chunk_data.extend_from_slice(keyword.as_bytes());
    chunk_data.push(0);
    chunk_data.extend_from_slice(text.as_bytes());

    let mut crc_hasher = crc32fast::Hasher::new();
    crc_hasher.update(b"tEXt");
    crc_hasher.update(&chunk_data);
    let crc = crc_hasher.finalize();

    let mut out = Vec::with_capacity(png_bytes.len() + 12 + chunk_data.len());
    out.extend_from_slice(&PNG_SIG);

    let mut pos = 8usize;
    let mut inserted = false;
    while pos + 8 <= png_bytes.len() {
        let len = u32::from_be_bytes(png_bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let chunk_type = &png_bytes[pos + 4..pos + 8];
        let total_chunk_len = 4 + 4 + len + 4;
        if pos + total_chunk_len > png_bytes.len() {
            bail!("corrupt PNG chunk at offset {pos}");
        }

        // Skip any existing tEXt chunk with the same keyword
        if chunk_type == b"tEXt" {
            let data = &png_bytes[pos + 8..pos + 8 + len];
            if let Some(null_pos) = data.iter().position(|&b| b == 0) {
                if &data[..null_pos] == keyword.as_bytes() {
                    pos += total_chunk_len;
                    continue;
                }
            }
        }

        if chunk_type == b"IEND" && !inserted {
            out.extend_from_slice(&(chunk_data.len() as u32).to_be_bytes());
            out.extend_from_slice(b"tEXt");
            out.extend_from_slice(&chunk_data);
            out.extend_from_slice(&crc.to_be_bytes());
            inserted = true;
        }

        out.extend_from_slice(&png_bytes[pos..pos + total_chunk_len]);
        pos += total_chunk_len;
    }

    if !inserted {
        bail!("PNG missing IEND chunk");
    }
    Ok(out)
}

/// Reads a `tEXt` chunk with `keyword` from a PNG buffer.
pub fn read_png_text_chunk(png_bytes: &[u8], keyword: &str) -> Result<Option<String>> {
    if png_bytes.len() < 8 || png_bytes[0..8] != PNG_SIG {
        bail!("invalid PNG signature");
    }
    let mut pos = 8usize;
    while pos + 8 <= png_bytes.len() {
        let len = u32::from_be_bytes(png_bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let chunk_type = &png_bytes[pos + 4..pos + 8];
        let total_chunk_len = 4 + 4 + len + 4;
        if pos + total_chunk_len > png_bytes.len() {
            bail!("corrupt PNG chunk at offset {pos}");
        }
        if chunk_type == b"tEXt" {
            let data = &png_bytes[pos + 8..pos + 8 + len];
            if let Some(null_pos) = data.iter().position(|&b| b == 0) {
                if &data[..null_pos] == keyword.as_bytes() {
                    let s = String::from_utf8(data[null_pos + 1..].to_vec())
                        .context("invalid UTF-8 in PNG tEXt chunk")?;
                    return Ok(Some(s));
                }
            }
        }
        pos += total_chunk_len;
    }
    Ok(None)
}
