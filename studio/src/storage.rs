use anyhow::{anyhow, bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io::Cursor;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// PE-I2I: `<imageN>` when the image should keep that reference's aspect.
    #[serde(default)]
    pub ratio_follow: String,
}

#[derive(Debug, Clone)]
pub struct BlobStore {
    data_dir: PathBuf,
    /// Blobs are content-addressed and shared, and `put_blob` skips a file that already
    /// exists: writers hold [`Self::writing`] from `put_blob` until the record pointing at
    /// the blob is committed, and deleters hold [`Self::collecting`] from deciding a blob is
    /// unreferenced until its file is gone, so no record ends up on a deleted file.
    gc_lock: Arc<tokio::sync::RwLock<()>>,
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
        Ok(Self {
            data_dir,
            gc_lock: Arc::new(tokio::sync::RwLock::new(())),
        })
    }

    /// Held from `put_blob` until the record pointing at the blob is committed.
    pub async fn writing(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        self.gc_lock.clone().read_owned().await
    }

    /// Held from choosing unreferenced blobs until their files are deleted.
    pub async fn collecting(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        self.gc_lock.clone().write_owned().await
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

    /// Writes a JPEG thumbnail at each of [`THUMB_SIZES`] (longer side, never upscaled), of a
    /// job's PNG or an uploaded image (key `up_<sha256>`, upright by its EXIF orientation).
    /// CPU-heavy (decodes the full image): call it from a blocking task.
    pub fn write_thumbnails(&self, job_id: &str, image_bytes: &[u8]) -> Result<()> {
        let img = decode_upright(image_bytes).context("failed to decode image for thumbnail")?;
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

/// Image formats `POST /api/uploads` accepts, as `(format, MIME type)`.
const UPLOAD_FORMATS: [(ImageFormat, &str); 3] = [
    (ImageFormat::Png, "image/png"),
    (ImageFormat::Jpeg, "image/jpeg"),
    (ImageFormat::WebP, "image/webp"),
];

/// Largest reference image accepted: the model resizes references to ~1 MP anyway, and a
/// small file can claim a huge size that thumbnailing would then decode.
const UPLOAD_MAX_SIDE: u32 = 16384;
const UPLOAD_MAX_PIXELS: u64 = 64_000_000;

/// Upload thumbnails share the thumbnail directory with jobs; job ids are at most 64
/// characters, so `up_` plus a 64-character hash never collides with one.
pub fn upload_thumb_key(sha256: &str) -> String {
    format!("up_{sha256}")
}

/// Whether `s` is a lowercase SHA-256 hex digest (blob names are nothing else).
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Decodes an image and turns it upright by its EXIF orientation (phone photos).
fn decode_upright(bytes: &[u8]) -> Result<DynamicImage> {
    let mut decoder = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// MIME type and upright size of an uploaded reference image; PNG, JPEG or WebP only. Reads
/// the header and EXIF, not the pixels.
pub fn probe_upload_image(bytes: &[u8]) -> Result<(&'static str, u32, u32)> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let format = reader.format().ok_or_else(|| anyhow!("not an image"))?;
    let mime = UPLOAD_FORMATS
        .iter()
        .find(|(f, _)| *f == format)
        .map(|(_, m)| *m)
        .ok_or_else(|| anyhow!("unsupported image format {format:?} (PNG, JPEG or WebP)"))?;
    let mut decoder = reader.into_decoder().context("invalid image")?;
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 {
        bail!("image has no pixels");
    }
    if w.max(h) > UPLOAD_MAX_SIDE || (w as u64) * (h as u64) > UPLOAD_MAX_PIXELS {
        bail!("image is {w}×{h}; at most {UPLOAD_MAX_SIDE} px a side and 64 MP");
    }
    // Orientations 5–8 turn the image by 90°.
    let turned = matches!(
        decoder.orientation()?,
        image::metadata::Orientation::Rotate90
            | image::metadata::Orientation::Rotate270
            | image::metadata::Orientation::Rotate90FlipH
            | image::metadata::Orientation::Rotate270FlipH
    );
    Ok(if turned { (mime, h, w) } else { (mime, w, h) })
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
    let mut parsed: EnhancedJson =
        serde_json::from_slice(bytes).context("invalid enhanced.json payload")?;
    if parsed.rewritten_prompt.trim().is_empty() {
        bail!("enhanced.json rewritten_prompt cannot be empty");
    }
    // Only `<imageN>` names an input; anything else is dropped.
    let names_input = parsed
        .ratio_follow
        .strip_prefix("<image")
        .and_then(|r| r.strip_suffix('>'))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()));
    if !names_input {
        parsed.ratio_follow.clear();
    }
    Ok(parsed)
}

pub fn validate_embeds_safetensors(
    bytes: &[u8],
    require_negative: bool,
    references: usize,
) -> Result<SafetensorsHeader> {
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
    if references > 0 {
        // The transformer needs to know which embedding positions hold image tokens.
        if !header.tensors.contains_key("image_pad_mask") {
            bail!("embeds.safetensors missing 'image_pad_mask' required with reference images");
        }
        let sizes: Vec<[u32; 2]> = header
            .metadata
            .get("reference_sizes")
            .and_then(|v| serde_json::from_str(v).ok())
            .ok_or_else(|| anyhow!("embeds.safetensors missing metadata 'reference_sizes'"))?;
        if sizes.len() != references {
            bail!(
                "embeds.safetensors encodes {} reference images, the job has {references}",
                sizes.len()
            );
        }
    }
    Ok(header)
}

/// `ref<i>.safetensors`: VAE latents of the reference image with hash `source_sha256`,
/// resized for `reference_res`. Returns the header and the resized `(width, height)`.
pub fn validate_reference_latents(
    bytes: &[u8],
    source_sha256: &str,
    reference_res: u32,
) -> Result<(SafetensorsHeader, u32, u32)> {
    let header = parse_safetensors_header(bytes)?;
    let t = header
        .tensors
        .get("latents")
        .ok_or_else(|| anyhow!("reference latents missing tensor 'latents'"))?;
    if t.shape.len() != 3 {
        bail!("reference latents tensor 'latents' must be (1, tokens, channels), got {:?}", t.shape);
    }
    for req_meta in ["width", "height", "reference_res", "source_sha256", "model", "rev", "dtype"] {
        match header.metadata.get(req_meta) {
            Some(v) if !v.is_empty() => {}
            _ => bail!("reference latents missing metadata '{req_meta}'"),
        }
    }
    let meta = |k: &str| header.metadata.get(k).map(|s| s.as_str()).unwrap_or_default();
    if meta("source_sha256") != source_sha256 {
        bail!("reference latents are of image {}, expected {source_sha256}", meta("source_sha256"));
    }
    if meta("reference_res") != reference_res.to_string() {
        bail!("reference latents at resolution {}, expected {reference_res}", meta("reference_res"));
    }
    let (Ok(w), Ok(h)) = (meta("width").parse::<u32>(), meta("height").parse::<u32>()) else {
        bail!("reference latents width/height are not numbers");
    };
    // One latent token per 16x16 pixel tile.
    if t.shape[1] != ((w / 16) * (h / 16)) as usize {
        bail!("reference latents hold {} tokens, {w}x{h} needs {}", t.shape[1], (w / 16) * (h / 16));
    }
    Ok((header, w, h))
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
