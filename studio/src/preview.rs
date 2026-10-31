//! A rough preview of a job's image from its denoised latent, before decode: one linear map
//! from each latent pixel and its eight neighbours (64 channels each, plus a bias) to the
//! 4x4 RGB pixels it covers, so the preview is a quarter of the image's size. No model and
//! no GPU: about 28 thousand multiply-adds per latent pixel, some 20 ms on one CPU core for
//! the largest image (2752x1536). The web app shows it on the tile while decode waits.

use crate::storage::parse_safetensors_header;
use anyhow::{anyhow, bail, Result};
use image::codecs::jpeg::JpegEncoder;
use image::ExtendedColorType;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Least-squares fit on Qwen-Image-2.1 VAE encodings of sample images (held-out PSNR about
/// 23 dB against the half-size image): `INPUTS` rows of `OUTPUTS` little-endian f32.
const MATRIX: &[u8] = include_bytes!("../assets/latent_preview.bin");
/// Latent channels of the Qwen-Image-2.1 VAE.
const CHANNELS: usize = 64;
/// 3x3 neighbourhood of channels, then the bias.
const INPUTS: usize = 9 * CHANNELS + 1;
/// Preview pixels per latent pixel, per side.
const SCALE: usize = 4;
const OUTPUTS: usize = SCALE * SCALE * 3;

/// Recent previews by latent hash: each tile, browser and refetch would otherwise render
/// again. A latent never changes, so its preview never goes stale.
static CACHE: Mutex<VecDeque<(String, Arc<Vec<u8>>)>> = Mutex::new(VecDeque::new());
const CACHE_LEN: usize = 32;

pub fn cached(latent_sha256: &str) -> Option<Arc<Vec<u8>>> {
    let cache = CACHE.lock().ok()?;
    cache
        .iter()
        .find(|(k, _)| k == latent_sha256)
        .map(|(_, v)| v.clone())
}

pub fn remember(latent_sha256: &str, jpeg: Arc<Vec<u8>>) {
    if let Ok(mut cache) = CACHE.lock() {
        cache.retain(|(k, _)| k != latent_sha256);
        if cache.len() >= CACHE_LEN {
            cache.pop_front();
        }
        cache.push_back((latent_sha256.to_string(), jpeg));
    }
}

/// Per-step previews from denoise workers (ARCHITECTURE_V3.md §7.5): the newest JPEG per
/// denoise task, with its step, oldest dropped first. In memory only.
type Live = VecDeque<(String, u32, Arc<Vec<u8>>)>;
static LIVE: Mutex<Live> = Mutex::new(VecDeque::new());
const LIVE_LEN: usize = 64;

pub fn store_live(task_id: &str, step: u32, jpeg: Vec<u8>) {
    if let Ok(mut live) = LIVE.lock() {
        live.retain(|(k, _, _)| k != task_id);
        if live.len() >= LIVE_LEN {
            live.pop_front();
        }
        live.push_back((task_id.to_string(), step, Arc::new(jpeg)));
    }
}

/// The newest per-step preview of a denoise task: its step and the JPEG.
pub fn live(task_id: &str) -> Option<(u32, Arc<Vec<u8>>)> {
    let live = LIVE.lock().ok()?;
    live.iter()
        .find(|(k, _, _)| k == task_id)
        .map(|(_, step, jpeg)| (*step, jpeg.clone()))
}

/// The job's denoise task, if it is still drawing (assigned, running or uploading).
pub fn drawing_task(job: &crate::db::JobRecord) -> Option<&str> {
    job.tasks
        .iter()
        .find(|t| t.stage == "denoise" && t.status.is_active())
        .map(|t| t.id.as_str())
}

/// Sets `preview_step` on jobs whose denoise worker has sent a preview.
pub fn attach_steps(jobs: &mut [crate::db::JobRecord]) {
    for job in jobs {
        job.preview_step = drawing_task(job).and_then(live).map(|(step, _)| step);
    }
}

/// `(width, height)` of a JPEG, from its header.
pub fn jpeg_size(jpeg: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::with_format(std::io::Cursor::new(jpeg), image::ImageFormat::Jpeg)
        .into_dimensions()
        .ok()
}

/// Models whose latents the matrix was fitted for.
pub fn supports(model_id: &str) -> bool {
    model_id.starts_with("Qwen/Qwen-Image-2.1")
}

fn matrix() -> Vec<f32> {
    MATRIX
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The `latent` tensor of a `latent.safetensors`, as f32, with its latent height and width.
/// Packed `(1, h * w, 64)`, `h` and `w` as decode unpacks them for `width` x `height`.
fn read_latent(bytes: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, usize, usize)> {
    let header = parse_safetensors_header(bytes)?;
    let t = header
        .tensors
        .get("latent")
        .ok_or_else(|| anyhow!("no tensor 'latent'"))?;
    let (h, w) = (2 * (height as usize / 32), 2 * (width as usize / 32));
    if t.shape != [1, h * w, CHANNELS] {
        bail!(
            "latent shape {:?}, expected [1, {}, {CHANNELS}]",
            t.shape,
            h * w
        );
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    let data = bytes
        .get(8 + header_len + t.data_offsets[0]..8 + header_len + t.data_offsets[1])
        .ok_or_else(|| anyhow!("latent data out of range"))?;
    let values: Vec<f32> = match t.dtype.as_str() {
        "F32" => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        "BF16" => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        "F16" => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => bail!("latent dtype {other} not supported"),
    };
    if values.len() != h * w * CHANNELS {
        bail!(
            "latent holds {} values, expected {}",
            values.len(),
            h * w * CHANNELS
        );
    }
    Ok((values, h, w))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// RGB pixels of the preview, `(w * SCALE) x (h * SCALE)`, from a latent laid out
/// token-major (`y * w + x`, then channel).
fn render_rgb(latent: &[f32], h: usize, w: usize, m: &[f32]) -> Vec<u8> {
    let (pw, ph) = (w * SCALE, h * SCALE);
    let mut rgb = vec![0u8; pw * ph * 3];
    let mut acc = [0f32; OUTPUTS];
    for y in 0..h {
        for x in 0..w {
            acc.copy_from_slice(&m[(INPUTS - 1) * OUTPUTS..]);
            for dy in 0..3 {
                // Edge pixels repeat, as in the fit.
                let sy = (y + dy).saturating_sub(1).min(h - 1);
                for dx in 0..3 {
                    let sx = (x + dx).saturating_sub(1).min(w - 1);
                    let z = &latent[(sy * w + sx) * CHANNELS..][..CHANNELS];
                    let rows = &m[(dy * 3 + dx) * CHANNELS * OUTPUTS..][..CHANNELS * OUTPUTS];
                    for (zc, row) in z.iter().zip(rows.as_chunks::<OUTPUTS>().0) {
                        for (a, r) in acc.iter_mut().zip(row) {
                            *a += zc * r;
                        }
                    }
                }
            }
            for sy in 0..SCALE {
                for sx in 0..SCALE {
                    let o = ((y * SCALE + sy) * pw + x * SCALE + sx) * 3;
                    let a = (sy * SCALE + sx) * 3;
                    for c in 0..3 {
                        rgb[o + c] = (acc[a + c].clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
    }
    rgb
}

/// A JPEG preview from the bytes of a job's `latent.safetensors`.
pub fn render_jpeg(latent_safetensors: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let (latent, h, w) = read_latent(latent_safetensors, width, height)?;
    let rgb = render_rgb(&latent, h, w, &matrix());
    let mut out = Vec::new();
    JpegEncoder::new_with_quality(&mut out, 85).encode(
        &rgb,
        (w * SCALE) as u32,
        (h * SCALE) as u32,
        ExtendedColorType::Rgb8,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors(dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
        let header = serde_json::json!({
            "latent": { "dtype": dtype, "shape": shape, "data_offsets": [0, data.len()] },
            "__metadata__": { "model": "Qwen/Qwen-Image-2.1" },
        })
        .to_string();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn matrix_has_the_expected_size() {
        assert_eq!(MATRIX.len(), INPUTS * OUTPUTS * 4);
    }

    #[test]
    fn renders_a_quarter_size_jpeg_from_bf16() {
        // 128x64 px: an 8x4 latent, so a 32x16 preview.
        let (h, w) = (4, 8);
        let data: Vec<u8> = (0..h * w * CHANNELS)
            .map(|i| ((((i % 7) as f32 - 3.0) * 0.3).to_bits() >> 16) as u16)
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let jpeg =
            render_jpeg(&safetensors("BF16", &[1, h * w, CHANNELS], &data), 128, 64).unwrap();
        let img = image::load_from_memory(&jpeg).unwrap();
        assert_eq!((img.width(), img.height()), (32, 16));
    }

    #[test]
    fn a_zero_latent_is_the_bias_everywhere() {
        let m = matrix();
        let rgb = render_rgb(&vec![0.0; 2 * 2 * CHANNELS], 2, 2, &m);
        let bias = &m[(INPUTS - 1) * OUTPUTS..];
        assert_eq!(rgb[0], (bias[0].clamp(0.0, 1.0) * 255.0).round() as u8);
        assert_eq!(rgb.len(), 8 * 8 * 3);
    }

    #[test]
    fn refuses_a_latent_of_the_wrong_size() {
        let data = vec![0u8; 4 * CHANNELS * 4];
        assert!(render_jpeg(&safetensors("F32", &[1, 4, CHANNELS], &data), 128, 64).is_err());
    }

    #[test]
    fn half_floats_convert() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
