//! Search (docs/SEARCH.md): images ranked by what they show and by the words of their
//! prompts. SigLIP 2 embeds each finished image (from its thumbnail) and each search
//! query, on Studio's CPU through ONNX Runtime; the image vectors live in SQLite and in
//! memory, and a query is one pass over them. SQLite FTS5 matches the words; the two
//! rankings are fused, and each result says which of them found it.

use crate::db::JobPageQuery;
use crate::scheduler::AppState;
use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use ort::session::Session;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// A query matches an image when their cosine is at least this, and at least
/// [`RELATIVE_COSINE`] of the query's best. Tuned on a small set (docs/SEARCH.md §3):
/// relevant images scored 0.078 and up (median 0.14), others a median of 0.002.
pub const MIN_COSINE: f32 = 0.07;
pub const RELATIVE_COSINE: f32 = 0.5;
/// A one-word query's floor instead of [`MIN_COSINE`]: a word describes less of an image than
/// a phrase, so its images score lower (on the small set "door" found its image at 0.061).
/// The extra results it lets in are mostly [`Strength::Weak`], which come last.
pub const MIN_COSINE_ONE_WORD: f32 = 0.06;
/// [`Strength`] bands of a query's cosine. On the small set, of the results shown, 92% were
/// right from 0.12 up, 68% from 0.10, 15% below.
pub const STRONG_COSINE: f32 = 0.12;
pub const POSSIBLE_COSINE: f32 = 0.10;
/// "More like this": images at least this close to the chosen one. On the small set, images
/// of one subject scored a median of 0.79 with each other, unrelated ones 0.51 (95% under
/// 0.67): 0.65 kept 88% of the first and 7% of the second.
pub const MIN_SIMILAR_COSINE: f32 = 0.65;
/// Reciprocal rank fusion's constant: how much the top of each list outweighs the rest.
const FUSION_K: f32 = 60.0;
/// Bumped when how images are prepared for the model changes: stored vectors of another
/// version are made again.
const PREPARE_VERSION: u32 = 1;
const QUERY_CACHE_LEN: usize = 64;

/// Which ranking found a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Match {
    Semantic,
    Text,
    Both,
}

impl Match {
    pub fn as_str(self) -> &'static str {
        match self {
            Match::Semantic => "semantic",
            Match::Text => "text",
            Match::Both => "both",
        }
    }
}

/// How closely an image's look matches a search query, from their cosine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strength {
    Strong,
    Possible,
    Weak,
}

impl Strength {
    pub fn of(cosine: f32) -> Self {
        if cosine >= STRONG_COSINE {
            Strength::Strong
        } else if cosine >= POSSIBLE_COSINE {
            Strength::Possible
        } else {
            Strength::Weak
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Strength::Strong => "strong",
            Strength::Possible => "possible",
            Strength::Weak => "weak",
        }
    }
}

/// One search result: which ranking found it and, when its look did, how closely.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: String,
    pub found: Match,
    pub strength: Option<Strength>,
}

impl Hit {
    /// Found only by a weak look: these come after every other result.
    pub fn weak(&self) -> bool {
        self.found == Match::Semantic && self.strength == Some(Strength::Weak)
    }
}

/// `manifest.json` as reference/search/export.py writes it.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub revision: String,
    pub image_size: u32,
    pub max_tokens: usize,
    pub dim: usize,
}

struct Encoder {
    vision: Mutex<Session>,
    text: Mutex<Session>,
    tokenizer: Tokenizer,
    manifest: Manifest,
    /// Stored with each vector: the model, its revision and [`PREPARE_VERSION`].
    key: String,
}

impl Encoder {
    fn load(dir: &Path, threads: usize) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(dir.join("manifest.json")).context("reading manifest.json")?,
        )?;
        let env = ort::init_from(dir.join("libonnxruntime.so"))
            .map_err(|e| anyhow!("loading ONNX Runtime: {e}"))?
            .with_name("aries-search")
            .build()?;
        let session = |file: &str| -> Result<Session> {
            // Prepacking keeps a second, rearranged copy of the weights (~230 MB for the
            // image encoder) for no speed we could measure; the arena holds peak memory.
            let e = |e: ort::Error<_>| anyhow!("{e}");
            Session::builder(&env)?
                .with_intra_threads(threads)
                .map_err(e)?
                .with_inter_threads(1)
                .map_err(e)?
                .with_prepacking(false)
                .map_err(e)?
                .with_execution_providers([ort::ep::CPU::default()
                    .with_arena_allocator(false)
                    .build()])
                .map_err(e)?
                .commit_from_file(dir.join(file))
                .with_context(|| format!("loading {file}"))
        };
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: manifest.max_tokens,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("{e}"))?;
        let key = format!(
            "{}@{}/{PREPARE_VERSION}",
            manifest.model,
            &manifest.revision[..manifest.revision.len().min(12)]
        );
        Ok(Self {
            vision: Mutex::new(session("vision.onnx")?),
            text: Mutex::new(session("text.onnx")?),
            tokenizer,
            manifest,
            key,
        })
    }

    /// The unit-length embedding of an image (any size: it is squashed to the model's square).
    fn embed_image(&self, image: &image::DynamicImage) -> Result<Vec<f32>> {
        let pixels = prepare_image(image, self.manifest.image_size);
        let n = self.manifest.image_size as usize;
        let tensor = ort::value::Tensor::from_array(([1usize, 3, n, n], pixels))?;
        let mut session = self.vision.lock().map_err(|_| anyhow!("vision lock"))?;
        let out = session.run(ort::inputs!["pixel_values" => tensor])?;
        let (_, v) = out["embeds"].try_extract_tensor::<f32>()?;
        Ok(normalize(v.to_vec()))
    }

    /// The unit-length embedding of a search query.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let tensor = ort::value::Tensor::from_array(([1usize, ids.len()], ids))?;
        let mut session = self.text.lock().map_err(|_| anyhow!("text lock"))?;
        let out = session.run(ort::inputs!["input_ids" => tensor])?;
        let (_, v) = out["embeds"].try_extract_tensor::<f32>()?;
        Ok(normalize(v.to_vec()))
    }
}

/// Pixels as the model takes them: `size` x `size` RGB, channels first, scaled to [-1, 1]
/// (SigLIP's processor: resize without keeping the aspect, mean and std 0.5).
pub fn prepare_image(image: &image::DynamicImage, size: u32) -> Vec<f32> {
    let rgb = image
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8();
    let plane = (size * size) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, p) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = p.0[c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Short English words a description is written with but a prompt need not contain: "a red
/// fox in the snow" should find "red fox, deep snow".
const FILLER_WORDS: &[&str] = &[
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "from", "with", "and", "or", "is",
    "are", "its", "it", "this", "that", "some",
];

/// An FTS5 query from what someone typed: every word must appear, the last one as a prefix
/// (results follow the typing), except [`FILLER_WORDS`] (unless there are only those).
/// Words are quoted, so FTS5 syntax in the text is just text.
pub fn fts_query(text: &str) -> String {
    let words: Vec<String> = required_words(text)
        .iter()
        .map(|w| format!("\"{w}\""))
        .collect();
    match words.split_last() {
        Some((last, rest)) => {
            let mut q: Vec<String> = rest.to_vec();
            q.push(format!("{last}*"));
            q.join(" ")
        }
        None => String::new(),
    }
}

/// Words as FTS5's tokenizer splits them, lowercased.
fn words_of(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The words of a query a prompt must contain: all but [`FILLER_WORDS`] (unless there are
/// only those).
fn required_words(text: &str) -> Vec<String> {
    let all = words_of(text);
    let content: Vec<String> = all
        .iter()
        .filter(|w| !FILLER_WORDS.contains(&w.as_str()))
        .cloned()
        .collect();
    if content.is_empty() {
        all
    } else {
        content
    }
}

/// Where a word match found the query's words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Words {
    /// All of them in the prompt as written.
    Prompt,
    /// Only in the enhancer's rewrite.
    Enhanced,
    /// The digits in the seed.
    Seed,
}

impl Words {
    pub fn as_str(self) -> &'static str {
        match self {
            Words::Prompt => "prompt",
            Words::Enhanced => "enhanced",
            Words::Seed => "seed",
        }
    }
}

/// Whether `word` is `have`, or begins it when it is the query's last (still being typed).
fn word_matches(have: &str, word: &str, last: bool) -> bool {
    if last {
        have.starts_with(word)
    } else {
        have == word
    }
}

/// Where a job's text holds `query`'s words, and whether as the exact phrase (two words or
/// more, in order, side by side). FTS5 also folds diacritics; a match that only does so here
/// counts as the prompt.
pub fn words_found(query: &str, prompt: &str, enhanced: Option<&str>, seed: u64) -> (Words, bool) {
    let required = required_words(query);
    let phrase = words_of(query);
    let holds = |text: &str| -> Option<bool> {
        let have = words_of(text);
        let all = required.iter().enumerate().all(|(i, w)| {
            let last = i + 1 == required.len();
            have.iter().any(|h| word_matches(h, w, last))
        });
        all.then(|| {
            phrase.len() >= 2
                && have.windows(phrase.len()).any(|win| {
                    win.iter()
                        .zip(&phrase)
                        .enumerate()
                        .all(|(i, (h, w))| word_matches(h, w, i + 1 == phrase.len()))
                })
        })
    };
    if let Some(p) = holds(prompt) {
        return (Words::Prompt, p);
    }
    if let Some(p) = enhanced.and_then(holds) {
        return (Words::Enhanced, p);
    }
    let q = query.trim();
    if !q.is_empty() && q.chars().all(|c| c.is_ascii_digit()) && seed.to_string().contains(q) {
        return (Words::Seed, false);
    }
    (Words::Prompt, false)
}

/// Fuses two best-first lists by reciprocal rank: an item's score is the sum of
/// `1 / (FUSION_K + rank)` over the lists it is in. Ties keep `order`'s order. Results found
/// only by a weak look ([`Hit::weak`]) follow all the others.
pub fn fuse(semantic: &[(String, f32)], text: &[String], order: &[String]) -> Vec<Hit> {
    // Score, cosine (when its look matched), found by words.
    let mut scores: HashMap<&str, (f32, Option<f32>, bool)> = HashMap::new();
    for (rank, (id, cosine)) in semantic.iter().enumerate() {
        let e = scores.entry(id).or_default();
        e.0 += 1.0 / (FUSION_K + rank as f32 + 1.0);
        e.1 = Some(*cosine);
    }
    for (rank, id) in text.iter().enumerate() {
        let e = scores.entry(id).or_default();
        e.0 += 1.0 / (FUSION_K + rank as f32 + 1.0);
        e.2 = true;
    }
    let mut out: Vec<(usize, f32, Hit)> = order
        .iter()
        .enumerate()
        .filter_map(|(i, id)| {
            let (score, cosine, t) = *scores.get(id.as_str())?;
            let found = match (cosine.is_some(), t) {
                (true, true) => Match::Both,
                (true, false) => Match::Semantic,
                _ => Match::Text,
            };
            let strength = cosine.map(Strength::of);
            Some((
                i,
                score,
                Hit {
                    id: id.clone(),
                    found,
                    strength,
                },
            ))
        })
        .collect();
    out.sort_by(|a, b| {
        a.2.weak()
            .cmp(&b.2.weak())
            .then(b.1.total_cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });
    out.into_iter().map(|(_, _, hit)| hit).collect()
}

/// The images that match a query embedding, best first, with their cosines: [`MIN_COSINE`]
/// ([`MIN_COSINE_ONE_WORD`] for a query of one word) and [`RELATIVE_COSINE`] of the best.
pub fn semantic_matches(scored: &mut [(String, f32)], one_word: bool) -> Vec<(String, f32)> {
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let best = scored.first().map(|s| s.1).unwrap_or(0.0);
    let min = if one_word {
        MIN_COSINE_ONE_WORD
    } else {
        MIN_COSINE
    };
    let floor = min.max(best * RELATIVE_COSINE);
    scored
        .iter()
        .take_while(|(_, s)| *s >= floor)
        .cloned()
        .collect()
}

/// The search model, the image vectors and the indexer's wake-up.
#[derive(Default)]
pub struct Search {
    encoder: Option<Arc<Encoder>>,
    index: RwLock<HashMap<String, Arc<Vec<f32>>>>,
    queries: Mutex<VecDeque<(String, Arc<Vec<f32>>)>>,
    wake: tokio::sync::Notify,
}

impl Search {
    /// Loads the model from `dir`; without one (or if it fails to load), search matches
    /// words only.
    pub fn new(dir: Option<&Path>, threads: usize) -> Self {
        let encoder = dir.and_then(|dir| match Encoder::load(dir, threads) {
            Ok(e) => {
                info!(model = %e.key, threads, "Search model loaded");
                Some(Arc::new(e))
            }
            Err(e) => {
                warn!(dir = %dir.display(), error = %format!("{e:#}"), "Search model not loaded: searching words only");
                None
            }
        });
        Self {
            encoder,
            ..Default::default()
        }
    }

    pub fn semantic(&self) -> bool {
        self.encoder.is_some()
    }

    /// An image's embedding, on the calling thread (it takes tens to hundreds of ms).
    pub fn encode_image(&self, image: &image::DynamicImage) -> Result<Vec<f32>> {
        self.encoder
            .as_ref()
            .ok_or_else(|| anyhow!("no search model"))?
            .embed_image(image)
    }

    /// A query's embedding, on the calling thread.
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        self.encoder
            .as_ref()
            .ok_or_else(|| anyhow!("no search model"))?
            .embed_text(text)
    }

    /// A job finished: its image can be embedded.
    pub fn image_added(&self) {
        self.wake.notify_one();
    }

    pub fn indexed(&self) -> usize {
        self.index.read().map(|i| i.len()).unwrap_or(0)
    }

    async fn query_vector(&self, state: &AppState, text: &str) -> Option<Arc<Vec<f32>>> {
        let encoder = self.encoder.clone()?;
        if let Some(v) = self
            .queries
            .lock()
            .ok()?
            .iter()
            .find(|(q, _)| q == text)
            .map(|(_, v)| v.clone())
        {
            return Some(v);
        }
        let t0 = Instant::now();
        let owned = text.to_string();
        let v = tokio::task::spawn_blocking(move || encoder.embed_text(&owned))
            .await
            .ok()?;
        let v = match v {
            Ok(v) => Arc::new(v),
            Err(e) => {
                warn!(error = %format!("{e:#}"), "embedding a search query failed");
                return None;
            }
        };
        state
            .metrics
            .observe_search_embed("query", t0.elapsed().as_secs_f64());
        if let Ok(mut q) = self.queries.lock() {
            if q.len() >= QUERY_CACHE_LEN {
                q.pop_front();
            }
            q.push_back((text.to_string(), v.clone()));
        }
        Some(v)
    }

    /// Scores of `candidates` that have a vector, against `v`.
    fn scores(&self, v: &[f32], candidates: &[String]) -> Vec<(String, f32)> {
        let Ok(index) = self.index.read() else {
            return Vec::new();
        };
        candidates
            .iter()
            .filter_map(|id| Some((id.clone(), dot(v, index.get(id)?))))
            .collect()
    }

    /// `candidates` (newest first) that match `text`, best first.
    pub async fn rank_text(
        &self,
        state: &AppState,
        text: &str,
        candidates: &[String],
    ) -> Result<Vec<Hit>> {
        let allowed: HashSet<&str> = candidates.iter().map(String::as_str).collect();
        let text = text.trim();
        let digits = text.chars().all(|c| c.is_ascii_digit()).then_some(text);
        let words: Vec<String> = state
            .db
            .text_matches(&fts_query(text), digits)
            .await?
            .into_iter()
            .filter(|id| allowed.contains(id.as_str()))
            .collect();
        // A number (a seed) has no look.
        let vector = match digits {
            Some(_) => None,
            None => self.query_vector(state, text).await,
        };
        let semantic = match vector {
            Some(v) => semantic_matches(
                &mut self.scores(&v, candidates),
                required_words(text).len() <= 1,
            ),
            None => Vec::new(),
        };
        Ok(fuse(&semantic, &words, candidates))
    }

    /// `candidates` that look like the image of `job_id`, most alike first; empty when it has
    /// no vector.
    pub fn rank_similar(&self, job_id: &str, candidates: &[String]) -> Vec<Hit> {
        let Some(v) = self.index.read().ok().and_then(|i| i.get(job_id).cloned()) else {
            return Vec::new();
        };
        let mut scored: Vec<(String, f32)> = self
            .scores(&v, candidates)
            .into_iter()
            .filter(|(id, s)| id != job_id && *s >= MIN_SIMILAR_COSINE)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
            .into_iter()
            .map(|(id, _)| Hit {
                id,
                found: Match::Semantic,
                strength: None,
            })
            .collect()
    }

    /// Forgets a deleted job's vector (the database row goes with the job).
    pub fn remove(&self, job_id: &str) {
        if let Ok(mut i) = self.index.write() {
            i.remove(job_id);
        }
    }
}

/// One page of a search: `before` is `o:<offset>` past the first page.
pub async fn page(
    state: &AppState,
    q: &JobPageQuery,
    text: Option<&str>,
    like: Option<&str>,
) -> Result<(Vec<crate::db::JobRecord>, Option<String>)> {
    let candidates = state.db.search_candidates(q).await?;
    let ranked = match (like, text) {
        (Some(id), _) => state.search.rank_similar(id, &candidates),
        (None, Some(text)) => state.search.rank_text(state, text, &candidates).await?,
        (None, None) => Vec::new(),
    };
    let offset = q
        .before
        .as_deref()
        .and_then(|b| b.strip_prefix("o:"))
        .and_then(|o| o.parse::<usize>().ok())
        .unwrap_or(0);
    let limit = q.limit.clamp(1, 200) as usize;
    let slice: Vec<&Hit> = ranked.iter().skip(offset).take(limit).collect();
    let ids: Vec<String> = slice.iter().map(|h| h.id.clone()).collect();
    let mut jobs = state.db.get_jobs(&ids).await?;
    let order: HashMap<&str, (usize, &Hit)> = slice
        .iter()
        .enumerate()
        .map(|(i, h)| (h.id.as_str(), (i, *h)))
        .collect();
    jobs.sort_by_key(|j| order.get(j.id.as_str()).map(|o| o.0).unwrap_or(usize::MAX));
    for job in &mut jobs {
        let Some((_, hit)) = order.get(job.id.as_str()) else {
            continue;
        };
        job.search_match = Some(hit.found.as_str());
        job.search_strength = hit.strength.map(Strength::as_str);
        if let (Some(text), Match::Text | Match::Both) = (text, hit.found) {
            let (words, phrase) =
                words_found(text, &job.prompt, job.enhanced_prompt.as_deref(), job.seed);
            job.search_words = Some(words.as_str());
            job.search_phrase = phrase;
        }
    }
    let next = (offset + limit < ranked.len()).then(|| format!("o:{}", offset + limit));
    Ok((jobs, next))
}

/// The thumbnail an image is embedded from, made from the PNG if it is missing.
async fn thumbnail(state: &AppState, job_id: &str) -> Option<Vec<u8>> {
    if let Some(bytes) = state.blobs.read_thumb(job_id, 512).await {
        return Some(bytes);
    }
    let art = state
        .db
        .get_job_artifact_by_name(job_id, "image.png")
        .await
        .ok()??;
    let (blobs, id) = (state.blobs.clone(), job_id.to_string());
    tokio::task::spawn_blocking(move || {
        let png = blobs.get_blob(&art.sha256).ok()?;
        blobs.write_thumbnails(&id, &png).ok()?;
        std::fs::read(blobs.thumb_path(&id, 512)).ok()
    })
    .await
    .ok()
    .flatten()
}

/// Loads the stored vectors, then embeds every finished image without one, newest first,
/// and each new one as it finishes. One image at a time, on a blocking thread.
pub fn spawn_indexer(state: Arc<AppState>) {
    let Some(encoder) = state.search.encoder.clone() else {
        return;
    };
    tokio::spawn(async move {
        match state.db.load_embeddings(&encoder.key).await {
            Ok(rows) => {
                if let Ok(mut index) = state.search.index.write() {
                    for (id, bytes) in rows {
                        let v = from_bytes(&bytes);
                        if v.len() == encoder.manifest.dim {
                            index.insert(id, Arc::new(v));
                        }
                    }
                }
                info!(images = state.search.indexed(), "Search index loaded");
            }
            Err(e) => warn!(error = %e, "loading the search index failed"),
        }
        state
            .metrics
            .set_search_indexed(state.search.indexed() as u64);
        // Images that could not be embedded (no PNG, undecodable): not tried again until
        // a restart.
        let mut failed: HashSet<String> = HashSet::new();
        loop {
            let ids = state
                .db
                .jobs_to_embed(&encoder.key, failed.len() as u32 + 16)
                .await
                .unwrap_or_default();
            let ids: Vec<String> = ids.into_iter().filter(|id| !failed.contains(id)).collect();
            if ids.is_empty() {
                state.search.wake.notified().await;
                continue;
            }
            for id in ids {
                let t0 = Instant::now();
                let vector = match thumbnail(&state, &id).await {
                    Some(bytes) => {
                        let enc = encoder.clone();
                        tokio::task::spawn_blocking(move || {
                            let img = image::load_from_memory(&bytes)?;
                            enc.embed_image(&img)
                        })
                        .await
                        .map_err(|e| anyhow!("{e}"))
                        .and_then(|r| r)
                    }
                    None => Err(anyhow!("no image")),
                };
                match vector {
                    Ok(v) => {
                        state
                            .metrics
                            .observe_search_embed("image", t0.elapsed().as_secs_f64());
                        match state
                            .db
                            .store_embedding(&id, &encoder.key, &to_bytes(&v))
                            .await
                        {
                            Ok(true) => {
                                if let Ok(mut index) = state.search.index.write() {
                                    index.insert(id, Arc::new(v));
                                }
                            }
                            Ok(false) => {}
                            Err(e) => {
                                warn!(job_id = %id, error = %e, "storing an embedding failed");
                                failed.insert(id);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(job_id = %id, error = %format!("{e:#}"), "embedding an image failed");
                        failed.insert(id);
                    }
                }
                state
                    .metrics
                    .set_search_indexed(state.search.indexed() as u64);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn fts_query_quotes_words_and_prefixes_the_last() {
        assert_eq!(fts_query("Red fox"), "\"red\" \"fox\"*");
        assert_eq!(
            fts_query("  \"NEAR(x y)\" NOT -z "),
            "\"near\" \"x\" \"y\" \"not\" \"z\"*"
        );
        // Filler words are not required, unless they are all there is.
        assert_eq!(
            fts_query("a red fox in the snow"),
            "\"red\" \"fox\" \"snow\"*"
        );
        assert_eq!(fts_query("the"), "\"the\"*");
        assert_eq!(fts_query(" .,! "), "");
        assert_eq!(fts_query("café"), "\"café\"*");
    }

    fn scored(v: &[(&str, f32)]) -> Vec<(String, f32)> {
        v.iter().map(|(id, s)| (id.to_string(), *s)).collect()
    }

    fn found(hits: &[Hit]) -> Vec<(&str, Match, Option<Strength>)> {
        hits.iter()
            .map(|h| (h.id.as_str(), h.found, h.strength))
            .collect()
    }

    #[test]
    fn fusion_ranks_both_first_and_labels_the_source() {
        let order = ids(&["a", "b", "c", "d", "e"]);
        let fused = fuse(
            &scored(&[("c", 0.2), ("a", 0.11)]),
            &ids(&["a", "e"]),
            &order,
        );
        assert_eq!(
            found(&fused),
            vec![
                ("a", Match::Both, Some(Strength::Possible)),
                ("c", Match::Semantic, Some(Strength::Strong)),
                ("e", Match::Text, None),
            ]
        );
    }

    #[test]
    fn fusion_breaks_ties_by_the_candidates_order() {
        // Same rank in either list: the newer one (earlier in `order`) first.
        let fused = fuse(&scored(&[("d", 0.2)]), &ids(&["b"]), &ids(&["b", "d"]));
        assert_eq!(fused[0].id, "b");
    }

    #[test]
    fn weak_looks_come_after_everything_else() {
        let order = ids(&["a", "b", "c", "d"]);
        // "a" ranks first by look but weakly; "d" is weak too but also found by its words.
        let semantic = scored(&[("a", 0.09), ("b", 0.13), ("d", 0.08)]);
        let fused = fuse(&semantic, &ids(&["c", "d"]), &order);
        assert_eq!(
            found(&fused),
            vec![
                ("d", Match::Both, Some(Strength::Weak)),
                ("c", Match::Text, None),
                ("b", Match::Semantic, Some(Strength::Strong)),
                ("a", Match::Semantic, Some(Strength::Weak)),
            ]
        );
    }

    #[test]
    fn strength_bands() {
        assert_eq!(Strength::of(0.30), Strength::Strong);
        assert_eq!(Strength::of(0.12), Strength::Strong);
        assert_eq!(Strength::of(0.11), Strength::Possible);
        assert_eq!(Strength::of(0.07), Strength::Weak);
    }

    #[test]
    fn semantic_matches_keep_the_close_ones() {
        let mut s = scored(&[
            ("weak", 0.06),
            ("best", 0.20),
            ("half", 0.11),
            ("below_half", 0.09),
        ]);
        assert_eq!(
            semantic_matches(&mut s, false),
            scored(&[("best", 0.20), ("half", 0.11)])
        );
        assert!(semantic_matches(&mut scored(&[("a", 0.05), ("b", 0.03)]), false).is_empty());
        // One word scores lower: a lower floor.
        let mut door = scored(&[("door", 0.061), ("noise", 0.03)]);
        assert!(semantic_matches(&mut door.clone(), false).is_empty());
        assert_eq!(
            semantic_matches(&mut door, true),
            scored(&[("door", 0.061)])
        );
    }

    #[test]
    fn words_found_say_where_and_whether_as_typed() {
        let prompt = "A red fox standing in deep snow, golden hour";
        let enhanced = Some("A vivid red fox in fresh snow under a pale winter sky");
        assert_eq!(
            words_found("red fox", prompt, enhanced, 7),
            (Words::Prompt, true)
        );
        // The last word is a prefix, in the phrase too.
        assert_eq!(
            words_found("red fo", prompt, enhanced, 7),
            (Words::Prompt, true)
        );
        // Every word, apart.
        assert_eq!(
            words_found("fox snow", prompt, enhanced, 7),
            (Words::Prompt, false)
        );
        // Filler words need not be there, but are part of the phrase.
        assert_eq!(
            words_found("fox in snow", prompt, enhanced, 7),
            (Words::Prompt, false)
        );
        assert_eq!(
            words_found("fox in fresh snow", prompt, enhanced, 7),
            (Words::Enhanced, true)
        );
        assert_eq!(
            words_found("winter", prompt, enhanced, 7),
            (Words::Enhanced, false)
        );
        assert_eq!(
            words_found("4242", prompt, enhanced, 142420),
            (Words::Seed, false)
        );
    }

    #[test]
    fn images_are_squashed_to_the_square_and_scaled() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            40,
            20,
            image::Rgb([255, 0, 128]),
        ));
        let px = prepare_image(&img, 8);
        assert_eq!(px.len(), 3 * 64);
        assert_eq!((px[0], px[64], px[128]), (1.0, -1.0, 128.0 / 127.5 - 1.0));
    }

    #[test]
    fn vectors_round_trip_as_bytes() {
        let v = vec![0.25f32, -1.5, 3.0];
        assert_eq!(from_bytes(&to_bytes(&v)), v);
        let n = normalize(vec![3.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6);
    }
}
