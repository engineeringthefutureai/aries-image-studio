# Search

The gallery's search box finds images by **what they show** and by **the words of their
prompts** (and seeds); **More like this** ranks the gallery by how alike images look to one
of them. Everything runs in Studio, on its CPU: search works while the GPU host sleeps or
its cards are busy, and nothing leaves Studio.

## 1. What it does

- **Search** (`/gallery?q=…`): results come best first, not newest first. Each carries a
  small badge with what found it:
  - an eye (what the image shows) with signal bars for how closely: three **strong**, two
    **possible**, one **weak**;
  - a **T** (the words), a quote mark when they are the exact phrase typed, dimmed when
    only the enhanced prompt has them; a **#** for a seed (a query of digits also finds
    seeds containing them).

  Tapping the badge spells it out ("Strong look match · Exact phrase in your prompt"); with
  a mouse, the tile's hover overlay does. Results found only by a weak look come last,
  under **Weaker matches**.
- **More like this** (`/gallery?like=<id>`, from a tile's hover actions or the viewer):
  the gallery ranked by how alike each image looks to that one. Typing a search replaces it.
- Favorites and aspect filters apply to both; results page as you scroll.

## 2. How it works

**The model.** SigLIP 2 base, patch 16, 256 px (`google/siglip2-base-patch16-256` at a
pinned revision): an image encoder and a text encoder that map images and text into one
768-dimension space, where a matching caption and image point the same way. Studio runs
both through ONNX Runtime (the `ort` crate, the library loaded at startup), the tokenizer
through the `tokenizers` crate. Both encoders stay loaded: queries never wait for a model.

- The image encoder is f32. The text encoder's 256k-token embedding table is stored as
  int8 (it is only looked up, never multiplied: query embeddings stay within 0.997 cosine
  of f32), which takes the text encoder from 1.1 GB to 0.5 GB. Int8 for the image encoder
  cost too much (some embeddings fell to 0.57 cosine of f32; retrieval mAP 0.98 → 0.82–0.94).
- Weight prepacking is off: it keeps a second, rearranged copy of the weights (~230 MB for
  the image encoder) for no speed that could be measured.

**Indexing.** When an image finishes, Studio writes its thumbnails and then embeds the
512 px one: squashed to 256 × 256 (SigLIP's own preprocessing, aspect not kept), scaled
to [-1, 1], through the image encoder, normalized. At startup a background pass embeds
every finished image without a vector, newest first; one image at a time, on a blocking
thread. Vectors are stored in SQLite (`job_embeddings`: job id, model key, 768 little-endian
f32) and held in memory (3 KB an image: 30 MB for 10,000). The model key is the model, its
revision and a preprocessing version; a new key re-embeds everything. Deleting a job
deletes its vector (a trigger) and drops it from memory.

**A search** ranks the account's images that pass the filters, two ways:

- **Meaning.** The query is tokenized (64 tokens, SigLIP's length) and embedded; each
  image's score is its cosine with the query, one pass over the vectors. An image matches
  when its score is at least **0.07** (**0.06** for a query of one word, filler words
  aside) and at least **half** the best score of that query (§3). Its strength: **strong**
  from 0.12, **possible** from 0.10, **weak** below. The last 64 query embeddings are kept,
  so paging and retyping cost nothing.
- **Words.** SQLite FTS5 over the prompt and the enhanced prompt (`jobs_fts`, kept in
  step with `jobs` by triggers), ranked by BM25. Every word must appear, the last as a
  prefix (results follow the typing); short filler words (a, the, in, of, …) are not
  required. Punctuation separates words and is never query syntax. A query of digits also
  matches seeds containing them.

The two lists are fused by **reciprocal rank**: an image's score is the sum of
`1 / (60 + rank)` over the lists it is in, so one found both ways rises to the top; ties
go to the newer image; results found only by a weak look follow all the others. Each
result says which list found it (`search_match`: `semantic`, `text` or `both`); when its
look did, how closely (`search_strength`: `strong`, `possible`, `weak`); when its words did,
where (`search_words`: `prompt` when the prompt as written has every word, else `enhanced`,
or `seed`) and whether as the exact phrase typed (`search_phrase`, two words or more in
order, the last as a prefix). Pages are offsets into the ranking (`next` is `o:<offset>`).

**More like this** ranks by the cosine between the chosen image's vector and the others',
from **0.65** up (§3), most alike first. Only someone who may see the chosen image can
rank by it.

**Without a model** (`ARIES_SEARCH_MODEL_DIR` unset or unreadable) Studio logs why and
searches words only; More like this finds nothing.

## 3. Measurements

A set of 50 local images (generated pictures, reference photos, and Studio screenshots
as distractors) and 15 queries written against it ("goblin wizard with a black cat",
"rainy night street in Tokyo with neon signs", "terms of service page", …). Small, and the
queries were written by someone who had seen the images: a sanity check, not a benchmark.

| | mAP | Top hit right | Memory (resident) |
|---|---|---|---|
| f32, ONNX Runtime defaults | 0.978 | 15 / 15 | 1.8 GB (peak 2.2) |
| **Image f32, text int8 table, no prepacking** (shipped) | **0.977** | **15 / 15** | **0.9 GB (peak 1.1)** |
| Everything int8 | 0.820 | | 0.5 GB |

Studio's Rust pipeline gives the same vectors as the Python reference (cosine ≥ 0.9998 for
images, identical for queries) and the same retrieval. In the Studio container (2 threads)
the whole set indexed at about 180 ms an image (with JPEG decoding), a search answered in
about 52 ms, and the container held 1.09 GB.

Speed of the encoders on a Core Ultra 5 250K Plus:

| Threads | Image | Query |
|---|---|---|
| 1 | 330 ms | 80 ms |
| 4 | 90 ms | 30 ms |

**Thresholds.** Query to image: relevant images scored a cosine of 0.078 and up (median
0.14), others a median of 0.002 (1% above 0.10). At least 0.07 and half the best kept every
relevant image. Image to image: images of one subject scored a median of 0.79 with each
other, unrelated ones 0.51 (95% under 0.67); 0.65 keeps 88% of the first and 7% of the
second, most of those fair neighbours (the same characters in another scene). Lowercasing
queries changed nothing.

**Strength.** Of the results shown for the 15 queries and 10 single words, 92% were right
from a cosine of 0.12 up, 68% from 0.10 to 0.12, 15% below 0.10: hence the three levels,
and the weak ones last. SigLIP's own match probability (`sigmoid(cosine × scale + bias)`)
was no use as a confidence: right images had a median of 31% and as little as 0.03%, a
wrong one 94%.

**Single words.** Ten one-word queries ("fox", "door", "screenshot", …) ranked their images
as well as the phrases did (mAP 0.99), but lower: "laptop" found its image at 0.071, "door"
at 0.061, under the 0.07 floor. A word describes less of an image than a phrase. A floor
of 0.06 for one word keeps all of them, with 5.4 wrong images a query instead of 3.2
(mostly weak, so after the divider). Tried and dropped: wrapping the word in a caption
("a photo of a door", mAP 0.84), or scoring each image against the query's scores over the
whole gallery (no better than the cosine, and worse when many images match).

## 4. The model files

`reference/search/export.py` downloads the model at its pinned revision, exports the two
encoders to ONNX (the text encoder with its int8 table), and writes them with
`tokenizer.json`, the ONNX Runtime library (`libonnxruntime.so`, from the same
`onnxruntime` wheel) and `manifest.json` (model, revision, sizes, file hashes). Studio's
Dockerfile runs it in a build stage, at build time only, and copies the result to
`/app/search-model` (about 940 MB; the image is about 830 MB compressed, 1.9 GB unpacked). The stage stays
cached until the script changes, and rebuilds give byte-identical model files.

To run Studio from source with search, export once and point Studio at the folder:

```bash
python reference/search/export.py ~/aries-search-model   # torch, transformers, onnx, onnxruntime
ARIES_SEARCH_MODEL_DIR=~/aries-search-model cargo run -p aries-image-studio
```

## 5. Configuration and metrics

| Variable | Default | Meaning |
|---|---|---|
| `ARIES_SEARCH_MODEL_DIR` | `/app/search-model` in the image; unset otherwise | The exported model; unset or unreadable: words only |
| `ARIES_SEARCH_THREADS` | `2` | CPU threads for each embedding |

- `aries_search_embed_seconds{kind="image"|"query"}`: time to embed an image (from reading
  its thumbnail) or a query, histogram.
- `aries_search_indexed_images`: images with a vector.

## 6. Later: embeddings on the GPU host's NPU (concept)

The GPU host's CPU has an NPU (Intel AI Boost). An optional `embed` stage could run the
same image encoder there through OpenVINO, for the backfill of a large gallery or just to
use the hardware: the agent declares the stage, Studio sends it thumbnails, and the vectors
land in the same table under the same model key (same weights, same space). Studio keeps
the text encoder, so search never depends on the GPU host being awake, and keeps embedding
images itself whenever no agent offers the stage. Not built; the CPU path is fast enough
for a gallery of thousands.
