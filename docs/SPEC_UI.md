# Aries Image Studio — UI Spec

Originally written as the design brief for the mockups in `mockups/`; it remains the
reference for UI behavior. Items marked *(future)* are not built.

A web app for one person or a few, for generating images from text prompts. You type
a prompt, pick a few settings, and get pictures. Everything you've ever made stays
in a browsable history with its settings attached.

The app is a **hybrid of a chat timeline and a photo gallery**. There is no
conversation with an AI — the "chat" shape is only a way to show *what I asked for*
next to *what I got*, in order.

This brief covers the UI only. Build constraints are noted where they affect design.

---

## 1. Design principles

1. **Images are the content.** Interface chrome recedes; pictures get the space.
2. **One step from "again, differently."** Every image offers an obvious way to make
   a variation or reproduce it exactly. Both fill the composer rather than starting a
   job, so the user can change anything before sending.
3. **Calm while waiting.** A 2K image takes 1–3 minutes. Waiting should feel
   informative and unhurried, not anxious: show real progress, never fake spinners.
4. **Honest metadata.** Every image can tell you exactly how it was made.
5. **Few knobs, well chosen.** No advanced panels, no node graphs, no jargon in the
   main flow.

Original visual identity, please. Lightweight "labs-style" image tools are a fair
mood reference for tone — quiet, playful-but-restrained, image-forward — but don't
reproduce any existing product's layout or branding.

---

## 2. Context of use

- **One user or a few**, at home, on a desktop browser (primary) and a phone (secondary).
  With sign-in on (AUTH.md), each signs in with Google and sees only their own history.
- Generation is **slow** (1–4 min per image) and mostly **sequential** (two graphics
  cards share the stages; others wait in a queue). Users often submit a few prompts
  and come back later.
- The generating machine is sometimes **asleep or offline**. History must still be
  fully browsable then; new requests wait until it's back.
- Images are **large** (≈2048 px on the long side) and come in **7 aspect ratios**,
  from tall 9:16 to wide 16:9. Layouts must handle mixed ratios gracefully.

---

## 3. Information architecture

```
App                                          URL
├── Sign-in screen (sign-in on, signed out) — "Sign in with Google"    any URL
├── Top bar: app name · view switch (Timeline / Gallery) · generator status · theme · account
├── Timeline view (default)      — chat-shaped history + composer      /
├── Gallery view                 — grid of all images + filters        /gallery?fav=1&aspect=16:9&q=…
├── Image viewer (a page)        — full image + details + actions      /images/{id}
└── System view (a page)         — agents, cards, queues               /status
```

Every screen is a page with its own URL, so any of them can be linked, bookmarked or
opened in a new tab, and Back/Forward move between them. Pages never stack: the viewer
and the system view replace the list rather than covering it, so scrolling one can never
scroll another. Back from the viewer or the system view returns to the list where it was
left (its loaded pages and scroll position). The shortcuts sheet is the one dialog left; the
page underneath does not scroll while it is open.

---

## 4. Screens and components

### 4.1 Timeline view (default)

**Newest turn on top**; older turns load 40 jobs at a time as the page scrolls down. The
composer is docked at the bottom, and the page opens at the latest turn.

Each entry is a **turn** — one submission:

| Element | Notes |
|---|---|
| Prompt | The text as typed. Can be long (up to a paragraph); truncate with expand. |
| Enhanced prompt | An AI-rewritten, much longer version, with the rewriter's reasoning collapsed. Clearly secondary. To use it, Rerun the image and pick "Enhanced" in the composer (§4.2). |
| Reference images | Small numbered thumbnails under the prompt, in the order the prompt names them ("image 1", …); each opens the original. |
| Setting chips | e.g. `16:9 · seed 48213 · 20 steps · 1m 12s`. Compact, quiet. |
| Image tiles | The images of one send (rows × count, up to 16), left to right in the order they were sent and are processed. Shown at their true aspect ratio. A batch is a carousel: one row at full tile height that snaps tile by tile, with edge fades and arrows while more is off-screen, and dots below. Swipe, wheel, the arrows, the dots, the batch pills or `←`/`→` (with the row focused) move through it, and an image the user picks is scrolled into view; a new turn starts at its first image and does not move as images finish. |
| Lineage marker *(planned)* | When made from another image: "↳ variation of …" linking back to the source. |

**Tile states** (each needs a design):

| State | What's shown |
|---|---|
| Queued | Aspect-correct placeholder naming the stage it waits for (enhancement, text encoding, drawing, decoding). *(future: "2 ahead · starts in ~4 min")* Cancel. |
| Generating | Placeholder that names the phase it's in — see below. Cancel. |
| Done | The image. Hover/tap reveals quick actions (§4.4). |
| Failed | Calm error ("The generator stopped unexpectedly"), **Retry**. Not alarming. When a cloud enhancer rewrote the prompt, also **Built-in**: retry with the built-in enhancer. |
| Cancelled | Muted placeholder, "Cancelled", Retry. |

A turn with several tiles can be in mixed states (one done, one generating, two queued).

**Generating has phases.** Real progress, in order — design each so the tile never
looks stuck:

| Phase | Typical length | What the tile says |
|---|---|---|
| Waking *(future)* | 10–30 s | "Waking generator" |
| Enhancing | about a minute, when Enhance is on | "Enhancing" with the sub-stage (reading prompt · thinking · rewriting prompt), tokens/s and time left. The bar is tokens generated against the expected finish line, 85 % of the enhancer's token limit (4096 by default, plus 2048 per reference image; the worker reports a raised limit and the bar follows it); past it the bar turns indeterminate with "almost done…" until the run ends (at the limit at the latest). With a cloud enhancer: seconds, "waiting for the enhancer" while a rate limit holds it, then "rewriting prompt", with an indeterminate bar |
| Reading prompt | a few seconds | "Reading prompt" with an indeterminate bar (the encoder reports no steps) |
| Drawing | most of the time | "Drawing", percent, live speed and time left ("6.40s/step · ~180s left") |
| Finishing | 5–15 s | "Finishing" with an indeterminate bar — decoding the full-size image; must not look frozen |

Each stage runs in a freshly started worker, so a phase may begin with a few seconds
of model loading; the tile shows the GPU running it.

### 4.2 Composer

Docked at the bottom of the Timeline. Should feel like a message box, not a form.

- **Folded by default** — a one-line pill with the send button: full width at the bottom
  on a phone, bottom right (26rem wide) from 640 px up. Beside the prompt it shows what the
  next send makes (`16:9 · ¼ · 12 st · 4 img`, ✦ for Enhance, an image icon with the
  number of reference images, a sliders icon while Advanced is open). Focusing the prompt, a tap on that summary, `/`, or Rerun / Vary
  opens it to the full composer (growing from its corner on wider screens). A send, `Esc`
  or a tap outside folds it again; the prompt and settings stay. Folded with Advanced
  open, the send button opens the composer instead of sending, so nothing hidden is sent.

- **Prompt field** — multi-line, grows with content up to a limit. Enter submits,
  Shift+Enter makes a new line, and the field empties once the job is sent. Placeholder
  suggests what to type (with the keyboard hint only where there is a mouse).
- **Settings row** — compact chips/controls beside or under the field:

| Setting | Options | Default |
|---|---|---|
| Aspect ratio | 1:1 · 4:3 · 3:4 · 3:2 · 2:3 · 16:9 · 9:16 | 1:1 |
| Count | 1 · 2 · 3 · 4 (per row; shown at every width, phone too) | 1 |
| Quality | **Draft** (12 steps) · **Standard** (20) · **High** (40) | Standard |
| Size | Full · ½ · ¼ (faster previews) | Full |
| Enhance prompt | On/off toggle | Off |
| Enhancer | Shown next to Enhance while it is on, when Studio offers cloud enhancers (CLOUD_ENHANCERS.md): Built-in · Sonnet 5.5 · Gemini 3.5 Flash Lite (one entry per configured model). Its tooltip says the prompt and any reference images go to the provider, whose terms apply. A cloud enhancer reads up to 10 reference images | Built-in |

Every image gets its own random seed; the seed is not shown unless Advanced is open.

- **Advanced** (a toggle in the settings row; closed by default). Its settings apply only
  while it is open; closed, a send uses a preset's steps (a custom count counts as
  Standard), a random seed and no tiling. Advanced is for one send: sending closes it and
  resets its settings (custom steps to Standard, random seed, no tiling, the default
  enhancer token limit, extra rows removed), so nothing hidden carries over to the next
  send.

| Advanced setting | Options | Default |
|---|---|---|
| Steps | Any number from 1 to 100; a preset's number selects that preset | the preset's |
| Seed | **Random seed** or **Fixed seed** (number, with a dice to roll a new one) | Random |
| Tiling | VAE tiling at decode: faster and lighter, but can leave seams | Off |
| Enhancer tokens | The prompt enhancer's token limit, reasoning included: 256–24576, for long, detailed prompts such as multi-part infographics (~30 tokens a second). Shown when a row has Enhance on; applies to those rows. Reasoning that runs out of room continues with the limit doubled, up to 24576. The viewer's time breakdown shows tokens used against the final limit, and notes a raised limit | 4096 + 2048 per reference image |
| Rows | Row 1 is the main settings; **Add row** copies the last row; each extra row sets aspect, size, steps, Enhance and tiling, and can be removed | one row |

  All rows share the prompt and the seeds, so they compare like for like: image *k* of
  every row gets the same seed. With Random seed each image of a row gets its own random
  seed; with Fixed seed they count up from it (seed, seed + 1, …). A send makes rows × count images, at
  most 16 (Add row and Generate are disabled beyond that). One send is one turn; each
  image's picker is as short as possible: number, seed, only the settings that differ in
  that turn (e.g. "3:2 12st"), and icons for Enhance (✦) and tiling (⊞) when on; the
  tooltip spells everything out.

- **Reference images** (docs/REFERENCE_IMAGES.md). An attach button in the settings row
  (with `n/4` once some are attached), a drop anywhere on the page (the composer opens
  and outlines itself while files are dragged) or a paste into the prompt adds PNG, JPEG
  or WebP images, up to the limit Studio reports (`/api/capabilities`
  `max_references_ever`; the button is disabled at 0). Each uploads at once and shows in
  a strip above the prompt: a thumbnail with its number (the prompt's "image 1", …), a
  spinner while uploading, a red outline if the upload failed, × to remove and ‹ to move
  it one place earlier. The strip says how the numbers map to the prompt; a short note
  replaces that for a wrong file type or too many images. While references are attached,
  Enhance uses the image-aware enhancer, which reads them; when no agent has one for that
  many images (`/api/capabilities` `enhance_max_references_ever`), Enhance (and each row's)
  is off and disabled, and its tooltip says why; a cloud enhancer reads them instead.
  Attachments belong to one send, like the prompt: sending clears the strip.
- **Generate** button. Disabled when the prompt is empty, and while a reference is still
  uploading or has failed (remove it to send). Sending scrolls the timeline to the top,
  where the new turn appears.
- Aspect ratio picker should show the shapes visually, not just the numbers.
- Settings are remembered between visits.
- **Filled from an image.** Rerun and Vary (from a tile, the timeline or the viewer) fill
  the composer and never submit by themselves: prompt, aspect, size, steps and the
  image's Enhance setting and enhancer, enhancer token limit and tiling; Rerun locks the image's seed (opening Advanced so it
  applies), Vary sets the seed to Random. Extra rows are cleared. The image's reference
  images replace any attached ones.
  A bar above the prompt shows "From #seed" with a dismiss button. If the image had an
  enhanced prompt, the bar has an **Original · Enhanced** switch: Original is the prompt
  as typed with the image's Enhance setting; Enhanced puts the rewritten prompt in with
  Enhance off, so it is used as written. Sending the job clears the bar.

### 4.3 Gallery view

All finished images, newest first, infinite scroll (Studio filters and pages them: the
browser holds only what was scrolled to). The filters live in the URL.

- **Mixed-aspect grid** (masonry or justified rows — designer's call) using thumbnails.
- **Filters**: Favorites only · aspect ratio · text search over prompts (and seeds). The
  counts on the pills are over all finished images.
- Hover/tap shows quick actions (§4.4). Click opens the Image viewer.
- Must look good with mostly 1:1, and with a mix of 9:16 and 16:9.

### 4.4 Quick actions (on a tile, on hover / long-press)

- Favorite (toggle)
- Vary — fills the composer with this image's prompt and settings and a random seed (§4.2)
- Use as reference (in the turn's action row) — attaches the image to the composer, which
  opens to show it; hidden when the composer is at its limit
- Open
- Download

### 4.5 Image viewer (`/images/{id}`)

Opens from Timeline or Gallery, or from a shared link. A page of its own that fills the
window, image dominant; no top bar or composer (the composer keeps its text for when the
list comes back). A copy-link button copies the page's URL. Esc or close goes back to the
list it came from; opened from a link, to the timeline.

- **Image**: fit to screen; zoom to 100 % and pan. 100 % is the image's actual size, one
  image pixel per device pixel (Fit shows its own percentage). Wheel or pinch zooms at
  the pointer, a click zooms in to 100 % at the pointer and the next click (without
  dragging) back to Fit, and `+`/`-`, `0` (Fit) and `1`
  (100 %) step through zoom levels up to 800 %. When the image is larger than the
  screen, drag, arrow keys (Shift for bigger steps) or a navigator thumbnail (the visible
  part outlined; click or drag it) move around it. Previous/next navigates within the
  list it was opened from (the buttons always; `←`/`→` when the image fits), in that view's
  order: from the timeline, left to right through a turn and then on to the next (older)
  turn; from the gallery, newest first. Stepping replaces the URL (Back returns to the list,
  not through every image), and older images load as the end of the list comes near. A
  shared link steps through the timeline when the image is in it.
- **Details panel** (collapsible on desktop; a sheet on phone):
  - Prompt (copy button), its reference images, and the enhanced prompt (its reasoning is
    fetched when opened: job lists leave it out)
  - Seed (copy button), size and aspect, steps / quality
  - For each stage, the card that ran it and its model (id and revision)
  - Total time (submit to done), and a Time breakdown: each stage's time (enhance with
    tokens and tok/s, denoise with steps × s/step, decode with its mode), their sum as
    generation time, then queue & model loading (the rest: waiting for a GPU, starting
    workers, loading models) and the total
  - Expandable pipeline details: stages, decode mode, the GPU that decoded it and peak
    memory
  - Created date/time
- **Actions**:

| Action | Behavior |
|---|---|
| Rerun | Fills the composer with this image's prompt, settings and seed (locked), and closes the viewer; sending it should reproduce the same picture. |
| Vary | The same, with the seed set to Random. |
| Download | PNG with settings embedded. |
| Copy prompt / Copy seed | |
| Favorite | |
| Delete | Asks first, then removes the image and its settings for good (Enter confirms, Esc cancels). The same on a tile and for a whole turn; a failed or cancelled card, with no image to lose, goes at once. |
| Use as reference | Attaches the image to the composer as the next reference image and closes the viewer. Hidden when the composer is at its limit. |

### 4.6 Top bar

- App name and mark: "Aries Image Studio"; the mark is a ram charging out of a pixel trail inside a picture frame (`web/public/favicon.svg`, `web/src/components/Logo.tsx`)
- View switch: Timeline / Gallery
- **Generator status** pill:

| Status | Meaning |
|---|---|
| Ready | Idle, will start immediately |
| Busy · 3 queued | Working; shows queue size |
| Error | Every card is unavailable; tap for details |
| Offline | The generator can't be reached |
| Asleep *(future)* | Generator machine is off; requests will wait |

  Tapping it opens the **status panel**, a compact live readout:
  - Status in words
  - **Cards**: per graphics card, its state, the stage it runs (and sub-stage), and a
    memory bar with peak, reserved and any spill into host RAM
  - **Queue**: jobs waiting per stage
  - **Host memory**: available RAM (memory-heavy tasks wait for the agent's floor)
  - **Recent worker runs**: stage, card, duration, jobs processed, why it exited
  - *(future)* queue clear estimate, last image timing, a link to a dashboard

  It should read in two seconds, not look like an admin console. Numbers update live.
- Theme toggle (light / dark) — **light by default**, not "follow system"
- **Account** (sign-in on): the account's Google picture (or its initial); a menu with the
  name, the email and **Sign out**. Lend / Disable / Enable on the status page are shown
  only to the owner.

### 4.7 Empty, edge and error states (please design)

- Loading: until Studio's first job list arrives, the timeline and gallery show a spinner,
  never the empty state, so the starter does not flash on every page load.
- First run: no history yet — inviting, with a few example prompts to tap.
- Gallery with no favorites / no search results.
- Generator Asleep while user submits — reassure that it's queued.
- Very long prompt; very long run of history (hundreds of turns).
- Network lost in the browser (reconnecting).
- Delete confirmation.
- Signed out (sign-in on): instead of the app, a centered card with the mark, "Sign in to
  see your images" and **Sign in with Google**, in the system's light or dark theme; under
  it, one line of small print, always visible: what Studio is (a private, self-hosted image
  generator, invitation only, open models on the host's computers) and that Google sign-in
  only checks you are invited, no Google API making or editing images. Any
  URL shows it, and signing in returns there. A refused or failed sign-in says why above
  the button ("That Google account doesn't have access to this Studio."). A session that
  expires while the app is open brings the card back.

---

## 5. Visual and interaction direction

- **Tone**: calm, focused, a little delightful. A creative tool, not an admin panel.
- **Color**: neutral base so images carry the color; one accent for primary actions
  and progress. **Light is the default** and the primary design target; dark must be
  equally polished.
- **Shape**: rounded tiles and controls, soft depth, generous spacing.
- **Typography**: one clean family; prompts readable at length.
- **Motion**: brief and purposeful — new tiles arrive softly, progress is smooth,
  the viewer opens from the tile it came from. Respect reduced-motion settings.
- **Density**: Timeline is airy; Gallery can be denser.

---

## 6. Future features — leave room, don't fully design

1. **Aspect suggestion from the enhancer.** The enhancer already returns a
   suggested aspect ratio (`wh_ratio`, or with references `ratio_follow`: keep image N's
   aspect); the UI does not use it yet.
2. **Fast mode.** A third Quality option producing images in seconds at lower quality.
3. **Transparent images.** Results are RGBA, so a prompt that asks for a transparent
   background may get one; tiles and the viewer would then show a checkerboard behind
   it (JPEG thumbnails drop alpha today). None of 40 recent results had transparent
   areas, only 0.1–6 % slightly translucent pixels, so this waits for a real case.

---

## 7. Responsive and accessibility

- **Desktop** (primary): ≥1280 px. Timeline column centered with comfortable width;
  viewer with side details panel.
- **Tablet**: same structure, narrower.
- **Phone** (≥360 px): single column; composer docked at bottom with settings in a
  compact row or a sheet; viewer details as a bottom sheet; no horizontal scrolling.
  The image page runs to the screen's edges (under iOS's home indicator), its controls
  kept clear of them. Once the details are scrolled they take a quarter of the image's
  space: the image keeps its scale and moves up by half that, with its zoom controls; back
  at the top of the details, it returns.
  The header shows the mark without the name and the status pill's one-word state.
- Keyboard: `/` focuses the prompt · `T` / `G` switch views · `←`/`→` in the viewer ·
  `F` favorites · `+`/`-`/`0`/`1` zoom · `Esc` closes · `?` shows the shortcut sheet.
- Visible focus states, WCAG AA contrast, all icon buttons labeled.

---

## 8. Realistic sample content (use in mockups)

Prompts:
- `A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement`
- `Three red cubes and two blue spheres on a white table, studio lighting`
- `Portrait of an elderly fisherman, Rembrandt lighting, weathered skin, 85mm photo`
- `Panoramic mountain lake at dawn, mist over the water, a single canoe` (16:9)
- `A lighthouse on a cliff during a storm, waves crashing` (9:16)
- `A cute cartoon dragon sticker, transparent background`

Metadata example:
```
Prompt     A lighthouse on a cliff during a storm, waves crashing
Seed       48213
Size       1536 × 2752 (9:16)
Quality    High · 40 steps
Model      Qwen-Image 2.1 · bf16
Time       enhance 1m 12s · encode 3.1s · denoise 2m 32s (40 × 3.8s) · decode 5.2s
           generation 3m 52s · queue & model loading 4m 10s · total 8m 2s
Created    Sat, 09:41
```
