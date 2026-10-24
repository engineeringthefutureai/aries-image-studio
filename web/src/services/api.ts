import type {
  CapabilitiesInfo,
  CardStatusInfo,
  Generation,
  JobParams,
  JobPhase,
  JobStatus,
  JobTimings,
  SizeMultiplier,
  SystemAgentInfo,
  SystemQueueInfo,
  UserPreferences,
  V3JobRecord,
  WorkerHealth,
} from '../types';
import { ASPECT_RATIOS, type AspectRatio } from '../types';
import { sanitizeEnhancedPrompt } from '../utils/prompt';

/** UI metadata the SPA stores on each job in Studio (`meta`); Studio does not read it. */
export interface JobMeta {
  batch_id?: string;
  aspect?: AspectRatio;
  size_multiplier?: SizeMultiplier;
  quality?: JobParams['quality'];
  /** Position in its send (rows × count), so a turn shows its images in that order. */
  index?: number;
}

/** The composer's aspect label for a size (exact or scaled-down preset), else "w:h". */
export function aspectOf(width: number, height: number): AspectRatio {
  for (const [label, dim] of Object.entries(ASPECT_RATIOS)) {
    if (Math.abs(width / height - dim.width / dim.height) < 0.01) return label as AspectRatio;
  }
  return `${width}:${height}` as AspectRatio;
}

const STAGE_PHASE: Record<string, JobPhase> = {
  enhance: 'enhancing',
  encode: 'encoding',
  denoise: 'denoising',
  decode: 'decoding',
};

/** Thumbnail sizes Studio makes (longer side, px; `THUMB_SIZES` in studio/src/storage.rs). */
const THUMB_SIZES = [512, 768, 1024];

/**
 * `srcset` for an image's thumbnails, each with its real width: the size is the longer
 * side, and thumbnails are never larger than the image.
 */
function thumbSrcSet(id: string, width: number, height: number): string {
  const longer = Math.max(width, height);
  const seen = new Set<number>();
  return THUMB_SIZES.flatMap((size) => {
    const w = Math.round((Math.min(size, longer) * width) / longer);
    if (seen.has(w)) return [];
    seen.add(w);
    return [`/api/images/${encodeURIComponent(id)}/thumb?size=${size} ${w}w`];
  }).join(', ');
}

/** A Studio job record as the app's Generation; images are served by Studio. */
export function jobToGeneration(v3: V3JobRecord): Generation {
  const tasks = v3.tasks || [];
  const active = tasks.find((t) => t.status === 'running' || t.status === 'assigned' || t.status === 'uploading');
  const current = active || tasks.find((t) => t.status === 'queued');
  const meta = (v3.meta || {}) as JobMeta;

  const status: JobStatus =
    v3.status === 'done' || v3.status === 'failed' || v3.status === 'cancelled'
      ? v3.status
      : active
      ? 'running'
      : 'queued';

  const timings: JobTimings = {};
  if (v3.finished_at && v3.created_at) {
    timings.total_s = Math.max(0, v3.finished_at - v3.created_at);
  }
  let peakVram = 0;
  for (const [stage, prov] of Object.entries(v3.provenance || {})) {
    peakVram = Math.max(peakVram, prov.peak_mem_bytes || 0);
    // Each stage's worker timings (enhance_s, s_per_step, generated_tokens, ...) merge into
    // one record; load_s is per stage and stays out.
    for (const [key, value] of Object.entries(prov.timings || {})) {
      if (key !== 'load_s' && typeof value === 'number') {
        (timings as Record<string, number>)[key] = value;
      }
    }
    if (stage === 'decode') timings.decode_mode = v3.tiling ? 'tiled' : 'untiled';
  }

  // Denoise counts steps; the enhancer counts tokens against its limit (shown via stats).
  const isEnhancing = current?.stage === 'enhance';
  const step = active && !isEnhancing ? active.step || 0 : 0;
  const sPerStep = active && !isEnhancing ? active.s_per_step || undefined : undefined;
  const remaining = Math.max(0, (active?.steps || 0) - step);
  const executorTask = active || (status === 'done' ? undefined : current);
  const executorName = executorTask?.assigned_executor
    ? `${executorTask.assigned_agent || 'agent'}/${executorTask.assigned_executor}`
    : v3.provenance?.decode
    ? `${v3.provenance.decode.agent}/${v3.provenance.decode.executor}`
    : undefined;
  const cardDigits = (active?.assigned_executor || v3.provenance?.decode?.executor || '').match(/(\d+)$/);
  const done = status === 'done';

  return {
    id: v3.id,
    batchId: meta.batch_id || v3.id,
    batchIndex: meta.index,
    prompt: v3.prompt,
    enhancedPrompt: v3.enhanced_prompt ? sanitizeEnhancedPrompt(v3.enhanced_prompt) : undefined,
    thinking: v3.thinking || undefined,
    hasThinking: !!(v3.has_thinking || v3.thinking),
    params: {
      width: v3.width,
      height: v3.height,
      aspect: meta.aspect || aspectOf(v3.width, v3.height),
      sizeMultiplier: meta.size_multiplier,
      steps: v3.steps,
      seed: v3.seed,
      quality: meta.quality || 'custom',
      enhance: v3.enhance,
      enhanceMaxTokens: v3.enhance_max_tokens ?? undefined,
      enhancer: v3.enhancer ?? undefined,
      tiling: !!v3.tiling,
    },
    status,
    phase: done ? 'done' : status === 'running' && current ? STAGE_PHASE[current.stage] : 'queued',
    stage: current?.stage || (v3.enhance ? 'enhance' : 'encode'),
    subStage: isEnhancing && active ? active.sub_stage || 'prefill' : undefined,
    enhanceStats: isEnhancing && active ? active.stats || undefined : undefined,
    step,
    totalSteps: v3.steps,
    secondsPerStep: sPerStep,
    etaSeconds: sPerStep && remaining > 0 ? remaining * sPerStep : undefined,
    imageUrl: done ? `/api/images/${encodeURIComponent(v3.id)}` : undefined,
    thumbUrl: done ? `/api/images/${encodeURIComponent(v3.id)}/thumb?size=1024` : undefined,
    thumbSrcSet: done ? thumbSrcSet(v3.id, v3.width, v3.height) : undefined,
    error: v3.error_detail || v3.error_class || undefined,
    blockedReason: !done ? v3.blocked_reason || current?.blocked_reason || undefined : undefined,
    provenance: v3.provenance,
    createdAt: v3.created_at * 1000,
    completedAt: v3.finished_at ? v3.finished_at * 1000 : undefined,
    timings,
    favorite: !!v3.favorite,
    executor: executorName,
    card: cardDigits ? Number(cardDigits[1]) : undefined,
    peakVram: peakVram || undefined,
    references: v3.references?.length ? v3.references : undefined,
  };
}

/**
 * Upload an image to use as a reference (`POST /api/uploads`). Resolves to its hash and
 * size; rejects with Studio's reason (not an image, too large, ...).
 */
export async function uploadReferenceApi(file: Blob): Promise<{ sha256: string; width: number; height: number }> {
  const res = await fetch('/api/uploads', {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Upload failed (${res.status})`);
  return data;
}

/**
 * Fetch the backend's `/api/system` snapshot for the status pill and V3 system modal.
 */
export async function fetchWorkerHealth(): Promise<WorkerHealth> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const res = await fetch('/api/system', { signal: controller.signal, cache: 'no-store' });
    if (res.ok) {
      const data = await res.json();
      const agents: SystemAgentInfo[] = data.agents || [];
      const queues: SystemQueueInfo[] = data.queues || [];

      const cards: CardStatusInfo[] = [];
      for (const ag of agents) {
        for (const ex of ag.executors || []) {
          const isBusy = ex.state === 'busy' || ex.state === 'starting';
          const isUnavail = ex.state === 'offline' || ex.state === 'disabled' || ex.state === 'lent';
          cards.push({
            card: cards.length,
            state: isBusy ? 'running' : isUnavail ? 'unavailable' : 'idle',
            stage: ex.current_stage || null,
          });
        }
      }

      let state: WorkerHealth['state'] = 'ready';
      if (agents.length === 0 || !agents.some((a) => a.online)) {
        state = 'offline';
      } else if (cards.length > 0 && cards.every((c) => c.state === 'unavailable')) {
        state = 'error';
      } else if (cards.some((c) => c.state === 'running')) {
        state = 'busy';
      }
      return { state, cards, agents, queues };
    }
    return { state: 'error' };
  } catch {
    return { state: 'offline' };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Fetch capabilities (`/api/capabilities`) for composer size & stage greying.
 */
export async function fetchCapabilitiesApi(): Promise<CapabilitiesInfo | null> {
  try {
    const res = await fetch('/api/capabilities', { cache: 'no-store' });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/**
 * Operator controls for an executor (`/api/executors/{id}/{action}`).
 */
export async function controlExecutorApi(
  executorId: string,
  action: 'lend' | 'return' | 'disable' | 'enable',
  reason?: string
): Promise<boolean> {
  try {
    const res = await fetch(`/api/executors/${encodeURIComponent(executorId)}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: reason ? JSON.stringify({ reason }) : undefined,
    });
    return res.ok;
  } catch {
    return false;
  }
}

export interface JobsPage {
  version: number;
  jobs: Generation[];
  /** Jobs deleted since `changedSince` (empty otherwise). */
  deleted: string[];
  /** Cursor of the next page (paged requests); null: no more. */
  next: string | null;
}

/** Which jobs a paged list shows: all (the timeline) or finished images with filters (the gallery). */
export interface JobQuery {
  doneOnly?: boolean;
  favorites?: boolean;
  q?: string;
  /** `w:h` of the image size (not the label: presets are only near their label's ratio). */
  ratio?: string;
}

async function getJobs(params: URLSearchParams): Promise<JobsPage> {
  const res = await fetch(`/api/jobs?${params}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch jobs (${res.status})`);
  }
  const data = await res.json();
  return {
    version: data.version,
    jobs: ((data.jobs || []) as V3JobRecord[]).map(jobToGeneration),
    deleted: data.deleted || [],
    next: data.next ?? null,
  };
}

/** Jobs that changed (or were deleted) since a version, for the poll loop. */
export function fetchJobChangesApi(changedSince: number): Promise<JobsPage> {
  return getJobs(new URLSearchParams({ changed_since: String(changedSince) }));
}

/** One page of a list, newest first, after `before` (the previous page's `next`). */
export function fetchJobsPageApi(query: JobQuery, limit: number, before?: string | null): Promise<JobsPage> {
  const p = new URLSearchParams({ limit: String(limit) });
  if (before) p.set('before', before);
  if (query.doneOnly) p.set('status', 'done');
  if (query.favorites) p.set('favorite', 'true');
  if (query.q?.trim()) p.set('q', query.q.trim());
  if (query.ratio) p.set('aspect', query.ratio);
  return getJobs(p);
}

/** One job with everything (its enhancer reasoning too); null when Studio has no such job. */
export async function fetchJobApi(id: string): Promise<Generation | null> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { cache: 'no-store' });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to fetch job (${res.status})`);
  return jobToGeneration(await res.json());
}

export interface GalleryFacets {
  total: number;
  favorites: number;
  sizes: { width: number; height: number; count: number }[];
}

/** Counts for the gallery's filters, over all finished images. */
export async function fetchGalleryFacetsApi(): Promise<GalleryFacets | null> {
  try {
    const res = await fetch('/api/gallery/facets', { cache: 'no-store' });
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

export interface JobSubmission {
  id: string;
  prompt: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  enhance: boolean;
  /** The prompt rewriter's token limit; omitted: Studio's default. */
  enhance_max_tokens?: number;
  /** A cloud enhancer's id; omitted: the built-in enhancer. */
  enhancer?: string;
  tiling: boolean;
  meta: JobMeta;
  /** Reference images in prompt order: uploads (hash) or earlier results (job id). */
  references?: ({ upload: string } | { image: string })[];
}

/**
 * Submit jobs to Studio (`POST /api/jobs`), one request each; ids are the client's.
 */
export async function submitBatchJobsApi(
  payloads: JobSubmission[]
): Promise<{ ok: boolean; error?: string }> {
  try {
    for (const p of payloads) {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(p),
      });
      if (res.status === 422) {
        const errData = await res.json().catch(() => ({}));
        return { ok: false, error: errData.rejected || errData.error || 'Job rejected: no capable executor' };
      }
      if (!res.ok) {
        return { ok: false, error: `Job submission failed (${res.status}): ${await res.text()}` };
      }
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : 'Network error submitting jobs' };
  }
}

/**
 * Cancel a job; returns the job as Studio left it. A job that finished before the cancel
 * arrived stays done. `null`: no answer (the next poll shows the real state).
 */
export async function cancelJobApi(jobId: string): Promise<Generation | null> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
    return res.ok ? jobToGeneration(await res.json()) : null;
  } catch {
    return null;
  }
}

/**
 * Retry a failed or cancelled job in place, optionally with another enhancer (`builtin`
 * after a cloud enhancer failed). `refused`: Studio will not retry it (409, e.g. it is not
 * failed any more); `missing`: no such job, or Studio could not be reached.
 */
export async function retryJobApi(jobId: string, enhancer?: string): Promise<'retried' | 'refused' | 'missing'> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/retry`, {
      method: 'POST',
      ...(enhancer ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enhancer }) } : {}),
    });
    if (res.ok) return 'retried';
    return res.status === 409 ? 'refused' : 'missing';
  } catch {
    return 'missing';
  }
}

export async function deleteJobApi(jobId: string): Promise<boolean> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    return res.ok || res.status === 404;
  } catch {
    return false;
  }
}

export async function setFavoriteApi(jobId: string, favorite: boolean): Promise<boolean> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ favorite }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** The UI's preferences, kept in Studio (`{}` until first saved). */
export async function fetchPreferencesApi(): Promise<Partial<UserPreferences> | null> {
  try {
    const res = await fetch('/api/preferences', { cache: 'no-store' });
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}

export async function savePreferencesApi(prefs: UserPreferences): Promise<void> {
  try {
    await fetch('/api/preferences', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prefs),
    });
  } catch {
    // retried with the next change
  }
}
