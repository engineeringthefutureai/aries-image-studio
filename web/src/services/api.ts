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
}

/** The composer's aspect label for a size (exact or scaled-down preset), else "w:h". */
function aspectOf(width: number, height: number): AspectRatio {
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
    if (stage === 'decode') timings.decode_mode = 'untiled'; // the only mode V3 schedules
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
    prompt: v3.prompt,
    enhancedPrompt: v3.enhanced_prompt ? sanitizeEnhancedPrompt(v3.enhanced_prompt) : undefined,
    thinking: v3.thinking || undefined,
    params: {
      width: v3.width,
      height: v3.height,
      aspect: meta.aspect || aspectOf(v3.width, v3.height),
      sizeMultiplier: meta.size_multiplier,
      steps: v3.steps,
      seed: v3.seed,
      quality: meta.quality || 'custom',
      enhance: v3.enhance,
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
    thumbUrl: done ? `/api/images/${encodeURIComponent(v3.id)}/thumb` : undefined,
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
  };
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
  /** Jobs deleted since `changedSince` (empty for a full load). */
  deleted: string[];
}

/**
 * Jobs from Studio (`/api/jobs`): everything, or what changed since a version.
 */
export async function fetchJobsApi(changedSince?: number): Promise<JobsPage> {
  const url = changedSince !== undefined ? `/api/jobs?changed_since=${changedSince}` : '/api/jobs';
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch jobs (${res.status})`);
  }
  const data = await res.json();
  return {
    version: data.version,
    jobs: ((data.jobs || []) as V3JobRecord[]).map(jobToGeneration),
    deleted: data.deleted || [],
  };
}

export interface JobSubmission {
  id: string;
  prompt: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  enhance: boolean;
  meta: JobMeta;
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

export async function cancelJobApi(jobId: string): Promise<void> {
  try {
    await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  } catch {
    // the next poll shows the real state
  }
}

export async function retryJobApi(jobId: string): Promise<boolean> {
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
    return res.ok;
  } catch {
    return false;
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
