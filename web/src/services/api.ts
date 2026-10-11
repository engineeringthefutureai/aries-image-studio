import type { CardStatusInfo, JobPhase, ManagerJob, StageRunItem, WorkerHealth } from '../types';

export interface JobImageResult {
  blob: Blob;
  url: string;
  sha256?: string;
}

/**
 * Fetch the manager's status snapshot for the status pill and panel.
 */
export async function fetchWorkerHealth(): Promise<WorkerHealth> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const res = await fetch('/status', { signal: controller.signal, cache: 'no-store' });
    if (!res.ok) {
      return { state: 'error' };
    }
    const data = await res.json();
    const cards: CardStatusInfo[] = data.cards || [];

    let state: WorkerHealth['state'] = 'ready';
    if (cards.length > 0 && cards.every((c) => c.state === 'unavailable')) {
      state = 'error';
    } else if (cards.some((c) => c.state === 'running')) {
      state = 'busy';
    }

    return {
      state,
      cards,
      queue_depths: data.queue_depths,
      decode_ram_floor_bytes: data.decode_ram_floor_bytes,
      host_available_ram_bytes: data.host_available_ram_bytes,
    };
  } catch {
    return { state: 'offline' };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Fetch recent stage worker runs and eviction audit logs from Manager.
 */
export async function fetchStageRunsApi(limit: number = 10): Promise<StageRunItem[]> {
  try {
    const res = await fetch(`/stage_runs?limit=${limit}`, { cache: 'no-store' });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

/**
 * Fetch all currently active and stored jobs from Manager.
 */
export async function fetchCurrentJobsApi(): Promise<ManagerJob[]> {
  const res = await fetch('/jobs', { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch current jobs (${res.status})`);
  }
  return res.json();
}

/**
 * Fetch full details for a single job from Manager.
 */
export async function fetchJobDetailApi(jobId: string): Promise<ManagerJob | null> {
  try {
    const res = await fetch(`/jobs/${jobId}`, { cache: 'no-store' });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export interface JobSubmission {
  id: string;
  prompt: string;
  aspect: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  enhance: boolean;
  decode_mode: 'untiled' | 'tiled';
}

/**
 * Submit a batch of jobs in one call. Ids are client-generated ULIDs, so a resubmit is idempotent.
 */
export async function submitBatchJobsApi(
  payloads: JobSubmission[]
): Promise<{ ok: boolean; jobIds?: string[]; error?: string }> {
  try {
    const res = await fetch('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloads),
    });

    if (res.status === 503) {
      const errText = await res.text();
      return { ok: false, error: `Manager storage watermark exceeded: ${errText}` };
    }

    if (!res.ok) {
      const errText = await res.text();
      return { ok: false, error: `Batch submission failed (${res.status}): ${errText}` };
    }

    const data = await res.json();
    return { ok: true, jobIds: data.job_ids };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : 'Network error submitting batch jobs' };
  }
}

/**
 * Cancel a queued or running job on Manager.
 */
export async function cancelJobApi(jobId: string): Promise<void> {
  try {
    await fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
  } catch (err) {
    console.warn(`Failed to cancel job ${jobId}:`, err);
  }
}

/**
 * Fetch finished PNG image with SHA-256 integrity header.
 */
export async function fetchJobImageBlob(jobId: string): Promise<JobImageResult> {
  const res = await fetch(`/jobs/${jobId}/image`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to load image (${res.status})`);
  }
  const sha256 = res.headers.get('X-Content-SHA256') || undefined;
  const blob = await res.blob();
  return { blob, url: URL.createObjectURL(blob), sha256 };
}

/**
 * Acknowledge receipt of finished image to Manager (triggers intermediate & image GC).
 */
export async function acknowledgeJobImageApi(jobId: string): Promise<void> {
  try {
    const res = await fetch(`/jobs/${jobId}/image`, { method: 'DELETE' });
    if (!res.ok) {
      console.warn(`Acknowledge job ${jobId} returned status ${res.status}`);
    }
  } catch (err) {
    console.warn(`Failed to acknowledge job ${jobId}:`, err);
  }
}

/**
 * Map a manager job status to the UI phase. Queued stages map to 'queued'; the
 * job's `stage` tells which queue it waits in.
 */
export function mapStatusToPhase(status: string): JobPhase {
  switch (status) {
    case 'enhancing':
      return 'enhancing';
    case 'encoding':
      return 'encoding';
    case 'denoising':
      return 'denoising';
    case 'decoding':
      return 'decoding';
    case 'stored':
    case 'acknowledged':
      return 'done';
    default:
      return 'queued';
  }
}
