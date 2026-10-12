import type { JobTimings } from '../types';

const STAGE_KEYS = ['enhance_s', 'encode_s', 'denoise_s', 'decode_s'] as const;

/** Time spent generating: the sum of the stage times, excluding queue waits. */
export function generationSeconds(t?: JobTimings | null): number | undefined {
  if (!t) return undefined;
  const parts = STAGE_KEYS.map((k) => t[k]).filter((v): v is number => typeof v === 'number');
  return parts.length ? parts.reduce((a, b) => a + b, 0) : undefined;
}

/** Time spent waiting for a GPU: submit-to-finish minus generation time. */
export function queueSeconds(t?: JobTimings | null): number | undefined {
  const gen = generationSeconds(t);
  if (t?.total_s == null || gen == null) return undefined;
  return Math.max(0, t.total_s - gen);
}

export function formatDuration(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

export function formatGiB(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}
