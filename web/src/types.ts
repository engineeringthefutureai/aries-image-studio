export type AspectRatio = '1:1' | '4:3' | '3:4' | '3:2' | '2:3' | '16:9' | '9:16';

export interface AspectDimension {
  width: number;
  height: number;
  label: AspectRatio;
  ratioWidth: number;
  ratioHeight: number;
}

export const ASPECT_RATIOS: Record<AspectRatio, AspectDimension> = {
  '1:1': { width: 2048, height: 2048, label: '1:1', ratioWidth: 1, ratioHeight: 1 },
  '4:3': { width: 2400, height: 1792, label: '4:3', ratioWidth: 4, ratioHeight: 3 },
  '3:4': { width: 1792, height: 2400, label: '3:4', ratioWidth: 3, ratioHeight: 4 },
  '3:2': { width: 2528, height: 1696, label: '3:2', ratioWidth: 3, ratioHeight: 2 },
  '2:3': { width: 1696, height: 2528, label: '2:3', ratioWidth: 2, ratioHeight: 3 },
  '16:9': { width: 2752, height: 1536, label: '16:9', ratioWidth: 16, ratioHeight: 9 },
  '9:16': { width: 1536, height: 2752, label: '9:16', ratioWidth: 9, ratioHeight: 16 },
};

export type SizeMultiplier = 'full' | 'half' | 'quarter';

export interface SizeMultiplierConfig {
  value: SizeMultiplier;
  label: string;
  badge: string;
  factor: number;
  description: string;
}

export const SIZE_MULTIPLIERS: Record<SizeMultiplier, SizeMultiplierConfig> = {
  full: {
    value: 'full',
    label: 'Full',
    badge: '1x',
    factor: 1.0,
    description: 'Native 2K · High detail',
  },
  half: {
    value: 'half',
    label: 'Half',
    badge: '1/2',
    factor: 0.5,
    description: '4× faster · Light VRAM',
  },
  quarter: {
    value: 'quarter',
    label: 'Quarter',
    badge: '1/4',
    factor: 0.25,
    description: 'Fastest preview · Minimal VRAM',
  },
};

export function calculateDimensions(
  aspect: AspectRatio,
  multiplier: SizeMultiplier = 'full'
): { width: number; height: number } {
  const base = ASPECT_RATIOS[aspect] || ASPECT_RATIOS['1:1'];
  const factor = SIZE_MULTIPLIERS[multiplier]?.factor ?? 1.0;
  return {
    width: Math.round((base.width * factor) / 16) * 16,
    height: Math.round((base.height * factor) / 16) * 16,
  };
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export type JobPhase =
  | 'queued'
  | 'waking'
  | 'loading_model'
  | 'enhancing'
  | 'encoding'
  | 'denoising'
  | 'decoding'
  | 'saving'
  | 'done';

export interface JobParams {
  width: number;
  height: number;
  aspect: AspectRatio;
  sizeMultiplier?: SizeMultiplier;
  steps: number;
  seed: number;
  quality: 'standard' | 'draft' | 'custom';
  enhance?: boolean;
}

/** Sub-stage of a running enhance job, as reported by the enhance worker. */
export type EnhanceSubStage = 'prefill' | 'thinking' | 'rewriting';

export interface EnhanceStats {
  prompt_tokens?: number;
  thought_tokens?: number;
  reply_tokens?: number;
  generated_tokens?: number;
  /** The worker's token limit: the finish line for the progress bar. */
  max_new_tokens?: number;
  prefill_s?: number;
  prefill_tps?: number;
  generation_tps?: number;
}

/** Per-stage timings merged by the manager as each stage finishes. */
export interface JobTimings {
  total_s?: number;
  enhance_s?: number;
  encode_s?: number;
  denoise_s?: number;
  decode_s?: number;
  save_s?: number;
  steps?: number;
  s_per_step?: number;
  decode_mode?: string;
  prompt_source?: string;
  prefill_s?: number;
  prefill_tps?: number;
  generation_tps?: number;
  prompt_tokens?: number;
  generated_tokens?: number;
  thought_tokens?: number;
  reply_tokens?: number;
}

export interface Generation {
  id: string;
  batchId: string;
  prompt: string;
  enhancedPrompt?: string;
  thinking?: string;
  params: JobParams;
  status: JobStatus;
  phase?: JobPhase;
  stage?: string;
  subStage?: EnhanceSubStage;
  enhanceStats?: EnhanceStats;
  step?: number;
  totalSteps?: number;
  secondsPerStep?: number;
  etaSeconds?: number;
  imageUrl?: string;
  error?: string;
  createdAt: number;
  completedAt?: number;
  timings?: JobTimings;
  favorite?: boolean;
  lineage?: {
    parentId?: string;
    seed?: number;
    variation?: boolean;
  };
  attempts?: number;
  /** GPU running the current stage; for finished images, the GPU that decoded it. */
  card?: number;
  /** Highest VRAM use of any stage of this job, in bytes. */
  peakVram?: number;
  crashClass?: string;
}

export interface GenerationTurn {
  batchId: string;
  prompt: string;
  enhancedPrompt?: string;
  thinking?: string;
  createdAt: number;
  params: JobParams;
  generations: Generation[];
}

export interface CardStatusInfo {
  card: number;
  pci: string;
  state: 'idle' | 'running' | 'unavailable' | string;
  stage: string | null;
  sub_stage?: EnhanceSubStage | null;
  pid: number | null;
  job_id?: string | null;
  unavailable_reason: string | null;
  vram_used?: number;
  vram_reserved?: number;
  vram_total?: number;
  vram_peak?: number;
  vram_overflow?: number;
}

export interface StageRunItem {
  id: number;
  job_id: string | null;
  stage: string;
  card: number;
  pid: number;
  started_at: number;
  ended_at: number | null;
  exit_code: number | null;
  jobs_processed: number;
  exit_reason: string | null;
  queue_depths: Record<string, number> | null;
}

/** Manager `/status` snapshot plus the derived overall state. */
export interface WorkerHealth {
  state: 'ready' | 'busy' | 'error' | 'offline';
  cards?: CardStatusInfo[];
  queue_depths?: Record<string, number>;
  decode_ram_floor_bytes?: number;
  host_available_ram_bytes?: number;
}

export interface ManagerJob {
  id: string;
  prompt: string;
  enhanced_prompt?: string | null;
  thinking?: string | null;
  aspect: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  enhance: boolean | number;
  decode_mode: string;
  status: string;
  stage: string;
  sub_stage?: EnhanceSubStage | null;
  attempts: number;
  worker_pid?: number | null;
  card?: number | null;
  step?: number;
  s_per_step?: number | null;
  eta_s?: number | null;
  timings?: JobTimings | null;
  peak_vram?: number;
  sha256?: string | null;
  error?: string | null;
  crash_class?: string | null;
  enhance_stats?: EnhanceStats | null;
  created_at: number;
  updated_at: number;
  completed_at?: number | null;
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  aspect: AspectRatio;
  sizeMultiplier: SizeMultiplier;
  quality: 'standard' | 'draft';
  steps: number;
  count: number;
  isSeedLocked: boolean;
  lockedSeed: number;
  enhance: boolean;
}

