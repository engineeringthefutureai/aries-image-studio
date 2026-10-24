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

/** Quality preset: a named step count, or `custom` (the user's own number of steps). */
export type Quality = 'draft' | 'standard' | 'high' | 'custom';

export const QUALITY_STEPS: Record<Exclude<Quality, 'custom'>, number> = {
  draft: 12,
  standard: 20,
  high: 40,
};

/** Steps Studio accepts (`MAX_STEPS` in studio/src/scheduler.rs). */
/** The prompt rewriter's token limit (reasoning and reply): default and the range Studio accepts. */
export const ENHANCE_TOKENS_DEFAULT = 4096;
/** Each reference image adds this much to the default (the worker's rule). */
export const ENHANCE_TOKENS_PER_REFERENCE = 2048;
export const enhanceTokensDefault = (references: number) =>
  Math.min(MAX_ENHANCE_TOKENS, ENHANCE_TOKENS_DEFAULT + ENHANCE_TOKENS_PER_REFERENCE * references);
export const MIN_ENHANCE_TOKENS = 256;
export const MAX_ENHANCE_TOKENS = 24576;

export const MIN_STEPS = 1;
export const MAX_STEPS = 100;

export const clampSteps = (steps: number) =>
  Math.min(MAX_STEPS, Math.max(MIN_STEPS, Math.round(Number.isFinite(steps) ? steps : QUALITY_STEPS.standard)));

/** The preset a step count belongs to, or `custom`. */
export function qualityForSteps(steps: number): Quality {
  const preset = (Object.keys(QUALITY_STEPS) as (keyof typeof QUALITY_STEPS)[]).find((q) => QUALITY_STEPS[q] === steps);
  return preset ?? 'custom';
}

/** Steps a job gets from the composer's quality and steps preferences. */
export const stepsFor = (quality: Quality, steps: number) =>
  quality === 'custom' ? clampSteps(steps) : QUALITY_STEPS[quality] ?? QUALITY_STEPS.standard;

/**
 * The image the composer was filled from (Rerun or Vary). When it had an enhanced prompt,
 * the composer offers both: the original (with its Enhance setting) or the enhanced text
 * (Enhance off).
 */
export interface ComposerSource {
  seed: number;
  original: string;
  enhanced?: string;
  /** The image's own Enhance setting, restored when switching back to the original. */
  originalEnhance: boolean;
  mode: 'original' | 'enhanced';
}

/** A job's reference image, stored in Studio by hash (docs/REFERENCE_IMAGES.md). */
export interface JobReference {
  sha256: string;
  width: number;
  height: number;
}

/**
 * A reference attached in the composer, in prompt order ("image 1" first): a file being
 * uploaded or uploaded (`upload`, its hash), or an earlier result (`image`, its job id).
 */
export interface ComposerReference {
  key: string;
  previewUrl: string;
  status: 'uploading' | 'ready' | 'failed';
  upload?: string;
  image?: string;
  error?: string;
}

export function calculateDimensions(
  aspect: AspectRatio,
  multiplier: SizeMultiplier = 'full'
): { width: number; height: number } {
  const base = ASPECT_RATIOS[aspect] || ASPECT_RATIOS['1:1'];
  const factor = SIZE_MULTIPLIERS[multiplier]?.factor ?? 1.0;
  return {
    width: Math.round((base.width * factor) / 32) * 32,
    height: Math.round((base.height * factor) / 32) * 32,
  };
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export type JobPhase =
  | 'queued'
  | 'waking'
  | 'enhancing'
  | 'encoding'
  | 'denoising'
  | 'decoding'
  | 'done';

export interface JobParams {
  width: number;
  height: number;
  aspect: AspectRatio;
  sizeMultiplier?: SizeMultiplier;
  steps: number;
  seed: number;
  quality: Quality;
  enhance?: boolean;
  /** The prompt rewriter's token limit (advanced; unset: the worker's default, 4096 + 2048 per reference image). */
  enhanceMaxTokens?: number;
  /** The cloud enhancer that rewrites the prompt: its model id (`claude-sonnet-5-5`); unset: the built-in one. */
  enhancer?: string;
  /** VAE tiling at decode (advanced; off by default). */
  tiling?: boolean;
}

/** Sub-stage of a running enhance job, as reported by the enhance worker (or Studio, for a
 * cloud enhancer: `waiting` for the provider's rate limit, then `rewriting`). */
export type EnhanceSubStage = 'prefill' | 'thinking' | 'rewriting' | 'waiting';

export interface EnhanceStats {
  prompt_tokens?: number;
  thought_tokens?: number;
  reply_tokens?: number;
  generated_tokens?: number;
  max_new_tokens?: number;
  prefill_s?: number;
  prefill_tps?: number;
  generation_tps?: number;
}

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
  /** The enhancer's final token limit; with `limit_raised_from` it was doubled from that
   * because the reasoning ran out of room; `thinking_cut` 1: cut short for the reply at the ceiling. */
  max_new_tokens?: number;
  limit_raised_from?: number;
  thinking_cut?: number;
  /** 1 when the encode was reused from another image of the send (same prompt and references). */
  shared?: number;
}

export interface StageProvenance {
  agent: string;
  executor: string;
  device: string;
  backend_kind: string;
  model: {
    id: string;
    rev: string;
    dtype: string;
  };
  torch_version: string;
  worker_version: string;
  timings?: Record<string, number>;
  peak_mem_bytes?: number;
}

export interface Generation {
  id: string;
  batchId: string;
  /** Position in its send; images of a turn are shown (and processed) in this order. */
  batchIndex?: number;
  prompt: string;
  enhancedPrompt?: string;
  /** The enhancer's reasoning: only on a job fetched alone (lists leave it out, see hasThinking). */
  thinking?: string;
  /** The job has enhancer reasoning (fetched on demand: utils/thinking.ts). */
  hasThinking?: boolean;
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
  /** 1024 px thumbnail (the viewer's placeholder and navigator). */
  thumbUrl?: string;
  /** `srcset` of the thumbnails (512, 768 and 1024 px), for grid tiles. */
  thumbSrcSet?: string;
  error?: string;
  blockedReason?: string;
  provenance?: Record<string, StageProvenance>;
  createdAt: number;
  completedAt?: number;
  timings?: JobTimings;
  favorite?: boolean;
  card?: number;
  executor?: string;
  peakVram?: number;
  /** Reference images in prompt order (none for text-only jobs). */
  references?: JobReference[];
}

/** One executor as the header's status pill sees it. */
export interface CardStatusInfo {
  card: number;
  state: 'idle' | 'running' | 'unavailable';
  stage: string | null;
}

export interface SystemStageCap {
  stage: string;
  models: string[];
  max_px: number;
  modes?: string[];
}

export interface SystemExecutorInfo {
  key: string;
  id: string;
  agent_id: string;
  device: string;
  mem_total_bytes: number;
  mem_used_bytes: number;
  mem_available_bytes: number;
  overflow_host_bytes: number;
  stages: SystemStageCap[];
  state: string;
  current_stage?: string | null;
  current_model_id?: string | null;
  current_model_rev?: string | null;
  current_task_id?: string | null;
  worker_pid?: number | null;
  load_s?: number | null;
  disabled_reason?: string | null;
  lent: boolean;
  pending_lend: boolean;
}

export interface SystemAgentInfo {
  id: string;
  agent_version: string;
  backend_kind: string;
  metrics_url: string;
  online: boolean;
  host_ram_total_bytes: number;
  host_ram_available_bytes: number;
  scrape_age_s?: number | null;
  scrape_stale: boolean;
  executors: SystemExecutorInfo[];
}

export interface SystemQueueInfo {
  stage: string;
  queued: number;
  running: number;
  blocked: number;
  oldest_wait_s: number;
  blocked_reasons: string[];
}

export interface SizeCapability {
  width: number;
  height: number;
  px: number;
  servable_now: boolean;
  servable_ever: boolean;
  reason?: string | null;
}

export interface StageCapability {
  stage: string;
  servable_now: boolean;
  servable_ever: boolean;
  reason?: string | null;
}

export interface CloudEnhancerInfo {
  /** The model id; what a job names in `enhancer`. */
  id: string;
  /** `anthropic` | `google`. */
  provider: string;
  /** For people, e.g. "Claude Sonnet 5.5". */
  label: string;
  model: string;
  max_references: number;
}

export interface CapabilitiesInfo {
  max_px_now: number;
  max_px_ever: number;
  /** Most reference images a job may carry (0 or missing: none accepted). */
  max_references_now?: number;
  max_references_ever?: number;
  enhance_now: boolean;
  enhance_ever: boolean;
  /** Most reference images Enhance reads (an image-aware enhancer); 0 or missing: Enhance is off with references. */
  enhance_max_references_now?: number;
  enhance_max_references_ever?: number;
  /** Cloud prompt enhancers jobs may name (docs/CLOUD_ENHANCERS.md); empty unless Studio switched them on. */
  cloud_enhancers?: CloudEnhancerInfo[];
  sizes: SizeCapability[];
  stages: StageCapability[];
}

export interface WorkerHealth {
  state: 'ready' | 'busy' | 'error' | 'offline';
  cards?: CardStatusInfo[];
  agents?: SystemAgentInfo[];
  queues?: SystemQueueInfo[];
}

export interface V3TaskRecord {
  id: string;
  job_id: string;
  seq: number;
  stage: string;
  status: string;
  attempts: number;
  current_attempt: number;
  assigned_agent?: string | null;
  assigned_executor?: string | null;
  blocked_reason?: string | null;
  step: number;
  steps: number;
  s_per_step: number;
  /** enhance only: live sub-stage and token stats from the worker */
  sub_stage?: EnhanceSubStage | null;
  stats?: EnhanceStats | null;
  error_class?: string | null;
  error_detail?: string | null;
  queued_at?: number | null;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface V3JobRecord {
  id: string;
  status: string;
  prompt: string;
  enhanced_prompt?: string | null;
  /** Left out of job lists (`has_thinking` says whether there is some). */
  thinking?: string | null;
  has_thinking?: boolean;
  negative_prompt: string;
  /** UI metadata the SPA stored with the job (see services/api.ts JobMeta). */
  meta?: Record<string, unknown>;
  favorite?: boolean;
  width: number;
  height: number;
  steps: number;
  seed: number;
  true_cfg_scale: number;
  enhance: boolean;
  enhance_max_tokens?: number | null;
  /** The cloud enhancer that rewrites the prompt; unset: the built-in one. */
  enhancer?: string | null;
  tiling?: boolean;
  wh_ratio?: string | null;
  /** The image enhancer's aspect answer: `<imageN>` keeps that reference's aspect (not used yet). */
  ratio_follow?: string | null;
  model_id?: string | null;
  model_rev?: string | null;
  error_class?: string | null;
  error_detail?: string | null;
  blocked_reason?: string | null;
  version: number;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  tasks: V3TaskRecord[];
  provenance: Record<string, StageProvenance>;
  references?: JobReference[];
}

/** One row of the advanced table: a full parameter set sent with the shared prompt and seed. */
export interface ComposerRow {
  aspect: AspectRatio;
  sizeMultiplier: SizeMultiplier;
  steps: number;
  enhance: boolean;
  tiling: boolean;
}

/** Most images one send may queue (rows × count). */
export const MAX_IMAGES_PER_SEND = 16;

export interface UserPreferences {
  theme: 'light' | 'dark';
  aspect: AspectRatio;
  sizeMultiplier: SizeMultiplier;
  quality: Quality;
  steps: number;
  count: number;
  isSeedLocked: boolean;
  lockedSeed: number;
  enhance: boolean;
  /** Advanced options shown: custom steps, fixed seed, tiling and extra rows apply only then. */
  advancedOpen: boolean;
  tiling: boolean;
  /** Advanced: the prompt rewriter's token limit for Enhance rows. */
  enhanceMaxTokens: number;
  /** Who rewrites the prompt: `builtin`, or a cloud enhancer's id when Studio offers it. */
  enhancer: string;
  /** Rows after the first (the first row is the main settings above). */
  extraRows: ComposerRow[];
}
