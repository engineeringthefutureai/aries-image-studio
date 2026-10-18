import React, { useState, useRef, useEffect } from 'react';
import {
  type AspectRatio,
  type ComposerRow,
  type JobParams,
  type UserPreferences,
  type SizeMultiplier,
  type CapabilitiesInfo,
  type Quality,
  type ComposerSource,
  type ComposerReference,
  ASPECT_RATIOS,
  SIZE_MULTIPLIERS,
  QUALITY_STEPS,
  MIN_STEPS,
  MAX_STEPS,
  ENHANCE_TOKENS_DEFAULT,
  ENHANCE_TOKENS_PER_REFERENCE,
  enhanceTokensDefault,
  MIN_ENHANCE_TOKENS,
  MAX_ENHANCE_TOKENS,
  calculateDimensions,
  clampSteps,
  qualityForSteps,
  MAX_IMAGES_PER_SEND,
} from '../types';
import { buildJobs, sendSeeds, imagesPerSend, mainRow } from '../utils/composer';
import { REFERENCE_TYPES, moveEarlier, releasePreview } from '../utils/references';
import { uploadReferenceApi } from '../services/api';
import { generateUlid } from '../utils/ulid';
import {
  Sparkles,
  Dices,
  Send,
  ChevronDown,
  ChevronLeft,
  Check,
  Gauge,
  CornerDownRight,
  X,
  SlidersHorizontal,
  Plus,
  Grid2x2,
  ImagePlus,
  Image as ImageIcon,
  Loader2,
  AlertCircle,
} from 'lucide-react';

interface ComposerProps {
  preferences: UserPreferences;
  onUpdatePreferences: (updates: Partial<UserPreferences>) => void;
  /** One send: every job of the turn (rows × count, see utils/composer.ts) and its references. */
  onSubmit: (prompt: string, jobs: JobParams[], references: ComposerReference[]) => void;
  isSubmitting: boolean;
  workerState?: string;
  capabilities?: CapabilitiesInfo | null;
  prefilledPrompt?: string | null;
  prefilledAspect?: AspectRatio | null;
  /** The image the composer was filled from (Rerun / Vary), if any. */
  source?: ComposerSource | null;
  onSourceChange?: (source: ComposerSource | null) => void;
  onClearPrefill?: () => void;
  textareaRef?: React.RefObject<HTMLTextAreaElement | null>;
  /** Reference images, in prompt order ("image 1" first). */
  references: ComposerReference[];
  onReferencesChange: React.Dispatch<React.SetStateAction<ComposerReference[]>>;
  /** Most references a job may carry (0: Studio's agents accept none). */
  maxReferences: number;
}

function getAspectIconDims(ratioWidth: number, ratioHeight: number, maxSize = 15) {
  if (ratioWidth >= ratioHeight) {
    const width = maxSize;
    const height = Math.max(7, Math.round((maxSize * ratioHeight) / ratioWidth));
    return { width, height };
  } else {
    const height = maxSize;
    const width = Math.max(7, Math.round((maxSize * ratioWidth) / ratioHeight));
    return { width, height };
  }
}

const QUALITY_LABELS: Record<Exclude<Quality, 'custom'>, string> = {
  draft: 'Draft',
  standard: 'Standard',
  high: 'High',
};
const segmentOn = 'bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] shadow-2xs font-semibold';
const segmentOff = 'text-[#8c7169] dark:text-[#a09088]';
const chip = 'flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320]';
const selectCls =
  'px-1.5 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320] text-[#1c1c18] dark:text-[#f3f0e9] font-medium cursor-pointer outline-none';
const SIZE_SHORT: Record<SizeMultiplier, string> = { full: 'Full', half: '½', quarter: '¼' };
const TILING_TITLE =
  'VAE tiling at decode: faster and needs less memory, but can leave visible seams. Off by default.';

/** A whole-number field that can be typed in freely and settles on blur or Enter. */
const NumberField: React.FC<{
  value: number;
  onChange: (n: number) => void;
  label: string;
  title: string;
  min: number;
  max: number;
  unit: string;
  width?: string;
}> = ({ value, onChange, label, title, min, max, unit, width = 'w-8' }) => {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <label className={chip} title={title}>
      <input
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        step={1}
        value={draft ?? String(value)}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value !== '' && Number.isInteger(n) && n >= min && n <= max) onChange(n);
        }}
        onBlur={() => {
          // Out of range snaps to the nearest limit; empty or not a number keeps the last value.
          if (draft !== null && draft.trim() !== '' && Number.isFinite(Number(draft))) {
            onChange(Math.min(max, Math.max(min, Math.round(Number(draft)))));
          }
          setDraft(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        }}
        aria-label={label}
        className={`${width} bg-transparent text-right font-semibold tabular-nums text-[#1c1c18] dark:text-[#f3f0e9] outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none`}
      />
      <span className="opacity-60">{unit}</span>
    </label>
  );
};

/** A steps field (1–100). */
const StepsField: React.FC<{ value: number; onChange: (steps: number) => void; label: string }> = ({
  value,
  onChange,
  label,
}) => (
  <NumberField
    value={value}
    onChange={(n) => onChange(clampSteps(n))}
    label={label}
    title={`Denoising steps (${MIN_STEPS}–${MAX_STEPS})`}
    min={MIN_STEPS}
    max={MAX_STEPS}
    unit="steps"
  />
);

/** An on/off chip (Enhance, Tiling) inside the advanced panel. */
const ToggleChip: React.FC<{
  on: boolean;
  onClick: () => void;
  label: string;
  title: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}> = ({ on, onClick, label, title, icon, disabled }) => (
  <button
    type="button"
    onClick={onClick}
    aria-pressed={on}
    title={title}
    disabled={disabled}
    className={`flex items-center gap-1 px-2 py-1 rounded-lg border font-medium transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
      on
        ? 'bg-amber-500/10 text-amber-900 dark:text-amber-300 border-amber-500/30'
        : 'bg-[#f1eee7] dark:bg-[#232320] text-[#8c7169] dark:text-[#a09088] border-transparent'
    }`}
  >
    {icon}
    {label}
    {on && <Check className="w-3 h-3" />}
  </button>
);

/** Shows or hides its children with a height animation. Menus inside may overflow once it is open. */
const Fold: React.FC<{ open: boolean; children: React.ReactNode }> = ({ open, children }) => {
  const [settled, setSettled] = useState(open);
  useEffect(() => {
    const t = window.setTimeout(() => setSettled(open), 220);
    return () => window.clearTimeout(t);
  }, [open]);
  return (
    <div
      className={`grid transition-[grid-template-rows,opacity] duration-200 ease-out ${
        open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
      }`}
      inert={!open}
    >
      <div className={`min-h-0 ${open && settled ? 'overflow-visible' : 'overflow-hidden'}`}>{children}</div>
    </div>
  );
};

/** Why Enhance is off for a send with references, given the most images an enhancer reads. */
const enhanceOffTitle = (most: number) =>
  most > 0
    ? `Enhance reads at most ${most} reference ${most === 1 ? 'image' : 'images'}`
    : 'Enhance is off with reference images: the text enhancer cannot see them (install Qwen-Image-2.1-PE-I2I)';

/** One attached reference: its number in the prompt, remove, and move earlier. */
const ReferenceThumb: React.FC<{
  reference: ComposerReference;
  index: number;
  onRemove: () => void;
  onMoveEarlier?: () => void;
}> = ({ reference, index, onRemove, onMoveEarlier }) => (
  <div
    className={`group relative w-14 h-14 shrink-0 rounded-lg overflow-hidden bg-[#f1eee7] dark:bg-[#232320] ${
      reference.status === 'failed' ? 'ring-2 ring-rose-500' : ''
    }`}
    title={reference.status === 'failed' ? reference.error || 'Upload failed' : `Image ${index + 1}`}
    data-reference={index + 1}
  >
    {reference.previewUrl && <img src={reference.previewUrl} alt={`Reference image ${index + 1}`} className="w-full h-full object-cover" />}
    {reference.status === 'uploading' && (
      <div className="absolute inset-0 flex items-center justify-center bg-black/35" role="status" aria-label="Uploading">
        <Loader2 className="w-4 h-4 text-white animate-spin" />
      </div>
    )}
    {reference.status === 'failed' && (
      <div className="absolute inset-0 flex items-center justify-center bg-rose-900/40">
        <AlertCircle className="w-4 h-4 text-white" />
      </div>
    )}
    <span className="absolute bottom-0.5 left-0.5 min-w-4 h-4 px-1 rounded bg-black/60 text-white font-mono text-[10px] leading-4 text-center">
      {index + 1}
    </span>
    <button
      type="button"
      onClick={onRemove}
      className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full bg-black/60 text-white flex items-center justify-center cursor-pointer"
      title="Remove"
      aria-label={`Remove reference image ${index + 1}`}
    >
      <X className="w-3 h-3" />
    </button>
    {onMoveEarlier && (
      <button
        type="button"
        onClick={onMoveEarlier}
        className="absolute bottom-0.5 right-0.5 w-4 h-4 rounded bg-black/60 text-white flex items-center justify-center cursor-pointer sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100"
        title={`Make it image ${index}`}
        aria-label={`Move reference image ${index + 1} earlier`}
      >
        <ChevronLeft className="w-3 h-3" />
      </button>
    )}
  </div>
);

const rowSummary = (row: ComposerRow) =>
  [row.aspect, SIZE_MULTIPLIERS[row.sizeMultiplier].label, `${row.steps} st`, row.enhance && 'Enhance', row.tiling && 'Tiled']
    .filter(Boolean)
    .join(' · ');

export const Composer: React.FC<ComposerProps> = ({
  preferences,
  onUpdatePreferences,
  onSubmit,
  isSubmitting,
  capabilities,
  prefilledPrompt,
  prefilledAspect,
  source,
  onSourceChange,
  onClearPrefill,
  textareaRef,
  references,
  onReferencesChange,
  maxReferences,
}) => {
  const [prompt, setPrompt] = useState('');
  // A short note about attaching (too many, wrong type); clears itself.
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [isAspectOpen, setIsAspectOpen] = useState(false);
  const [isScaleOpen, setIsScaleOpen] = useState(false);
  // Folded to a one-line pill until the prompt is focused; a send, Esc or a tap outside folds it.
  const [expanded, setExpanded] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const inputRef = textareaRef || internalRef;

  // Sync prefill from parent (e.g. from sample prompt or reuse settings)
  useEffect(() => {
    if (prefilledPrompt !== undefined && prefilledPrompt !== null) {
      setPrompt(prefilledPrompt);
      if (prefilledAspect) {
        onUpdatePreferences({ aspect: prefilledAspect });
      }
      onClearPrefill?.();
      // Rerun / Vary: open it to show what was filled in (a phone may ignore the focus).
      setExpanded(true);
      inputRef.current?.focus();
    }
  }, [prefilledPrompt, prefilledAspect, onClearPrefill, onUpdatePreferences, inputRef]);

  // Auto-resize textarea; one line while folded
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    if (!expanded) {
      el.style.height = '1.625em';
      el.scrollTop = 0;
      return;
    }
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [prompt, inputRef, expanded]);

  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(null), 4000);
    return () => window.clearTimeout(t);
  }, [notice]);

  // A reference added from elsewhere (Rerun, "Use as reference") opens the composer to show it.
  const refCount = useRef(references.length);
  useEffect(() => {
    if (references.length > refCount.current) setExpanded(true);
    refCount.current = references.length;
  }, [references.length]);

  /** Attach image files (picker, drop, paste): each uploads at once, up to the limit. */
  const attachFiles = (files: File[]) => {
    const images = files.filter((f) => REFERENCE_TYPES.includes(f.type));
    if (images.length === 0) {
      if (files.length) setNotice('PNG, JPEG or WebP images only');
      return;
    }
    if (maxReferences === 0) {
      setNotice('No agent accepts reference images');
      return;
    }
    const room = maxReferences - references.length;
    if (room <= 0 || images.length > room) setNotice(`At most ${maxReferences} reference images`);
    const added = images.slice(0, Math.max(0, room)).map((file) => ({
      file,
      ref: { key: generateUlid(), previewUrl: URL.createObjectURL(file), status: 'uploading' } as ComposerReference,
    }));
    if (added.length === 0) return;
    onReferencesChange((prev) => [...prev, ...added.map((a) => a.ref)]);
    setExpanded(true);
    for (const { file, ref } of added) {
      const settle = (patch: Partial<ComposerReference>) =>
        onReferencesChange((prev) => prev.map((r) => (r.key === ref.key ? { ...r, ...patch } : r)));
      uploadReferenceApi(file).then(
        (up) => settle({ status: 'ready', upload: up.sha256 }),
        (err) => settle({ status: 'failed', error: err instanceof Error ? err.message : 'Upload failed' }),
      );
    }
  };
  const attachRef = useRef(attachFiles);
  useEffect(() => {
    attachRef.current = attachFiles;
  });

  // Images dropped anywhere on the page are attached; the composer shows where they go.
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files');
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth += 1;
      setDragging(true);
      setExpanded(true);
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      attachRef.current(Array.from(e.dataTransfer?.files || []));
    };
    window.addEventListener('dragenter', onEnter);
    window.addEventListener('dragleave', onLeave);
    window.addEventListener('dragover', onOver);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onEnter);
      window.removeEventListener('dragleave', onLeave);
      window.removeEventListener('dragover', onOver);
      window.removeEventListener('drop', onDrop);
    };
  }, []);

  const removeReference = (key: string) =>
    onReferencesChange((prev) => {
      prev.filter((r) => r.key === key).forEach(releasePreview);
      return prev.filter((r) => r.key !== key);
    });

  const collapse = () => {
    setExpanded(false);
    setIsAspectOpen(false);
    setIsScaleOpen(false);
    inputRef.current?.blur();
  };
  const expand = () => {
    setExpanded(true);
    inputRef.current?.focus();
  };

  // A tap or click outside folds it (the prompt and settings stay).
  useEffect(() => {
    if (!expanded) return;
    const onDown = (e: PointerEvent) => {
      if (rootRef.current?.contains(e.target as Node)) return;
      setExpanded(false);
      setIsAspectOpen(false);
      setIsScaleOpen(false);
      inputRef.current?.blur();
    };
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [expanded, inputRef]);

  const handleGenerate = () => {
    if (!prompt.trim() || isSubmitting || overLimit || referencesPending) return;
    // Without an image-aware enhancer Enhance is off for a send with references: the text
    // rewriter cannot see them and could rewrite away what the edit asks for.
    const jobs = buildJobs(preferences, sendSeeds(preferences)).map((j) =>
      enhanceBlocked ? { ...j, enhance: false } : j,
    );
    onSubmit(prompt.trim(), jobs, references);
    // Like the prompt, attachments belong to the send; Rerun brings them back.
    onReferencesChange((prev) => {
      prev.forEach(releasePreview);
      return [];
    });
    // Advanced is for one send: close it and drop its settings, so nothing hidden carries
    // over (custom steps back to Standard, random seed, no tiling, default token limit, one row).
    if (preferences.advancedOpen) {
      onUpdatePreferences({
        advancedOpen: false,
        extraRows: [],
        tiling: false,
        enhanceMaxTokens: 0,
        isSeedLocked: false,
        ...(preferences.quality === 'custom' ? { quality: 'standard', steps: QUALITY_STEPS.standard } : {}),
      });
    }
    // The prompt stays on the new turn (copy it or Rerun from there); the field starts fresh.
    setPrompt('');
    collapse();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  // Original: the image's prompt with its own Enhance setting. Enhanced: the rewritten
  // prompt with Enhance off, so it is used as written.
  const switchSource = (mode: ComposerSource['mode']) => {
    if (!source || mode === source.mode) return;
    const text = mode === 'enhanced' ? source.enhanced : source.original;
    if (text === undefined) return;
    setPrompt(text);
    onUpdatePreferences({ enhance: mode === 'enhanced' ? false : source.originalEnhance });
    onSourceChange?.({ ...source, mode });
    inputRef.current?.focus();
  };

  const rollSeed = () => {
    const nextSeed = Math.floor(Math.random() * 1000000);
    onUpdatePreferences({ lockedSeed: nextSeed });
  };

  const advanced = preferences.advancedOpen;
  const hasReferences = references.length > 0;
  // With references, Enhance needs an executor whose enhancer reads that many images.
  const enhanceMostRefs = capabilities?.enhance_max_references_ever ?? 0;
  const enhanceBlocked = hasReferences && references.length > enhanceMostRefs;
  const anyEnhance = !enhanceBlocked && (preferences.enhance || preferences.extraRows.some((r) => r.enhance));
  // Still uploading, or failed (remove it to send): the job would not have it.
  const referencesPending = references.some((r) => r.status !== 'ready');
  const main = { ...mainRow(preferences), ...(enhanceBlocked ? { enhance: false } : {}) };
  const count = Math.max(1, preferences.count || 1);
  const totalImages = imagesPerSend(preferences);
  const overLimit = totalImages > MAX_IMAGES_PER_SEND;
  // With Advanced closed a custom step count is not used: Standard is.
  const shownQuality: Quality = advanced || preferences.quality !== 'custom' ? preferences.quality : 'standard';
  const updateRow = (i: number, patch: Partial<ComposerRow>) =>
    onUpdatePreferences({ extraRows: preferences.extraRows.map((r, j) => (j === i ? { ...r, ...patch } : r)) });
  const addRow = () => {
    const last = preferences.extraRows[preferences.extraRows.length - 1] ?? main;
    onUpdatePreferences({ extraRows: [...preferences.extraRows, { ...last }] });
  };
  const canAddRow = (preferences.extraRows.length + 2) * count <= MAX_IMAGES_PER_SEND;
  // What the next send makes, shown in the folded pill.
  const summary = [
    main.aspect,
    SIZE_SHORT[main.sizeMultiplier],
    `${main.steps} st`,
    `${totalImages} img`,
  ].join(' · ');
  const sendTitle = referencesPending
    ? references.some((r) => r.status === 'failed')
      ? 'A reference image failed to upload: remove it to send'
      : 'Uploading reference images…'
    : 'Generate (Enter)';
  const summaryTitle = `${totalImages} ${totalImages === 1 ? 'image' : 'images'}: ${rowSummary(main)}${
    advanced ? ', Advanced on' : ''
  }${hasReferences ? `, ${references.length} reference ${references.length === 1 ? 'image' : 'images'}` : ''}`;

  return (
    <div
      ref={rootRef}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && expanded) {
          e.stopPropagation();
          collapse();
        }
      }}
      className="fixed bottom-0 left-0 right-0 sm:left-auto z-30 pointer-events-none px-4 pb-[max(1rem,env(safe-area-inset-bottom))]"
    >
      <div
        className={`ml-auto pointer-events-auto transition-[width] duration-200 ease-out w-full ${
          expanded ? 'sm:w-[min(56rem,calc(100vw-2rem))]' : 'sm:w-[26rem]'
        }`}
      >
        <div
          className={`relative bg-[#fcf9f2] dark:bg-[#1a1a17] border shadow-2xl transition-[border-radius,padding] duration-200 ${
            expanded ? 'rounded-2xl p-3 sm:p-4' : 'rounded-3xl py-1.5 pl-4 pr-1.5'
          } ${dragging ? 'border-dashed border-[#a53705] ring-2 ring-[#a53705]/30' : 'border-[#e5e2db] dark:border-[#2e2e2a]'}`}
        >
          <input
            ref={fileRef}
            type="file"
            accept={REFERENCE_TYPES.join(',')}
            multiple
            hidden
            onChange={(e) => {
              attachFiles(Array.from(e.target.files || []));
              e.target.value = '';
            }}
          />
          {/* Reference images, in prompt order: "image 1" is the first */}
          {(hasReferences || notice || dragging) && (
            <Fold open={expanded}>
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 pb-2 mb-2 border-b border-[#e5e2db]/50 dark:border-[#2e2e2a]/50" role="group" aria-label="Reference images">
                <div className="flex gap-2 shrink-0 max-w-full overflow-x-auto">
                  {references.map((ref, i) => (
                    <ReferenceThumb
                      key={ref.key}
                      reference={ref}
                      index={i}
                      onRemove={() => removeReference(ref.key)}
                      onMoveEarlier={i > 0 ? () => onReferencesChange((prev) => moveEarlier(prev, i)) : undefined}
                    />
                  ))}
                </div>
                <span className={`flex-1 min-w-[9rem] text-[11px] ${notice ? 'text-rose-600 dark:text-rose-400 font-medium' : 'text-[#8c7169] dark:text-[#a09088]'}`}>
                  {notice ||
                    (dragging
                      ? 'Drop images to use them as references'
                      : `"Image 1"${references.length > 1 ? `, "image 2"` : ''} in the prompt refer${references.length > 1 ? '' : 's'} to ${
                          references.length > 1 ? 'these, in order' : 'this'
                        }`)}
                </span>
              </div>
            </Fold>
          )}
          {/* Filled from an image (Rerun / Vary): which one, and original or enhanced prompt */}
          {source && (
            <Fold open={expanded}>
            <div className="flex items-center gap-2 pb-2 mb-2 border-b border-[#e5e2db]/50 dark:border-[#2e2e2a]/50 text-[11px] text-[#8c7169] dark:text-[#a09088]">
              <CornerDownRight className="w-3.5 h-3.5 shrink-0" />
              <span className="font-mono whitespace-nowrap" title={`Loaded from the image with seed ${source.seed}`}>
                From #{source.seed}
              </span>
              {source.enhanced && (
                <div className="flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5" role="group" aria-label="Prompt to use">
                  {(['original', 'enhanced'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => switchSource(mode)}
                      aria-pressed={source.mode === mode}
                      title={
                        mode === 'original'
                          ? `The prompt as typed${source.originalEnhance ? ', enhanced again' : ''}`
                          : 'The rewritten prompt, used as is (Enhance off)'
                      }
                      className={`flex items-center gap-1 px-2 py-0.5 rounded-md font-medium transition-all cursor-pointer ${
                        source.mode === mode ? segmentOn : segmentOff
                      }`}
                    >
                      {mode === 'enhanced' && <Sparkles className="w-3 h-3" />}
                      {mode === 'original' ? 'Original' : 'Enhanced'}
                    </button>
                  ))}
                </div>
              )}
              <button
                type="button"
                onClick={() => onSourceChange?.(null)}
                className="ml-auto p-0.5 rounded hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                title="Dismiss (keeps the prompt and settings)"
                aria-label="Dismiss source"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            </Fold>
          )}

          {/* Prompt: one line while folded, grows when open */}
          <div className="flex items-end gap-2">
            <textarea
              ref={inputRef}
              rows={1}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setExpanded(true)}
              onPaste={(e) => {
                const files = Array.from(e.clipboardData.files);
                if (files.some((f) => f.type.startsWith('image/'))) {
                  e.preventDefault();
                  attachFiles(files);
                }
              }}
              aria-label="Prompt"
              placeholder={
                expanded
                  ? "Describe an image you'd like to create... (Press Enter to submit, Shift+Enter for newline)"
                  : 'Describe an image…'
              }
              className={`flex-1 min-w-0 resize-none bg-transparent text-sm sm:text-base text-[#1c1c18] dark:text-[#f3f0e9] placeholder-[#8c7169]/60 dark:placeholder-[#a09088]/60 focus:outline-hidden leading-relaxed ${
                expanded ? 'min-h-[44px]' : 'self-center overflow-hidden'
              }`}
            />

            {/* Folded: what the next send makes; a tap opens the composer */}
            {!expanded && (
              <button
                type="button"
                onClick={expand}
                title={summaryTitle}
                className="self-center shrink-0 flex items-center gap-1 font-mono text-[11px] text-[#8c7169] dark:text-[#a09088] whitespace-nowrap cursor-pointer"
              >
                {summary}
                {hasReferences && (
                  <span className="flex items-center gap-0.5">
                    · <ImageIcon className="w-3 h-3" />
                    {references.length}
                  </span>
                )}
                {main.enhance && <Sparkles className="w-3 h-3 text-amber-600 dark:text-amber-400" />}
                {advanced && <SlidersHorizontal className="w-3 h-3" />}
              </button>
            )}

            {/* Submit Button. Folded with Advanced on, it opens the composer first: nothing hidden is sent. */}
            <button
              onClick={!expanded && advanced ? expand : handleGenerate}
              disabled={!prompt.trim() || isSubmitting || overLimit || referencesPending}
              className={`w-9 h-9 rounded-full bg-[#a53705] text-white flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:hover:opacity-30 transition-all shadow-md shrink-0 cursor-pointer ${
                expanded ? 'mb-1' : ''
              }`}
              title={sendTitle}
              aria-label="Generate"
            >
              <Send className="w-4 h-4 ml-0.5" />
            </button>
          </div>

          <Fold open={expanded}>
          {/* Controls and Settings Row */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-3 mt-2 border-t border-[#e5e2db]/50 dark:border-[#2e2e2a]/50 text-xs text-[#58423a] dark:text-[#a09088]">
            <div className="flex flex-wrap items-center gap-1.5">
              {/* Attach reference images (also: drop anywhere, or paste) */}
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={maxReferences === 0 || references.length >= maxReferences}
                className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320] hover:bg-[#e5e2db] dark:hover:bg-[#2c2c28] text-[#1c1c18] dark:text-[#f3f0e9] font-medium transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                title={
                  maxReferences === 0
                    ? 'No agent accepts reference images'
                    : `Attach reference images (up to ${maxReferences}); you can also drop or paste them`
                }
                aria-label="Attach reference images"
              >
                <ImagePlus className="w-3.5 h-3.5" />
                {hasReferences && <span className="font-mono">{references.length}/{maxReferences}</span>}
              </button>

              {/* Aspect Ratio Picker Chip & Popover */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setIsAspectOpen(!isAspectOpen)}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320] hover:bg-[#e5e2db] dark:hover:bg-[#2c2c28] text-[#1c1c18] dark:text-[#f3f0e9] font-medium transition-colors cursor-pointer"
                >
                  <div className="w-4 h-4 flex items-center justify-center">
                    <div
                      className="border border-current rounded-[2px] opacity-80"
                      style={{
                        width: `${getAspectIconDims(ASPECT_RATIOS[preferences.aspect].ratioWidth, ASPECT_RATIOS[preferences.aspect].ratioHeight, 13).width}px`,
                        height: `${getAspectIconDims(ASPECT_RATIOS[preferences.aspect].ratioWidth, ASPECT_RATIOS[preferences.aspect].ratioHeight, 13).height}px`,
                      }}
                    />
                  </div>
                  <span>{preferences.aspect}</span>
                  <ChevronDown className="w-3 h-3 opacity-60" />
                </button>

                {isAspectOpen && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setIsAspectOpen(false)} />
                    <div className="absolute left-0 bottom-full mb-2 z-50 p-2 rounded-xl bg-[#fcf9f2] dark:bg-[#1f1f1c] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl grid grid-cols-4 gap-1.5 min-w-[240px]">
                      {(Object.keys(ASPECT_RATIOS) as AspectRatio[]).map((r) => {
                        const item = ASPECT_RATIOS[r];
                        const isSelected = r === preferences.aspect;
                        const icon = getAspectIconDims(item.ratioWidth, item.ratioHeight, 15);
                        return (
                          <button
                            key={r}
                            type="button"
                            onClick={() => {
                              onUpdatePreferences({ aspect: r });
                              setIsAspectOpen(false);
                            }}
                            className={`flex flex-col items-center justify-center h-14 p-1.5 rounded-lg text-center transition-all cursor-pointer ${
                              isSelected
                                ? 'bg-[#a53705] text-white shadow-2xs font-semibold'
                                : 'hover:bg-[#f1eee7] dark:hover:bg-[#292925] text-[#1c1c18] dark:text-[#f3f0e9]'
                            }`}
                          >
                            <div className="w-5 h-5 flex items-center justify-center mb-1">
                              <div
                                className={`border rounded-[2px] transition-colors ${
                                  isSelected ? 'border-white' : 'border-current opacity-70'
                                }`}
                                style={{
                                  width: `${icon.width}px`,
                                  height: `${icon.height}px`,
                                }}
                              />
                            </div>
                            <span className="text-[10px] font-mono leading-tight">{r}</span>
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>

              {/* Size Multiplier Dropdown (Full, Half, Quarter) */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setIsScaleOpen(!isScaleOpen)}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320] hover:bg-[#e5e2db] dark:hover:bg-[#2c2c28] text-[#1c1c18] dark:text-[#f3f0e9] font-medium transition-colors cursor-pointer"
                  title="Resolution multiplier: Full 2K, Half, or Quarter for fast iteration"
                >
                  <Gauge className="w-3 h-3 text-[#a53705]" />
                  <span>{SIZE_MULTIPLIERS[preferences.sizeMultiplier || 'full'].label}</span>
                  <span className="font-mono text-[10px] opacity-60">
                    ({SIZE_MULTIPLIERS[preferences.sizeMultiplier || 'full'].badge})
                  </span>
                  <ChevronDown className="w-3 h-3 opacity-60" />
                </button>

                {isScaleOpen && (
                  <>
                    <div className="fixed inset-0 z-40" onClick={() => setIsScaleOpen(false)} />
                    <div className="absolute left-0 bottom-full mb-2 z-50 p-1.5 rounded-xl bg-[#fcf9f2] dark:bg-[#1f1f1c] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl flex flex-col gap-1 min-w-[230px]">
                      <div className="px-2.5 py-1 text-[10px] uppercase font-semibold text-[#8c7169] dark:text-[#a09088] tracking-wider border-b border-[#e5e2db]/50 dark:border-[#2e2e2a]/50">
                        Size Multiplier
                      </div>
                      {(Object.keys(SIZE_MULTIPLIERS) as SizeMultiplier[]).map((key) => {
                        const item = SIZE_MULTIPLIERS[key];
                        const isSelected = (preferences.sizeMultiplier || 'full') === key;
                        const dims = calculateDimensions(preferences.aspect, key);
                        const px = dims.width * dims.height;
                        const everMaxPx = capabilities?.max_px_ever ?? 0;
                        const nowMaxPx = capabilities?.max_px_now ?? 0;
                        const isNeverServable =
                          capabilities != null && everMaxPx > 0 && px > everMaxPx;
                        const isTemporarilyUnavailable =
                          capabilities != null && !isNeverServable && everMaxPx > 0 && px > nowMaxPx;

                        return (
                          <button
                            key={key}
                            type="button"
                            disabled={isNeverServable}
                            title={
                              isNeverServable
                                ? `No registered executor supports ${dims.width}×${dims.height} (max ${everMaxPx} px)`
                                : isTemporarilyUnavailable
                                ? `Capable executor currently offline, lent, or disabled — job will wait in queue`
                                : undefined
                            }
                            onClick={() => {
                              if (isNeverServable) return;
                              onUpdatePreferences({ sizeMultiplier: key });
                              setIsScaleOpen(false);
                            }}
                            className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-left transition-all ${
                              isNeverServable
                                ? 'opacity-40 cursor-not-allowed'
                                : isSelected
                                ? 'bg-[#a53705] text-white shadow-2xs font-semibold cursor-pointer'
                                : 'hover:bg-[#f1eee7] dark:hover:bg-[#292925] text-[#1c1c18] dark:text-[#f3f0e9] cursor-pointer'
                            }`}
                          >
                            <div className="flex flex-col">
                              <div className="flex items-center gap-1.5">
                                <span className="text-xs">{item.label}</span>
                                <span
                                  className={`font-mono text-[10px] px-1 py-0.2 rounded ${
                                    isSelected
                                      ? 'bg-black/20 text-white'
                                      : 'bg-[#e5e2db]/80 dark:bg-[#2e2e2a] text-[#8c7169] dark:text-[#a09088]'
                                  }`}
                                >
                                  {item.badge}
                                </span>
                                {isTemporarilyUnavailable && (
                                  <span className="text-[9px] font-mono px-1 rounded bg-amber-500/20 text-amber-700 dark:text-amber-300">
                                    queued
                                  </span>
                                )}
                              </div>
                              <span
                                className={`text-[10px] ${
                                  isSelected ? 'text-white/80' : 'text-[#8c7169] dark:text-[#a09088]'
                                }`}
                              >
                                {item.description}
                              </span>
                            </div>
                            <span
                              className={`font-mono text-[10px] ml-2 ${
                                isSelected ? 'text-white' : 'text-[#8c7169] dark:text-[#a09088]'
                              }`}
                            >
                              {dims.width}×{dims.height}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>

              {/* Quality presets; custom steps live in Advanced */}
              <div className="flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5" role="group" aria-label="Steps">
                {(['draft', 'standard', 'high'] as const).map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => onUpdatePreferences({ quality: q, steps: QUALITY_STEPS[q] })}
                    aria-pressed={shownQuality === q}
                    title={`${QUALITY_STEPS[q]} denoising steps`}
                    className={`px-2 py-0.5 rounded-md font-medium transition-all cursor-pointer whitespace-nowrap ${
                      shownQuality === q ? segmentOn : segmentOff
                    }`}
                  >
                    {QUALITY_LABELS[q]} <span className="opacity-60">{QUALITY_STEPS[q]}</span>
                  </button>
                ))}
              </div>

              {/* Image count (1-4), at every width */}
              <div className="flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5" role="group" aria-label="Images per row">
                {[1, 2, 3, 4].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => onUpdatePreferences({ count: n })}
                    aria-pressed={preferences.count === n}
                    title={n === 1 ? '1 image' : `${n} images`}
                    className={`w-6 h-5 rounded-md font-mono text-[11px] font-medium transition-all cursor-pointer ${
                      preferences.count === n
                        ? 'bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] shadow-2xs font-semibold'
                        : 'text-[#8c7169] dark:text-[#a09088]'
                    }`}
                  >
                    {n}x
                  </button>
                ))}
              </div>
            </div>

            {/* Right side: Advanced toggle and Prompt Enhancer Toggle */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onUpdatePreferences({ advancedOpen: !advanced })}
                aria-expanded={advanced}
                aria-controls="composer-advanced"
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition-all cursor-pointer ${
                  advanced
                    ? 'bg-[#1c1c18] text-[#fcf9f2] dark:bg-[#f3f0e9] dark:text-[#1c1c18]'
                    : 'bg-[#f1eee7] dark:bg-[#232320] text-[#8c7169] dark:text-[#a09088] hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]'
                }`}
                title="Advanced: custom steps, fixed seed, tiling, and rows of different settings"
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>Advanced</span>
                <ChevronDown className={`w-3 h-3 transition-transform ${advanced ? 'rotate-180' : ''}`} />
              </button>
              <button
                type="button"
                onClick={() => onUpdatePreferences({ enhance: !preferences.enhance })}
                aria-pressed={main.enhance}
                disabled={enhanceBlocked}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all border cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                  main.enhance
                    ? 'bg-amber-500/10 text-amber-900 dark:text-amber-300 border-amber-500/30'
                    : 'bg-[#f1eee7] dark:bg-[#232320] text-[#8c7169] dark:text-[#a09088] border-transparent hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]'
                }`}
                title={
                  enhanceBlocked
                    ? enhanceOffTitle(enhanceMostRefs)
                    : hasReferences
                      ? 'Enhance prompt before generating (Qwen-Image-2.1-PE-I2I reads the images and rewrites it on a free GPU)'
                      : 'Enhance prompt before generating (Qwen-Image-2.1-PE rewrites it on a free GPU)'
                }
              >
                <Sparkles
                  className={`w-3.5 h-3.5 ${
                    main.enhance ? 'text-amber-600 dark:text-amber-400' : ''
                  }`}
                />
                <span>Enhance</span>
                {main.enhance && (
                  <Check className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                )}
              </button>
            </div>
          </div>

          {/* Advanced: custom steps, fixed seed, tiling, rows. Applies only while open. */}
          {advanced && (
            <div
              id="composer-advanced"
              className="pt-3 mt-3 border-t border-[#e5e2db]/50 dark:border-[#2e2e2a]/50 space-y-2.5 text-xs text-[#58423a] dark:text-[#a09088] max-h-[40vh] overflow-y-auto"
            >
              <div className="flex flex-wrap items-center gap-1.5">
                <StepsField
                  label="Custom steps"
                  value={main.steps}
                  onChange={(n) => onUpdatePreferences({ steps: n, quality: qualityForSteps(n) })}
                />

                {/* Seed: random per send, or fixed */}
                <div className="flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5" role="group" aria-label="Seed">
                  {([false, true] as const).map((locked) => (
                    <button
                      key={String(locked)}
                      type="button"
                      onClick={() => onUpdatePreferences({ isSeedLocked: locked })}
                      aria-pressed={preferences.isSeedLocked === locked}
                      className={`px-2 py-0.5 rounded-md font-medium transition-all cursor-pointer ${
                        preferences.isSeedLocked === locked ? segmentOn : segmentOff
                      }`}
                      title={locked ? 'The same seed every send' : 'A new random seed every send'}
                    >
                      {locked ? 'Fixed seed' : 'Random seed'}
                    </button>
                  ))}
                  {preferences.isSeedLocked && (
                    <>
                      <input
                        type="number"
                        value={preferences.lockedSeed}
                        onChange={(e) => onUpdatePreferences({ lockedSeed: parseInt(e.target.value) || 0 })}
                        aria-label="Seed"
                        className="w-20 px-1.5 font-mono bg-transparent text-[#1c1c18] dark:text-[#f3f0e9] outline-none"
                      />
                      <button
                        type="button"
                        onClick={rollSeed}
                        className="px-1 hover:text-[#a53705] transition-colors cursor-pointer"
                        title="Roll a new seed"
                        aria-label="Roll a new seed"
                      >
                        <Dices className="w-3.5 h-3.5" />
                      </button>
                    </>
                  )}
                </div>

                <ToggleChip
                  on={preferences.tiling}
                  onClick={() => onUpdatePreferences({ tiling: !preferences.tiling })}
                  label="Tiling"
                  title={TILING_TITLE}
                  icon={<Grid2x2 className="w-3 h-3" />}
                />

                {/* The rewriter's token limit, for rows with Enhance on */}
                {anyEnhance && (
                  <NumberField
                    value={preferences.enhanceMaxTokens || enhanceTokensDefault(references.length)}
                    onChange={(n) =>
                      onUpdatePreferences({ enhanceMaxTokens: n === enhanceTokensDefault(references.length) ? 0 : n })
                    }
                    label="Enhancer token limit"
                    title={`Most tokens the prompt enhancer may write, reasoning included (${MIN_ENHANCE_TOKENS}–${MAX_ENHANCE_TOKENS}, default ${ENHANCE_TOKENS_DEFAULT} plus ${ENHANCE_TOKENS_PER_REFERENCE} per reference image). Raise it for long, detailed prompts such as multi-part infographics; ~30 tokens a second. Reasoning that runs out of room continues with the limit doubled, up to ${MAX_ENHANCE_TOKENS}.`}
                    min={MIN_ENHANCE_TOKENS}
                    max={MAX_ENHANCE_TOKENS}
                    unit="enhancer tokens"
                    width="w-12"
                  />
                )}
              </div>

              {/* Rows: each a full set of settings, sent with the same prompt and seed */}
              <div className="space-y-1.5" role="group" aria-label="Rows">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="w-4 text-center font-mono opacity-60">1</span>
                  <span className="truncate" title="Row 1 is the main settings above">
                    {rowSummary(main)} <span className="opacity-60">· main settings</span>
                  </span>
                </div>
                {preferences.extraRows.map((row, i) => (
                  <div key={i} className="flex flex-wrap items-center gap-1.5" data-row={i + 2}>
                    <span className="w-4 text-center font-mono opacity-60">{i + 2}</span>
                    <select
                      value={row.aspect}
                      onChange={(e) => updateRow(i, { aspect: e.target.value as AspectRatio })}
                      aria-label={`Row ${i + 2} aspect ratio`}
                      className={selectCls}
                    >
                      {(Object.keys(ASPECT_RATIOS) as AspectRatio[]).map((r) => (
                        <option key={r} value={r}>
                          {r}
                        </option>
                      ))}
                    </select>
                    <select
                      value={row.sizeMultiplier}
                      onChange={(e) => updateRow(i, { sizeMultiplier: e.target.value as SizeMultiplier })}
                      aria-label={`Row ${i + 2} size`}
                      className={selectCls}
                    >
                      {(Object.keys(SIZE_MULTIPLIERS) as SizeMultiplier[]).map((k) => (
                        <option key={k} value={k}>
                          {SIZE_MULTIPLIERS[k].label}
                        </option>
                      ))}
                    </select>
                    <StepsField label={`Row ${i + 2} steps`} value={row.steps} onChange={(n) => updateRow(i, { steps: n })} />
                    <ToggleChip
                      on={row.enhance && !enhanceBlocked}
                      onClick={() => updateRow(i, { enhance: !row.enhance })}
                      label="Enhance"
                      title={enhanceBlocked ? enhanceOffTitle(enhanceMostRefs) : 'Rewrite the prompt with Qwen-Image-2.1-PE first'}
                      icon={<Sparkles className="w-3 h-3" />}
                      disabled={enhanceBlocked}
                    />
                    <ToggleChip
                      on={row.tiling}
                      onClick={() => updateRow(i, { tiling: !row.tiling })}
                      label="Tiling"
                      title={TILING_TITLE}
                      icon={<Grid2x2 className="w-3 h-3" />}
                    />
                    <button
                      type="button"
                      onClick={() => onUpdatePreferences({ extraRows: preferences.extraRows.filter((_, j) => j !== i) })}
                      className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                      title="Remove this row"
                      aria-label={`Remove row ${i + 2}`}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={addRow}
                    disabled={!canAddRow}
                    className="flex items-center gap-1 px-2 py-1 rounded-lg border border-dashed border-[#d6cfc4] dark:border-[#3a3a35] font-medium hover:text-[#1c1c18] dark:hover:text-[#f3f0e9] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    title="Add a row: a copy of the last one, to change one setting"
                  >
                    <Plus className="w-3 h-3" />
                    Add row
                  </button>
                  <span className={overLimit ? 'text-rose-600 dark:text-rose-400 font-medium' : 'opacity-70'}>
                    {totalImages} {totalImages === 1 ? 'image' : 'images'} per send
                    {count > 1 && ` (${count} per row)`}
                    {overLimit && ` · at most ${MAX_IMAGES_PER_SEND}: remove a row or lower the count`}
                    {!overLimit && (count > 1 ? ', same seeds in every row' : ', same seed in every row')}
                  </span>
                </div>
              </div>
            </div>
          )}
          </Fold>
        </div>
      </div>
    </div>
  );
};
