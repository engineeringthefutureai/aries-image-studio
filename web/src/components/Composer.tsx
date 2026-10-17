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
  ASPECT_RATIOS,
  SIZE_MULTIPLIERS,
  QUALITY_STEPS,
  MIN_STEPS,
  MAX_STEPS,
  calculateDimensions,
  clampSteps,
  qualityForSteps,
  MAX_IMAGES_PER_SEND,
} from '../types';
import { buildJobs, sendSeeds, imagesPerSend, mainRow } from '../utils/composer';
import { Sparkles, Dices, Send, ChevronDown, Check, Gauge, CornerDownRight, X, SlidersHorizontal, Plus, Grid2x2 } from 'lucide-react';

interface ComposerProps {
  preferences: UserPreferences;
  onUpdatePreferences: (updates: Partial<UserPreferences>) => void;
  /** One send: every job of the turn (rows × count), see utils/composer.ts. */
  onSubmit: (prompt: string, jobs: JobParams[]) => void;
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

/** A steps number field (1–100) that can be typed in freely and settles on blur or Enter. */
const StepsField: React.FC<{ value: number; onChange: (steps: number) => void; label: string }> = ({
  value,
  onChange,
  label,
}) => {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <label className={chip} title={`Denoising steps (${MIN_STEPS}–${MAX_STEPS})`}>
      <input
        type="number"
        inputMode="numeric"
        min={MIN_STEPS}
        max={MAX_STEPS}
        step={1}
        value={draft ?? String(value)}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value !== '' && Number.isInteger(n) && n >= MIN_STEPS && n <= MAX_STEPS) onChange(n);
        }}
        onBlur={() => {
          // Out of range snaps to the nearest limit; empty or not a number keeps the last value.
          if (draft !== null && draft.trim() !== '' && Number.isFinite(Number(draft))) onChange(clampSteps(Number(draft)));
          setDraft(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        }}
        aria-label={label}
        className="w-8 bg-transparent text-right font-semibold tabular-nums text-[#1c1c18] dark:text-[#f3f0e9] outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
      <span className="opacity-60">steps</span>
    </label>
  );
};

/** An on/off chip (Enhance, Tiling) inside the advanced panel. */
const ToggleChip: React.FC<{ on: boolean; onClick: () => void; label: string; title: string; icon?: React.ReactNode }> = ({
  on,
  onClick,
  label,
  title,
  icon,
}) => (
  <button
    type="button"
    onClick={onClick}
    aria-pressed={on}
    title={title}
    className={`flex items-center gap-1 px-2 py-1 rounded-lg border font-medium transition-all cursor-pointer ${
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
}) => {
  const [prompt, setPrompt] = useState('');
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
    if (!prompt.trim() || isSubmitting || overLimit) return;
    onSubmit(prompt.trim(), buildJobs(preferences, sendSeeds(preferences)));
    // Advanced is for one send: close it and drop its settings, so nothing hidden carries
    // over (custom steps back to Standard, random seed, no tiling, one row).
    if (preferences.advancedOpen) {
      onUpdatePreferences({
        advancedOpen: false,
        extraRows: [],
        tiling: false,
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
  const main = mainRow(preferences);
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
  const summaryTitle = `${totalImages} ${totalImages === 1 ? 'image' : 'images'}: ${rowSummary(main)}${
    advanced ? ', Advanced on' : ''
  }`;

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
          className={`relative bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl transition-[border-radius,padding] duration-200 ${
            expanded ? 'rounded-2xl p-3 sm:p-4' : 'rounded-3xl py-1.5 pl-4 pr-1.5'
          }`}
        >
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
                {main.enhance && <Sparkles className="w-3 h-3 text-amber-600 dark:text-amber-400" />}
                {advanced && <SlidersHorizontal className="w-3 h-3" />}
              </button>
            )}

            {/* Submit Button. Folded with Advanced on, it opens the composer first: nothing hidden is sent. */}
            <button
              onClick={!expanded && advanced ? expand : handleGenerate}
              disabled={!prompt.trim() || isSubmitting || overLimit}
              className={`w-9 h-9 rounded-full bg-[#a53705] text-white flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:hover:opacity-30 transition-all shadow-md shrink-0 cursor-pointer ${
                expanded ? 'mb-1' : ''
              }`}
              title="Generate (Enter)"
              aria-label="Generate"
            >
              <Send className="w-4 h-4 ml-0.5" />
            </button>
          </div>

          <Fold open={expanded}>
          {/* Controls and Settings Row */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-3 mt-2 border-t border-[#e5e2db]/50 dark:border-[#2e2e2a]/50 text-xs text-[#58423a] dark:text-[#a09088]">
            <div className="flex flex-wrap items-center gap-1.5">
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
                aria-pressed={preferences.enhance}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all border cursor-pointer ${
                  preferences.enhance
                    ? 'bg-amber-500/10 text-amber-900 dark:text-amber-300 border-amber-500/30'
                    : 'bg-[#f1eee7] dark:bg-[#232320] text-[#8c7169] dark:text-[#a09088] border-transparent hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]'
                }`}
                title="Enhance prompt before generating (Qwen-Image-2.1-PE rewrites it on a free GPU)"
              >
                <Sparkles
                  className={`w-3.5 h-3.5 ${
                    preferences.enhance ? 'text-amber-600 dark:text-amber-400' : ''
                  }`}
                />
                <span>Enhance</span>
                {preferences.enhance && (
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
                      on={row.enhance}
                      onClick={() => updateRow(i, { enhance: !row.enhance })}
                      label="Enhance"
                      title="Rewrite the prompt with Qwen-Image-2.1-PE first"
                      icon={<Sparkles className="w-3 h-3" />}
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
