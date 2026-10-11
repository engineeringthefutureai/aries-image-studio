import React, { useState, useRef, useEffect } from 'react';
import {
  type AspectRatio,
  type JobParams,
  type UserPreferences,
  type SizeMultiplier,
  ASPECT_RATIOS,
  SIZE_MULTIPLIERS,
  calculateDimensions,
} from '../types';
import { Sparkles, Dices, Send, Lock, Unlock, ChevronDown, Check, Gauge } from 'lucide-react';

interface ComposerProps {
  preferences: UserPreferences;
  onUpdatePreferences: (updates: Partial<UserPreferences>) => void;
  onSubmit: (prompt: string, params: JobParams) => void;
  isSubmitting: boolean;
  workerState?: string;
  prefilledPrompt?: string | null;
  prefilledAspect?: AspectRatio | null;
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

export const Composer: React.FC<ComposerProps> = ({
  preferences,
  onUpdatePreferences,
  onSubmit,
  isSubmitting,
  prefilledPrompt,
  prefilledAspect,
  onClearPrefill,
  textareaRef,
}) => {
  const [prompt, setPrompt] = useState('');
  const [isAspectOpen, setIsAspectOpen] = useState(false);
  const [isScaleOpen, setIsScaleOpen] = useState(false);
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
      inputRef.current?.focus();
    }
  }, [prefilledPrompt, prefilledAspect, onClearPrefill, onUpdatePreferences, inputRef]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 180)}px`;
    }
  }, [prompt, inputRef]);

  const handleGenerate = () => {
    if (!prompt.trim() || isSubmitting) return;

    const { width, height } = calculateDimensions(
      preferences.aspect,
      preferences.sizeMultiplier || 'full'
    );
    const seed = preferences.isSeedLocked
      ? preferences.lockedSeed
      : Math.floor(Math.random() * 1000000);
    const steps = preferences.quality === 'draft' ? 20 : 40;

    const params: JobParams = {
      width,
      height,
      aspect: preferences.aspect,
      sizeMultiplier: preferences.sizeMultiplier || 'full',
      steps,
      seed,
      quality: preferences.quality,
      enhance: preferences.enhance,
    };

    onSubmit(prompt.trim(), params);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const rollSeed = () => {
    const nextSeed = Math.floor(Math.random() * 1000000);
    onUpdatePreferences({ lockedSeed: nextSeed });
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 pointer-events-none pb-4 px-4">
      <div className="w-full max-w-4xl mx-auto pointer-events-auto">
        <div className="relative rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl p-3 sm:p-4 transition-all">
          {/* Multiline Prompt Input Area */}
          <div className="relative">
            <textarea
              ref={inputRef}
              rows={1}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe an image you'd like to create... (Press Enter to submit, Shift+Enter for newline)"
              className="w-full resize-none bg-transparent text-sm sm:text-base text-[#1c1c18] dark:text-[#f3f0e9] placeholder-[#8c7169]/60 dark:placeholder-[#a09088]/60 focus:outline-hidden leading-relaxed pr-12 min-h-[44px]"
            />

            {/* Submit Button */}
            <button
              onClick={handleGenerate}
              disabled={!prompt.trim() || isSubmitting}
              className="absolute right-0 bottom-1 w-9 h-9 rounded-full bg-[#a53705] text-white flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:hover:opacity-30 transition-all shadow-md shrink-0 cursor-pointer"
              title="Generate (Enter)"
            >
              <Send className="w-4 h-4 ml-0.5" />
            </button>
          </div>

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

                        return (
                          <button
                            key={key}
                            type="button"
                            onClick={() => {
                              onUpdatePreferences({ sizeMultiplier: key });
                              setIsScaleOpen(false);
                            }}
                            className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-left transition-all cursor-pointer ${
                              isSelected
                                ? 'bg-[#a53705] text-white shadow-2xs font-semibold'
                                : 'hover:bg-[#f1eee7] dark:hover:bg-[#292925] text-[#1c1c18] dark:text-[#f3f0e9]'
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

              {/* Quality Preset */}
              <div className="flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5">
                <button
                  type="button"
                  onClick={() => onUpdatePreferences({ quality: 'standard', steps: 40 })}
                  className={`px-2 py-0.5 rounded-md font-medium transition-all cursor-pointer ${
                    preferences.quality === 'standard'
                      ? 'bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] shadow-2xs font-semibold'
                      : 'text-[#8c7169] dark:text-[#a09088]'
                  }`}
                >
                  Standard (40)
                </button>
                <button
                  type="button"
                  onClick={() => onUpdatePreferences({ quality: 'draft', steps: 20 })}
                  className={`px-2 py-0.5 rounded-md font-medium transition-all cursor-pointer ${
                    preferences.quality === 'draft'
                      ? 'bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] shadow-2xs font-semibold'
                      : 'text-[#8c7169] dark:text-[#a09088]'
                  }`}
                >
                  Draft (20)
                </button>
              </div>

              {/* Seed Chip */}
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320]">
                <button
                  type="button"
                  onClick={() =>
                    onUpdatePreferences({ isSeedLocked: !preferences.isSeedLocked })
                  }
                  className="hover:text-[#a53705] transition-colors cursor-pointer"
                  title={
                    preferences.isSeedLocked
                      ? 'Seed is locked. Click to randomize.'
                      : 'Seed is random. Click to lock.'
                  }
                >
                  {preferences.isSeedLocked ? (
                    <Lock className="w-3 h-3 text-[#a53705]" />
                  ) : (
                    <Unlock className="w-3 h-3 opacity-60" />
                  )}
                </button>

                {preferences.isSeedLocked ? (
                  <input
                    type="number"
                    value={preferences.lockedSeed}
                    onChange={(e) =>
                      onUpdatePreferences({ lockedSeed: parseInt(e.target.value) || 0 })
                    }
                    className="w-16 font-mono text-xs bg-transparent text-[#1c1c18] dark:text-[#f3f0e9] focus:outline-hidden"
                  />
                ) : (
                  <span className="font-mono text-xs">Random</span>
                )}

                <button
                  type="button"
                  onClick={rollSeed}
                  className="hover:text-[#a53705] transition-colors p-0.5 cursor-pointer"
                  title="Roll a new random seed"
                >
                  <Dices className="w-3 h-3" />
                </button>
              </div>

              {/* Image Count Selector (1-4) */}
              <div className="hidden sm:flex items-center rounded-lg bg-[#f1eee7] dark:bg-[#232320] p-0.5">
                {[1, 2, 3, 4].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => onUpdatePreferences({ count: n })}
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

            {/* Right side: Prompt Enhancer Toggle */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onUpdatePreferences({ enhance: !preferences.enhance })}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all border cursor-pointer ${
                  preferences.enhance
                    ? 'bg-amber-500/10 text-amber-900 dark:text-amber-300 border-amber-500/30'
                    : 'bg-[#f1eee7] dark:bg-[#232320] text-[#8c7169] dark:text-[#a09088] border-transparent hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]'
                }`}
                title="Enhance prompt before generating (runs Qwen-Image-2.1-PE on GPU 1)"
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
        </div>
      </div>
    </div>
  );
};
