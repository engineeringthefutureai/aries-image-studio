import React, { useState } from 'react';
import { type Generation, type AspectRatio, ASPECT_RATIOS } from '../types';
import { sanitizeEnhancedPrompt } from '../utils/prompt';
import { Logo } from './Logo';
import { formatDuration, generationSeconds, queueSeconds } from '../utils/timings';
import {
  Heart,
  Dices,
  Maximize2,
  Download,
  RotateCcw,
  XCircle,
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  Brain,
  Sparkles,
  AlertCircle,
  Loader2,
  Trash2,
  Sliders,
} from 'lucide-react';

/** Enhancer token limit, for progress from workers that do not report `max_new_tokens`. */
const DEFAULT_ENHANCE_TOKEN_LIMIT = 2048;
/** Most enhancer runs finish around this share of the token limit: the bar's finish line. */
const ENHANCE_EXPECTED_SHARE = 0.85;

interface TimelineProps {
  generations: Generation[];
  onOpenLightbox: (generation: Generation) => void;
  onToggleFavorite: (id: string) => void;
  onVary: (generation: Generation) => void;
  onRetry: (generation: Generation) => void;
  onCancel: (id: string) => void;
  onSelectPrompt: (prompt: string, aspect?: AspectRatio) => void;
  onRerunEnhanced?: (generation: Generation) => void;
  onReuseEnhanced?: (generation: Generation) => void;
  onDelete?: (id: string) => void;
  onDeleteTurn?: (batchId: string) => void;
}

export const Timeline: React.FC<TimelineProps> = ({
  generations,
  onOpenLightbox,
  onToggleFavorite,
  onVary,
  onRetry,
  onCancel,
  onSelectPrompt,
  onRerunEnhanced,
  onReuseEnhanced,
  onDelete,
  onDeleteTurn,
}) => {
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [selectedGenIdByTurn, setSelectedGenIdByTurn] = useState<Record<string, string>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Group generations by batchId - reverse chronological (newest on top)
  const turns = React.useMemo(() => {
    const map = new Map<string, Generation[]>();
    for (const gen of generations) {
      const arr = map.get(gen.batchId) || [];
      arr.push(gen);
      map.set(gen.batchId, arr);
    }
    // Return sorted newest to oldest for timeline
    return Array.from(map.values()).sort(
      (a, b) => (b[0]?.createdAt || 0) - (a[0]?.createdAt || 0)
    );
  }, [generations]);

  const handleCopyPrompt = (prompt: string, id: string) => {
    navigator.clipboard.writeText(prompt);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  if (turns.length === 0) {
    const suggestions = [
      {
        prompt: 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
        aspect: '16:9' as AspectRatio,
      },
      {
        prompt: 'A solitary granite lighthouse on a steep cliff during a violent sea tempest, colossal crashing waves',
        aspect: '9:16' as AspectRatio,
      },
      {
        prompt: 'Panoramic mountain lake at dawn, soft golden mist over the water, a single wooden canoe',
        aspect: '16:9' as AspectRatio,
      },
      {
        prompt: 'Portrait of an elderly seafaring fisherman, Rembrandt lighting, weathered textured skin, 85mm photo',
        aspect: '3:4' as AspectRatio,
      },
    ];

    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-2xl mx-auto text-center space-y-6">
        <Logo className="w-24 h-24 drop-shadow-md" title="Aries Image Studio" />
        <div>
          <h2 className="font-['Plus_Jakarta_Sans'] font-bold text-2xl text-[#1c1c18] dark:text-[#f3f0e9]">
            Aries Image Studio
          </h2>
          <p className="text-sm text-[#58423a] dark:text-[#a09088] mt-1.5 max-w-md">
            Type a prompt below to create images with Qwen-Image 2.1. Or start with a sample:
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => onSelectPrompt(s.prompt, s.aspect)}
              className="p-3.5 rounded-xl bg-[#f6f3ec] dark:bg-[#1f1f1c] hover:bg-[#ebe8e1] dark:hover:bg-[#272723] border border-[#e5e2db]/60 dark:border-[#2e2e2a] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[11px] font-mono text-[#8c7169] dark:text-[#a09088] mb-1">
                <span>{s.aspect}</span>
                <span className="text-[#a53705] opacity-0 group-hover:opacity-100 transition-opacity">
                  Use prompt →
                </span>
              </div>
              <p className="text-xs text-[#1c1c18] dark:text-[#f3f0e9] font-medium line-clamp-2 leading-relaxed">
                {s.prompt}
              </p>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 w-full max-w-5xl mx-auto px-4 py-6 space-y-6">
      {turns.map((turn) => {
        const primary = turn[0];
        if (!primary) return null;

        // Current selected generation in this turn (defaults to first finished with enhanced prompt, or first finished, or primary)
        const selectedGen =
          turn.find((g) => g.id === selectedGenIdByTurn[primary.batchId]) ||
          turn.find((g) => g.status === 'done' && g.enhancedPrompt) ||
          turn.find((g) => g.status === 'done') ||
          primary;

        const selectedIdx = turn.findIndex((g) => g.id === selectedGen.id);
        const cleanEnhanced = sanitizeEnhancedPrompt(selectedGen.enhancedPrompt);
        const isThinkingOpen = !!expandedThinking[selectedGen.id];
        const generationTps = selectedGen.timings?.generation_tps || selectedGen.enhanceStats?.generation_tps || 0;
        const genSeconds = generationSeconds(selectedGen.timings);
        const waitSeconds = queueSeconds(selectedGen.timings);
        const timeTitle =
          genSeconds != null
            ? `Generated in ${formatDuration(genSeconds)}` +
              (waitSeconds != null ? ` · waited ${formatDuration(waitSeconds)} for a GPU` : '')
            : undefined;

        return (
          <article
            key={primary.batchId}
            className="flex flex-col sm:flex-row gap-4 p-4 sm:p-5 rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2a2a26] shadow-xs hover:shadow-md transition-shadow"
          >
            {/* 1. Left Rail: Vertical Badges Column for Selected Image */}
            <div className="hidden sm:flex flex-col items-center justify-between gap-2 w-20 sm:w-24 shrink-0 border-r border-[#e5e2db]/60 dark:border-[#2a2a26] pr-3 text-[11px] font-mono text-[#58423a] dark:text-[#a09088]">
              <div className="flex flex-col items-center gap-1.5 w-full">
                {/* Aspect ratio */}
                <span className="w-full px-1.5 py-1 rounded-lg bg-[#f1eee7] dark:bg-[#232320] text-center font-bold text-[#1c1c18] dark:text-[#f3f0e9]">
                  {selectedGen.params.aspect}
                </span>

                {/* Resolution */}
                <span className="w-full px-1.5 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320] text-center text-[10px]">
                  {selectedGen.params.width}×{selectedGen.params.height}
                </span>

                {/* Size Multiplier */}
                {selectedGen.params.sizeMultiplier && selectedGen.params.sizeMultiplier !== 'full' && (
                  <span className="w-full px-1 py-0.5 rounded-md bg-amber-500/10 text-amber-800 dark:text-amber-300 font-semibold text-[10px] text-center">
                    {selectedGen.params.sizeMultiplier === 'half' ? '1/2' : '1/4'} size
                  </span>
                )}

                {/* Seed */}
                <span
                  className="w-full px-1.5 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320] text-center text-[10px] truncate"
                  title={`Seed: ${selectedGen.params.seed}`}
                >
                  #{selectedGen.params.seed}
                </span>

                {/* Steps */}
                <span className="w-full px-1.5 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320] text-center text-[10px]">
                  {selectedGen.params.steps} st
                </span>

                {/* Timings */}
                {genSeconds != null && (
                  <span
                    className="w-full px-1.5 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320] text-center text-[10px] font-medium"
                    title={timeTitle}
                  >
                    {formatDuration(genSeconds)}
                  </span>
                )}

                {/* Precision */}
                <span className="w-full px-1 py-0.5 rounded-md bg-amber-500/10 text-amber-800 dark:text-amber-300 text-[10px] text-center font-medium">
                  bf16
                </span>
              </div>

              {/* Timestamp */}
              <span className="text-[10px] text-center opacity-60">
                {new Date(selectedGen.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>

            {/* 2. Center: Bounded-Height Image(s) Visual Tile */}
            <div className="flex-initial min-w-0 max-w-full sm:max-w-[55%] md:max-w-[60%] flex items-center justify-start self-center sm:self-start">
              <div className={`flex gap-3.5 py-2.5 px-1.5 ${turn.length > 1 ? 'overflow-x-auto max-w-full scrollbar-thin' : ''}`}>
                {turn.map((gen) => (
                  <TileCard
                    key={gen.id}
                    gen={gen}
                    isSelected={gen.id === selectedGen.id}
                    onSelect={() =>
                      setSelectedGenIdByTurn((prev) => ({
                        ...prev,
                        [primary.batchId]: gen.id,
                      }))
                    }
                    onOpenLightbox={() => onOpenLightbox(gen)}
                    onToggleFavorite={() => onToggleFavorite(gen.id)}
                    onVary={() => onVary(gen)}
                    onRetry={() => onRetry(gen)}
                    onCancel={() => onCancel(gen.id)}
                    onDelete={onDelete ? () => onDelete(gen.id) : undefined}
                  />
                ))}
              </div>
            </div>

            {/* 3. Right: Content Details (Prompt, Enhanced Prompt, Reasoning & Toolbar) */}
            <div className="flex-1 min-w-0 sm:min-w-[280px] flex flex-col justify-between gap-3">
              <div className="space-y-2">
                {/* Batch image selector pills if turn has multiple images */}
                {turn.length > 1 && (
                  <div className="flex flex-wrap items-center gap-1.5 pb-0.5">
                    <span className="text-[11px] font-mono text-[#8c7169] dark:text-[#a09088] font-medium">
                      Batch image:
                    </span>
                    <div className="flex flex-wrap items-center gap-1.5">
                      {turn.map((gen, idx) => {
                        const isCurrent = gen.id === selectedGen.id;
                        const hasPrompt = Boolean(gen.enhancedPrompt);
                        return (
                          <button
                            key={gen.id}
                            onClick={() =>
                              setSelectedGenIdByTurn((prev) => ({
                                ...prev,
                                [primary.batchId]: gen.id,
                              }))
                            }
                            className={`px-2 py-0.5 rounded-lg text-[11px] font-mono transition-all flex items-center gap-1 ${
                              isCurrent
                                ? 'bg-amber-500 text-white font-semibold shadow-xs'
                                : 'bg-[#f1eee7] dark:bg-[#232320] text-[#58423a] dark:text-[#a09088] hover:bg-[#e5e2db] dark:hover:bg-[#2e2e2a]'
                            }`}
                            title={`Inspect image ${idx + 1} (Seed #${gen.params.seed})`}
                          >
                            <span>#{idx + 1}</span>
                            <span className="opacity-75 text-[10px]">({gen.params.seed})</span>
                            {hasPrompt && (
                              <Sparkles
                                className={`w-2.5 h-2.5 ${
                                  isCurrent ? 'text-amber-200' : 'text-amber-600 dark:text-amber-400'
                                }`}
                              />
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Header with Prompt & Action buttons */}
                <div className="flex items-start justify-between gap-3">
                  <p
                    data-turn-prompt
                    className="font-['Inter'] font-medium text-sm sm:text-base text-[#1c1c18] dark:text-[#f3f0e9] leading-snug break-words"
                  >
                    {selectedGen.prompt}
                  </p>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      onClick={() => handleCopyPrompt(selectedGen.prompt, selectedGen.id)}
                      className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#8c7169] dark:text-[#a09088] transition-colors"
                      title={copiedId === selectedGen.id ? 'Copied' : 'Copy prompt'}
                    >
                      {copiedId === selectedGen.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                    {onDeleteTurn && (
                      <button
                        onClick={() => onDeleteTurn(primary.batchId)}
                        className="p-1 rounded-lg hover:bg-rose-500/10 text-[#8c7169] hover:text-rose-600 dark:hover:text-rose-400 transition-colors"
                        title="Delete turn"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Lineage */}
                {selectedGen.lineage && (
                  <div className="flex items-center gap-1.5 text-xs text-[#8c7169] dark:text-[#a09088] font-mono">
                    <span>↳ variation of seed {selectedGen.lineage.seed}</span>
                  </div>
                )}

                {/* Pending Enhancement Indicator */}
                {selectedGen.params.enhance && !cleanEnhanced && selectedGen.status !== 'failed' && (
                  <div className="pt-2 border-t border-[#e5e2db]/50 dark:border-[#2e2e2a]/50">
                    <div className="flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400 animate-pulse">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span className="font-medium">
                        {selectedGen.phase === 'enhancing'
                          ? `Enhancing prompt details for Image #${selectedIdx + 1} with Qwen-Image-PE...`
                          : `Enhancement pending for Image #${selectedIdx + 1}...`}
                      </span>
                    </div>
                  </div>
                )}

                {/* Enhanced Prompt & Thinking Trace */}
                {cleanEnhanced && (
                  <div className="pt-2 border-t border-[#e5e2db]/50 dark:border-[#2e2e2a]/50 space-y-1.5">
                    <div className="flex flex-wrap items-center justify-between gap-1 text-xs text-[#8c7169] dark:text-[#a09088]">
                      <div className="flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                        <span className="font-semibold text-[#1c1c18] dark:text-[#f3f0e9]">
                          Enhanced Prompt
                          {turn.length > 1 && (
                            <span className="ml-1.5 font-mono text-[11px] font-normal text-[#8c7169] dark:text-[#a09088]">
                              · Image {selectedIdx + 1} (#{selectedGen.params.seed})
                            </span>
                          )}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        {onRerunEnhanced && selectedGen.status === 'done' && (
                          <button
                            onClick={() => onRerunEnhanced(selectedGen)}
                            className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-800 dark:text-amber-300 font-medium transition-colors"
                            title="Rerun generation only, keeping this enhanced prompt as is"
                          >
                            <RotateCcw className="w-3 h-3" />
                            <span>Rerun with this</span>
                          </button>
                        )}
                        {onReuseEnhanced && (
                          <button
                            onClick={() => onReuseEnhanced(selectedGen)}
                            className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-md bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/15 text-[#58423a] dark:text-[#a09088] transition-colors"
                            title="Load enhanced prompt into composer"
                          >
                            <Sliders className="w-3 h-3" />
                            <span>To composer</span>
                          </button>
                        )}
                        <button
                          onClick={() => handleCopyPrompt(cleanEnhanced, `${selectedGen.id}-enh`)}
                          className="p-1 rounded-md hover:bg-black/5 dark:hover:bg-white/5 text-[#8c7169] dark:text-[#a09088] transition-colors"
                          title="Copy enhanced prompt"
                        >
                          {copiedId === `${selectedGen.id}-enh` ? (
                            <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                        {selectedGen.thinking && (
                          <button
                            onClick={() =>
                              setExpandedThinking((prev) => ({
                                ...prev,
                                [selectedGen.id]: !prev[selectedGen.id],
                              }))
                            }
                            className="flex items-center gap-1 text-[11px] text-[#a53705] dark:text-[#f38d58] hover:underline ml-0.5"
                          >
                            <Brain className="w-3 h-3" />
                            <span>{isThinkingOpen ? 'Hide' : 'Reasoning'}</span>
                            {generationTps > 0 && (
                              <span className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088] font-normal">
                                ({generationTps.toFixed(1)} tok/s)
                              </span>
                            )}
                            {isThinkingOpen ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-[#58423a] dark:text-[#b0a098] leading-relaxed bg-[#f6f3ec] dark:bg-[#20201d] p-2.5 rounded-lg border border-[#e5e2db]/40 dark:border-[#2e2e2a] max-h-28 overflow-y-auto">
                      {cleanEnhanced}
                    </p>

                    {/* Thinking Block */}
                    {isThinkingOpen && selectedGen.thinking && (
                      <div className="mt-2 p-3 rounded-lg bg-[#f1eee7] dark:bg-[#232320] text-xs font-mono text-[#58423a] dark:text-[#a09088] space-y-1.5 border border-[#e5e2db]/50 dark:border-[#2e2e2a] max-h-40 overflow-y-auto">
                        <div className="flex items-center justify-between text-[11px] font-sans font-semibold text-[#8c7169] uppercase tracking-wider">
                          <span>Internal Reasoning Plan:</span>
                          {generationTps > 0 && (
                            <span className="text-[10px] font-mono lowercase font-normal bg-black/5 dark:bg-white/5 px-1.5 py-0.5 rounded text-amber-700 dark:text-amber-400">
                              ⚡ {generationTps.toFixed(1)} tok/s
                              {selectedGen.timings?.prefill_tps ? ` · prefill ${selectedGen.timings.prefill_tps.toFixed(1)} tok/s` : ''}
                            </span>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap leading-relaxed">{selectedGen.thinking}</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Mobile-only chips row */}
                <div className="sm:hidden flex flex-wrap items-center gap-1.5 pt-1 text-[11px] font-mono text-[#58423a] dark:text-[#a09088]">
                  <span className="px-2 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320]">
                    {selectedGen.params.aspect}
                  </span>
                  <span className="px-2 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320]">
                    seed {selectedGen.params.seed}
                  </span>
                  {genSeconds != null && (
                    <span className="px-2 py-0.5 rounded-md bg-[#f1eee7] dark:bg-[#232320]" title={timeTitle}>
                      {formatDuration(genSeconds)}
                    </span>
                  )}
                </div>
              </div>

              {/* Bottom Actions Row for Quick Interactions */}
              <div className="flex items-center justify-between pt-2 border-t border-[#e5e2db]/40 dark:border-[#2e2e2a]/40 text-xs text-[#8c7169] dark:text-[#a09088]">
                <div className="flex items-center gap-2">
                  {selectedGen.status === 'done' && (
                    <>
                      <button
                        onClick={() => onVary(selectedGen)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#f1eee7] dark:bg-[#232320] hover:bg-[#e5e2db] dark:hover:bg-[#2e2e2a] text-[#1c1c18] dark:text-[#f3f0e9] transition-colors"
                        title="Vary seed"
                      >
                        <Dices className="w-3.5 h-3.5" />
                        <span>Vary</span>
                      </button>
                      <button
                        onClick={() => onRetry(selectedGen)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#f1eee7] dark:bg-[#232320] hover:bg-[#e5e2db] dark:hover:bg-[#2e2e2a] text-[#1c1c18] dark:text-[#f3f0e9] transition-colors"
                        title="Rerun exact"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Rerun</span>
                      </button>
                    </>
                  )}
                </div>

                {selectedGen.status === 'done' && selectedGen.imageUrl && (
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => onToggleFavorite(selectedGen.id)}
                      className={`p-1.5 rounded-md transition-colors ${
                        selectedGen.favorite
                          ? 'text-rose-500 bg-rose-500/10'
                          : 'hover:bg-black/5 dark:hover:bg-white/5 text-[#8c7169]'
                      }`}
                      title="Favorite"
                    >
                      <Heart className={`w-3.5 h-3.5 ${selectedGen.favorite ? 'fill-current' : ''}`} />
                    </button>
                    <button
                      onClick={() => onOpenLightbox(selectedGen)}
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/5 text-[#8c7169] transition-colors"
                      title="Inspect in Lightbox"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                    <a
                      href={selectedGen.imageUrl}
                      download={`${selectedGen.id}.png`}
                      className="p-1.5 rounded-md hover:bg-black/5 dark:hover:bg-white/5 text-[#8c7169] transition-colors"
                      title="Download full PNG"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </a>
                  </div>
                )}
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
};

interface TileCardProps {
  gen: Generation;
  isSelected?: boolean;
  onSelect?: () => void;
  onOpenLightbox: () => void;
  onToggleFavorite: () => void;
  onVary: () => void;
  onRetry: () => void;
  onCancel: () => void;
  onDelete?: () => void;
}

const TileCard: React.FC<TileCardProps> = ({
  gen,
  isSelected,
  onSelect,
  onOpenLightbox,
  onToggleFavorite,
  onVary,
  onRetry,
  onCancel,
  onDelete,
}) => {
  const aspectConfig = ASPECT_RATIOS[gen.params.aspect] || ASPECT_RATIOS['1:1'];
  const aspectRatioStyle = `${aspectConfig.ratioWidth} / ${aspectConfig.ratioHeight}`;

  // 1. DONE STATE
  if (gen.status === 'done') {
    if (!gen.imageUrl) {
      return (
        <div
          onClick={onSelect}
          className={`h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 rounded-xl bg-[#f1eee7] dark:bg-[#232320] flex flex-col justify-between p-3.5 cursor-pointer transition-all ${
            isSelected ? 'ring-2 ring-amber-500 shadow-md' : 'border border-[#e5e2db] dark:border-[#2e2e2a]'
          }`}
          style={{ aspectRatio: aspectRatioStyle }}
        >
          <div className="flex justify-between items-center text-xs text-[#8c7169] dark:text-[#a09088]">
            <span className="font-mono text-[11px] font-medium">{gen.params.aspect}</span>
            {onDelete && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="p-1 rounded-md text-rose-600 dark:text-rose-400 hover:bg-rose-500/10 transition-colors"
                title="Delete image"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <div className="my-auto text-center space-y-2">
            <Loader2 className="w-6 h-6 mx-auto animate-spin text-[#8c7169]" />
            <div className="text-xs text-[#8c7169] dark:text-[#a09088]">Loading image...</div>
          </div>
          <div className="text-[10px] font-mono text-center text-[#8c7169] dark:text-[#a09088]">
            seed {gen.params.seed}
          </div>
        </div>
      );
    }

    return (
      <div
        onClick={() => {
          if (onSelect) onSelect();
        }}
        onDoubleClick={onOpenLightbox}
        className={`group relative h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 overflow-hidden rounded-xl bg-[#f1eee7] dark:bg-[#232320] shadow-sm hover:shadow-md transition-all duration-300 cursor-pointer ${
          isSelected ? 'ring-2 ring-amber-500 shadow-md' : 'border border-[#e5e2db] dark:border-[#2e2e2a]'
        }`}
        style={{ aspectRatio: aspectRatioStyle }}
      >
        <img
          src={gen.imageUrl}
          alt={gen.prompt}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-102"
        />

        {/* Hover overlay & quick actions */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-between p-3">
          <div className="flex justify-between items-center">
            {onDelete && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="w-8 h-8 rounded-full flex items-center justify-center bg-black/40 text-white/80 hover:bg-rose-600 hover:text-white backdrop-blur-md transition-all hover:scale-110"
                title="Delete image"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite();
              }}
              className={`w-8 h-8 rounded-full flex items-center justify-center backdrop-blur-md transition-transform hover:scale-110 ml-auto ${
                gen.favorite ? 'bg-rose-500 text-white' : 'bg-black/40 text-white hover:bg-black/60'
              }`}
              title="Favorite"
            >
              <Heart className={`w-4 h-4 ${gen.favorite ? 'fill-current' : ''}`} />
            </button>
          </div>

          <div className="flex items-center justify-between text-white">
            <span className="font-mono text-[10px] opacity-80">
              {gen.params.width}×{gen.params.height}
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onVary();
                }}
                className="w-8 h-8 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Vary seed"
              >
                <Dices className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onOpenLightbox();
                }}
                className="w-8 h-8 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Inspect in lightbox"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <a
                href={gen.imageUrl}
                download={`${gen.id}.png`}
                onClick={(e) => e.stopPropagation()}
                className="w-8 h-8 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Download PNG"
              >
                <Download className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 2. RUNNING STATE (Generating with live phases)
  if (gen.status === 'running') {
    const isEnhance = gen.phase === 'enhancing' || gen.stage === 'enhance';
    // Progress is tokens against the expected finish line (85 % of the token limit). Past
    // it the bar turns indeterminate ("almost done") until the run ends, at the limit
    // at the latest.
    const enhanceLabel =
      gen.subStage === 'rewriting' ? 'rewriting prompt' : gen.subStage === 'thinking' ? 'thinking' : 'reading prompt';
    const enhanceTokens = gen.enhanceStats?.generated_tokens || 0;
    const enhanceLimit = gen.enhanceStats?.max_new_tokens || DEFAULT_ENHANCE_TOKEN_LIMIT;
    const enhanceTps = gen.enhanceStats?.generation_tps || 0;
    const enhanceExpected = enhanceLimit * ENHANCE_EXPECTED_SHARE;
    const enhanceAlmostDone = enhanceTokens >= enhanceExpected;
    const enhancePct = Math.min(99, Math.round((enhanceTokens / enhanceExpected) * 100));
    const enhanceEta = enhanceTps > 0 && !enhanceAlmostDone ? (enhanceExpected - enhanceTokens) / enhanceTps : null;

    const totalSteps = gen.totalSteps || gen.params.steps;
    // Encode and decode report no steps: show an indeterminate bar rather than a frozen 0 %.
    const hasSteps = (isEnhance && !enhanceAlmostDone) || gen.phase === 'denoising' || gen.stage === 'denoise';
    const pct = isEnhance ? enhancePct : Math.min(100, Math.round(((gen.step || 0) / totalSteps) * 100));

    const getPhaseDisplay = () => {
      switch (gen.phase) {
        case 'waking':
          return 'Waking generator';
        case 'loading_model':
          return 'Loading model';
        case 'encoding':
          return 'Reading prompt';
        case 'denoising':
          return 'Drawing';
        case 'decoding':
          return 'Finishing';
        case 'saving':
          return 'Saving';
        default:
          return gen.stage === 'encode'
            ? 'Reading prompt'
            : gen.stage === 'decode'
            ? 'Finishing'
            : 'Drawing';
      }
    };

    return (
      <div
        onClick={() => {
          if (onSelect) onSelect();
        }}
        className={`h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 rounded-xl bg-[#f6f3ec] dark:bg-[#20201d] flex flex-col justify-between p-3.5 shadow-xs cursor-pointer transition-all ${
          isSelected ? 'ring-2 ring-amber-500 shadow-md' : 'border border-[#e5e2db] dark:border-[#2e2e2a]'
        }`}
        style={{ aspectRatio: aspectRatioStyle }}
      >
        <div className="flex justify-between items-center text-xs text-[#8c7169] dark:text-[#a09088]">
          <div className="flex items-center gap-1.5 shrink-0 whitespace-nowrap">
            <span className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 font-mono text-[10px] font-semibold">
              {gen.params.aspect}
            </span>
            {gen.card !== undefined && gen.card !== null && (
              <span className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                GPU {gen.card}
              </span>
            )}
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onCancel();
            }}
            className="p-0.5 rounded-full text-rose-600 dark:text-rose-400 hover:bg-rose-500/15 transition-colors cursor-pointer"
            title="Cancel generation"
            aria-label="Cancel generation"
          >
            <XCircle className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="my-auto space-y-2 text-center px-3">
          <div
            className={`w-8 h-8 mx-auto rounded-full flex items-center justify-center ${
              isEnhance
                ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400 animate-pulse'
                : 'bg-[#a53705]/10 text-[#a53705] animate-spin'
            }`}
          >
            {isEnhance ? (
              <Sparkles className="w-4 h-4 animate-pulse" />
            ) : (
              <Loader2 className="w-4 h-4" />
            )}
          </div>

          {isEnhance ? (
            <div>
              <div className="font-['Plus_Jakarta_Sans'] font-semibold text-xs sm:text-sm text-[#1c1c18] dark:text-[#f3f0e9]">
                Enhancing
              </div>
              <div className="text-[11px] font-medium text-amber-700 dark:text-amber-400 mt-0.5 capitalize">
                {enhanceLabel}
              </div>
              {gen.enhanceStats?.generation_tps ? (
                <div className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088] mt-0.5">
                  ✨ {gen.enhanceStats.generation_tps.toFixed(1)} T/s
                  {enhanceEta != null && ` · ~${Math.round(enhanceEta)}s left`}
                </div>
              ) : gen.enhanceStats?.prefill_tps ? (
                <div className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088] mt-0.5">
                  ⚡ {gen.enhanceStats.prefill_tps.toFixed(1)} T/s prefill
                </div>
              ) : null}
            </div>
          ) : (
            <div>
              <div className="font-['Plus_Jakarta_Sans'] font-semibold text-xs sm:text-sm text-[#1c1c18] dark:text-[#f3f0e9]">
                {getPhaseDisplay()}
              </div>
              {(gen.phase === 'denoising' || gen.stage === 'denoise') && gen.secondsPerStep != null && (
                <div className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088] mt-0.5">
                  {gen.secondsPerStep.toFixed(2)}s/step · ~{Math.round(gen.etaSeconds || 0)}s left
                </div>
              )}
            </div>
          )}

          {/* Progress Bar */}
          <div
            className="w-full max-w-[180px] mx-auto h-1.5 rounded-full bg-[#e5e2db] dark:bg-[#333330] overflow-hidden"
            role="progressbar"
            aria-valuenow={hasSteps ? pct : undefined}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            {hasSteps ? (
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  isEnhance ? 'bg-amber-500 animate-pulse' : 'bg-[#a53705]'
                }`}
                style={{ width: `${Math.max(5, pct)}%` }}
              />
            ) : (
              <div
                className={`h-full w-1/3 rounded-full progress-indeterminate ${isEnhance ? 'bg-amber-500' : 'bg-[#a53705]'}`}
              />
            )}
          </div>
          <div className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088]">
            {hasSteps ? `${pct}% completed` : isEnhance ? 'almost done…' : 'working…'}
          </div>
        </div>

        <div className="text-[10px] font-mono text-center text-[#8c7169] dark:text-[#a09088]">
          seed #{gen.params.seed}
        </div>
      </div>
    );
  }

  // 3. QUEUED STATE
  if (gen.status === 'queued') {
    return (
      <div
        onClick={() => {
          if (onSelect) onSelect();
        }}
        className={`h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 rounded-xl bg-[#f8f6f0] dark:bg-[#1e1e1b] flex flex-col justify-between p-3.5 cursor-pointer transition-all ${
          isSelected
            ? 'ring-2 ring-amber-500 shadow-md border-solid border-amber-500/40'
            : 'border border-dashed border-[#dfdcce] dark:border-[#33332d]'
        }`}
        style={{ aspectRatio: aspectRatioStyle }}
      >
        <div className="flex justify-between items-center text-xs text-[#8c7169] dark:text-[#a09088]">
          <span className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 font-mono text-[10px] font-semibold">
            {gen.params.aspect}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onCancel();
            }}
            className="p-0.5 rounded-full text-rose-600 dark:text-rose-400 hover:bg-rose-500/15 transition-colors cursor-pointer"
            title="Cancel generation"
            aria-label="Cancel generation"
          >
            <XCircle className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="my-auto text-center space-y-2 px-3">
          <div className="w-8 h-8 mx-auto rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center text-[#8c7169]">
            <Loader2 className="w-4 h-4 animate-spin" />
          </div>
          <div className="font-['Plus_Jakarta_Sans'] font-medium text-sm text-[#1c1c18] dark:text-[#f3f0e9]">
            Queued
          </div>
          <p className="text-xs text-[#8c7169] dark:text-[#a09088] max-w-[190px] mx-auto leading-relaxed">
            {gen.phase === 'enhancing' || gen.stage === 'enhance'
              ? 'Queued for prompt enhancement'
              : gen.phase === 'encoding' || gen.stage === 'encode'
              ? 'Queued for text encoding'
              : gen.phase === 'denoising' || gen.stage === 'denoise'
              ? 'Queued for drawing (waiting for GPU)'
              : gen.phase === 'decoding' || gen.stage === 'decode'
              ? 'Queued for decoding'
              : 'Waiting for generator to be ready'}
          </p>
        </div>

        <div className="text-[10px] font-mono text-center text-[#8c7169] dark:text-[#a09088]">
          seed #{gen.params.seed} · Waiting for GPU
        </div>
      </div>
    );
  }

  // 4. FAILED STATE
  if (gen.status === 'failed') {
    return (
      <div
        onClick={() => {
          if (onSelect) onSelect();
        }}
        className={`h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 rounded-xl bg-rose-500/5 dark:bg-rose-500/10 flex flex-col justify-between p-3.5 text-center cursor-pointer transition-all ${
          isSelected ? 'ring-2 ring-amber-500 shadow-md' : 'border border-rose-500/20'
        }`}
        style={{ aspectRatio: aspectRatioStyle }}
      >
        <div className="text-left text-xs font-mono text-[#8c7169] dark:text-[#a09088]">
          <span className="px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-700 dark:text-rose-400 font-mono text-[10px] font-semibold">
            {gen.params.aspect}
          </span>
        </div>

        <div className="my-auto space-y-2 px-3">
          <div className="w-8 h-8 mx-auto rounded-full bg-rose-500/10 text-rose-600 dark:text-rose-400 flex items-center justify-center">
            <AlertCircle className="w-4 h-4" />
          </div>
          <div className="font-semibold text-sm text-rose-800 dark:text-rose-300">
            Generation Stopped
          </div>
          <p className="text-xs text-[#8c7169] dark:text-[#a09088] line-clamp-2">
            {gen.error || 'The generator stopped unexpectedly'}
          </p>
          <div className="flex items-center justify-center gap-2 mt-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRetry();
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90 transition-opacity"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
            {onDelete && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-rose-500/10 text-rose-600 dark:text-rose-400 hover:bg-rose-500/20 transition-colors"
                title="Delete card"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Delete</span>
              </button>
            )}
          </div>
        </div>

        <div className="text-[10px] font-mono text-[#8c7169]">
          seed #{gen.params.seed}
        </div>
      </div>
    );
  }

  // 5. CANCELLED STATE
  return (
    <div
      onClick={() => {
        if (onSelect) onSelect();
      }}
      className={`h-[190px] sm:h-[220px] md:h-[240px] max-w-full shrink-0 rounded-xl bg-[#f6f3ec]/60 dark:bg-[#20201d]/60 flex flex-col justify-between p-3.5 text-center cursor-pointer transition-all ${
        isSelected ? 'ring-2 ring-amber-500 shadow-md' : 'border border-[#e5e2db]/50 dark:border-[#2e2e2a]/50'
      }`}
      style={{ aspectRatio: aspectRatioStyle }}
    >
      <div className="text-left text-xs font-mono text-[#8c7169]">
        <span className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 font-mono text-[10px] font-semibold">
          {gen.params.aspect}
        </span>
      </div>
      <div className="my-auto space-y-2">
        <div className="text-xs text-[#8c7169] font-medium">Cancelled</div>
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRetry();
            }}
            className="inline-flex items-center gap-1 px-3 py-1 rounded-md text-xs font-medium bg-[#f1eee7] dark:bg-[#2c2c28] hover:bg-[#e5e2db] text-[#1c1c18] dark:text-[#f3f0e9]"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Rerun</span>
          </button>
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium text-rose-600 dark:text-rose-400 hover:bg-rose-500/10 transition-colors"
              title="Delete card"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
      <div className="text-[10px] font-mono text-[#8c7169]">seed #{gen.params.seed}</div>
    </div>
  );
};
