import React, { useState, useEffect, useLayoutEffect, useCallback, useRef } from 'react';
import type { Generation } from '../types';
import { sanitizeEnhancedPrompt } from '../utils/prompt';
import { formatDuration, formatGiB, generationSeconds, queueSeconds } from '../utils/timings';
import { ZoomableImage } from './ZoomableImage';
import { ReferenceStrip } from './ReferenceStrip';
import { ReasoningText } from './ReasoningText';
import {
  X,
  ChevronLeft,
  ChevronRight,
  Heart,
  Download,
  Dices,
  RotateCcw,
  Copy,
  Check,
  Sparkles,
  Brain,
  Info,
  ChevronDown,
  Trash2,
  ImagePlus,
  Link2,
} from 'lucide-react';
import { modelLabel } from '../utils/enhancers';

interface ViewerProps {
  generation: Generation | null;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
  onToggleFavorite: (id: string) => void;
  /** Fill the composer from this image with a new (random) seed. */
  onVary: (generation: Generation) => void;
  /** Fill the composer from this image with its seed locked. */
  onRerun: (generation: Generation) => void;
  /** Attach this image as a reference in the composer (absent: no room or not supported). */
  onUseAsReference?: (generation: Generation) => void;
  /** Ask to delete this image; once it is deleted the parent leaves the viewer. */
  onDelete?: (id: string) => void;
}

/**
 * The image viewer at `/images/{id}`: a page of its own (shareable, Back returns to the list),
 * filling the window instead of covering the timeline, so scrolling it scrolls nothing else.
 */
export const Viewer: React.FC<ViewerProps> = ({
  generation,
  onClose,
  onPrev,
  onNext,
  hasPrev = false,
  hasNext = false,
  onToggleFavorite,
  onVary,
  onRerun,
  onUseAsReference,
  onDelete,
}) => {
  // The image is larger than the stage: ←/→ pan it instead of changing images.
  const [zoomedIn, setZoomedIn] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [copiedEnhanced, setCopiedEnhanced] = useState(false);
  const [copiedSeed, setCopiedSeed] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  // Phones: once the details scroll, they take a quarter of the image's space. The stage keeps
  // its size (so the image keeps its scale) and moves up by half of that, under the header.
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageH, setStageH] = useState(0);
  const [detailsUp, setDetailsUp] = useState(false);

  // Reset sub-panels when viewing a different generation (the image remounts at fit)
  const [shownId, setShownId] = useState(generation?.id);
  if (generation?.id !== shownId) {
    setShownId(generation?.id);
    setCopiedPrompt(false);
    setCopiedEnhanced(false);
    setCopiedSeed(false);
    setShowThinking(false);
    setDetailsUp(false);
  }

  const hasImage = !!generation?.imageUrl;
  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStageH(el.clientHeight);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasImage]);

  // Keyboard navigation & actions
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!generation) return;
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowLeft' && !zoomedIn && hasPrev && onPrev) {
        onPrev();
      } else if (e.key === 'ArrowRight' && !zoomedIn && hasNext && onNext) {
        onNext();
      } else if (e.key === 'f' || e.key === 'F') {
        onToggleFavorite(generation.id);
      }
    },
    [generation, onClose, onPrev, onNext, hasPrev, hasNext, onToggleFavorite, zoomedIn]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (!generation || !generation.imageUrl) return null;

  const handleCopyPrompt = () => {
    navigator.clipboard.writeText(generation.prompt);
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const timings = generation.timings;
  const genSeconds = generationSeconds(timings);
  const waitSeconds = queueSeconds(timings);
  const totalSeconds = timings?.total_s ?? genSeconds;
  const timeRows: { label: string; seconds: number; detail?: string }[] = [];
  if (timings?.enhance_s != null) {
    // A cloud enhancer is named; the built-in one shows in Provenance.
    const enhanceProv = generation.provenance?.enhance;
    timeRows.push({
      label: 'Enhance',
      seconds: timings.enhance_s,
      detail: [
        enhanceProv?.agent === 'cloud' ? modelLabel(enhanceProv.model.id) : null,
        timings.generated_tokens != null
          ? `${timings.generated_tokens} / ${timings.max_new_tokens ?? generation.params.enhanceMaxTokens ?? '?'} tokens`
          : null,
        // Reasoning ran out of room: the limit was doubled, or at the ceiling cut for the reply.
        timings.limit_raised_from ? `limit raised from ${timings.limit_raised_from}` : null,
        timings.thinking_cut ? 'reasoning cut short' : null,
        timings.generation_tps ? `${timings.generation_tps.toFixed(1)} tok/s` : null,
      ]
        .filter(Boolean)
        .join(' · '),
    });
  }
  // An image of a send can take another image's encode (same prompt and references).
  if (timings?.encode_s != null) {
    timeRows.push({ label: timings.shared ? 'Encode (reused)' : 'Encode', seconds: timings.encode_s });
  }
  if (timings?.denoise_s != null) {
    timeRows.push({
      label: 'Denoise',
      seconds: timings.denoise_s,
      detail: timings.s_per_step != null ? `${timings.steps ?? generation.params.steps} × ${timings.s_per_step.toFixed(2)}s` : undefined,
    });
  }
  if (timings?.decode_s != null) {
    timeRows.push({ label: 'Decode', seconds: timings.decode_s, detail: timings.decode_mode });
  }

  const handleCopyEnhancedPrompt = () => {
    const clean = sanitizeEnhancedPrompt(generation.enhancedPrompt);
    if (clean) {
      navigator.clipboard.writeText(clean);
      setCopiedEnhanced(true);
      setTimeout(() => setCopiedEnhanced(false), 2000);
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const grow = detailsUp ? Math.round(stageH / 4) : 0;
  const handleDetailsScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (window.matchMedia('(min-width: 1280px)').matches) return;
    const el = e.currentTarget;
    if (el.scrollTop <= 0) setDetailsUp(false);
    // Only when there is more to read than the room it gains: it would snap back otherwise.
    else if (!detailsUp && el.scrollHeight - el.clientHeight > stageH / 4) setDetailsUp(true);
  };

  const handleCopySeed = () => {
    navigator.clipboard.writeText(generation.params.seed.toString());
    setCopiedSeed(true);
    setTimeout(() => setCopiedSeed(false), 2000);
  };

  return (
    <div
      className="h-full w-full overflow-hidden flex flex-col xl:flex-row bg-[#0b0b0a] text-white select-none pl-[env(safe-area-inset-left)] pr-[env(safe-area-inset-right)]"
      style={{ '--grow': `${grow}px` } as React.CSSProperties}
    >
      {/* Top Left Close & Header HUD on mobile */}
      <div className="xl:hidden flex items-center justify-between px-4 pb-4 pt-[max(1rem,env(safe-area-inset-top))] border-b border-white/10 bg-[#0b0b0a] z-20">
        <span className="font-mono text-xs text-white/70">
          {generation.params.width} × {generation.params.height} · {generation.params.aspect}
        </span>
        <button
          onClick={onClose}
          className="p-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Main Artwork Stage Area */}
      <div
        ref={stageRef}
        className="relative flex-1 flex items-center justify-center overflow-hidden p-4 sm:p-8 min-h-0 translate-y-[calc(var(--grow)*-0.5)] xl:translate-y-0 transition-transform duration-200 ease-out motion-reduce:transition-none"
      >
        {/* Navigation Chevron: Prev */}
        {hasPrev && onPrev && (
          <button
            onClick={onPrev}
            className="absolute left-4 top-1/2 -translate-y-1/2 z-20 w-11 h-11 rounded-full bg-black/50 hover:bg-black/80 backdrop-blur-md text-white/90 hover:text-white flex items-center justify-center transition-all hover:scale-110 shadow-lg"
            title="Previous image (←)"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
        )}

        {/* Navigation Chevron: Next */}
        {hasNext && onNext && (
          <button
            onClick={onNext}
            className="absolute right-4 xl:right-8 top-1/2 -translate-y-1/2 z-20 w-11 h-11 rounded-full bg-black/50 hover:bg-black/80 backdrop-blur-md text-white/90 hover:text-white flex items-center justify-center transition-all hover:scale-110 shadow-lg"
            title="Next image (→)"
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        )}

        <ZoomableImage
          key={generation.id}
          src={generation.imageUrl}
          thumbSrc={generation.thumbUrl}
          alt={generation.prompt}
          width={generation.params.width}
          height={generation.params.height}
          onZoomedChange={setZoomedIn}
        />
      </div>

      {/* Right Details Inspector Panel */}
      <aside className="w-full xl:w-[410px] flex-shrink-0 bg-[#161614] border-t xl:border-t-0 xl:border-l border-white/10 flex flex-col h-[calc(40vh+var(--grow))] -mt-[var(--grow)] xl:mt-0 xl:h-full transition-[height,margin] duration-200 ease-out motion-reduce:transition-none z-30 select-text overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-white/10 bg-[#1a1a17] flex items-center justify-between gap-3">
          <h3 className="font-['Plus_Jakarta_Sans'] font-semibold text-base text-white leading-none min-w-0 truncate">
            Inspector
          </h3>

          <div className="flex items-center gap-1.5 shrink-0">
            <button
              onClick={() => onToggleFavorite(generation.id)}
              className={`p-2 rounded-lg transition-colors ${
                generation.favorite
                  ? 'bg-rose-500 text-white'
                  : 'bg-white/10 hover:bg-white/20 text-white/80 hover:text-white'
              }`}
              title="Toggle favorite (F)"
            >
              <Heart className={`w-4 h-4 ${generation.favorite ? 'fill-current' : ''}`} />
            </button>
            <button
              onClick={handleCopyLink}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
              title="Copy a link to this image"
            >
              {copiedLink ? <Check className="w-4 h-4" /> : <Link2 className="w-4 h-4" />}
            </button>
            <a
              href={generation.imageUrl}
              download={`${generation.id}.png`}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
              title="Download PNG"
            >
              <Download className="w-4 h-4" />
            </a>
            {onDelete && (
              <button
                onClick={() => onDelete(generation.id)}
                className="p-2 rounded-lg bg-white/10 hover:bg-rose-600/80 text-white/80 hover:text-white transition-colors"
                title="Delete image"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={onClose}
              className="hidden xl:flex p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 hover:text-white transition-colors"
              title="Close viewer (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Action Buttons Matrix */}
        {/* Both fill the composer (prompt, Enhance, size, steps) and close the viewer */}
        <div className="p-4 bg-[#141412] border-b border-white/10 grid grid-cols-2 gap-2">
          <button
            onClick={() => onRerun(generation)}
            className="flex flex-col items-center justify-center p-2.5 rounded-xl bg-[#a53705] hover:bg-[#b83e06] text-white transition-all group"
            title="Fill the composer from this image, seed locked"
          >
            <RotateCcw className="w-4 h-4 group-hover:-rotate-90 transition-transform" />
            <span className="text-xs font-semibold mt-1">Rerun</span>
            <span className="text-[10px] text-white/70">Same seed · to composer</span>
          </button>
          <button
            onClick={() => onVary(generation)}
            className="flex flex-col items-center justify-center p-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white transition-all group"
            title="Fill the composer from this image, random seed"
          >
            <Dices className="w-4 h-4 group-hover:rotate-45 transition-transform" />
            <span className="text-xs font-semibold mt-1">Vary</span>
            <span className="text-[10px] text-white/70">New seed · to composer</span>
          </button>
          {onUseAsReference && (
            <button
              onClick={() => onUseAsReference(generation)}
              className="col-span-2 flex items-center justify-center gap-1.5 p-2 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
              title="Attach this image to the next prompt as a reference"
            >
              <ImagePlus className="w-4 h-4" />
              Use as reference
            </button>
          )}
        </div>

        {/* Scrollable Inspector Body */}
        <div
          onScroll={handleDetailsScroll}
          className="flex-1 overflow-y-auto p-4 sm:p-5 pb-[calc(1rem+env(safe-area-inset-bottom))] sm:pb-[calc(1.25rem+env(safe-area-inset-bottom))] space-y-5 text-xs text-white/90"
        >
          {/* Active Prompt Section */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-white/50">
              <span>Prompt</span>
              <button
                onClick={handleCopyPrompt}
                className="flex items-center gap-1 text-[#f38d58] hover:text-[#ff9d6a] transition-colors"
              >
                {copiedPrompt ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                <span>{copiedPrompt ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 font-sans text-xs sm:text-sm leading-relaxed text-white/95">
              {generation.prompt}
            </div>
            {generation.references && <ReferenceStrip references={generation.references} dark />}
          </div>

          {/* Enhanced Prompt & Reasoning (if available) */}
          {generation.enhancedPrompt && (
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-mono uppercase tracking-wider text-amber-400/80">
                <span className="flex items-center gap-1">
                  <Sparkles className="w-3 h-3" /> Enhanced Prompt
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopyEnhancedPrompt}
                    className="flex items-center gap-1 text-[#f38d58] hover:text-[#ff9d6a] transition-colors normal-case"
                  >
                    {copiedEnhanced ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedEnhanced ? 'Copied' : 'Copy'}</span>
                  </button>
                  {generation.hasThinking && (
                    <button
                      onClick={() => setShowThinking((v) => !v)}
                      className="flex items-center gap-1 text-[11px] text-amber-300 hover:underline normal-case"
                    >
                      <Brain className="w-3 h-3" />
                      <span>{showThinking ? 'Hide' : 'Thinking'}</span>
                    </button>
                  )}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-100 text-xs leading-relaxed">
                {sanitizeEnhancedPrompt(generation.enhancedPrompt)}
              </div>

              {showThinking && generation.hasThinking && (
                <ReasoningText
                  generation={generation}
                  className="p-3 rounded-xl bg-black/40 border border-white/10 font-mono text-[11px] text-white/75 leading-relaxed whitespace-pre-wrap max-h-48 overflow-y-auto"
                />
              )}
            </div>
          )}

          {/* Generation Specs Grid */}
          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-white/50">
              Parameters
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-white/5 border border-white/10">
                <div className="text-[10px] text-white/50">Dimensions</div>
                <div className="font-semibold text-white mt-0.5">
                  {generation.params.width} × {generation.params.height}
                </div>
                <div className="text-[10px] text-white/60">
                  Ratio {generation.params.aspect}
                  {generation.params.sizeMultiplier && generation.params.sizeMultiplier !== 'full'
                    ? ` · ${generation.params.sizeMultiplier}`
                    : ''}
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-white/5 border border-white/10">
                <div className="flex items-center justify-between text-[10px] text-white/50">
                  <span>Seed</span>
                  <button
                    onClick={handleCopySeed}
                    className="text-[#f38d58] hover:text-[#ff9d6a]"
                    title="Copy seed"
                  >
                    {copiedSeed ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  </button>
                </div>
                <div className="font-semibold text-white mt-0.5">{generation.params.seed}</div>
                <div className="text-[10px] text-white/60">manual seed</div>
              </div>

              <div className="p-2.5 rounded-lg bg-white/5 border border-white/10">
                <div className="text-[10px] text-white/50">Steps & Quality</div>
                <div className="font-semibold text-white mt-0.5">
                  {generation.params.steps} steps
                </div>
                <div className="text-[10px] text-white/60 capitalize">
                  {generation.params.quality}
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-white/5 border border-white/10">
                <div className="text-[10px] text-white/50">Total Time</div>
                <div className="font-semibold text-white mt-0.5">
                  {totalSeconds != null ? formatDuration(totalSeconds) : 'N/A'}
                </div>
                <div className="text-[10px] text-white/60">
                  {timings?.total_s != null ? 'submit to done' : genSeconds != null ? 'stages only' : '—'}
                </div>
              </div>
            </div>
          </div>

          {/* Time breakdown: stage times add up to generation time; the rest is waiting */}
          {timeRows.length > 0 && (
            <div className="space-y-2">
              <div className="text-[11px] font-mono uppercase tracking-wider text-white/50">Time</div>
              <div className="p-3 rounded-xl bg-white/5 border border-white/10 font-mono text-[11px] space-y-1.5 text-white/75">
                {timeRows.map((row) => (
                  <div key={row.label} className="flex justify-between gap-3">
                    <span className="text-white/50">{row.label}</span>
                    <span className="text-right">
                      {row.detail && <span className="text-white/40">{row.detail} · </span>}
                      {formatDuration(row.seconds)}
                    </span>
                  </div>
                ))}
                {genSeconds != null && (
                  <div className="flex justify-between gap-3 pt-1.5 border-t border-white/10 text-white">
                    <span>Generation</span>
                    <span>{formatDuration(genSeconds)}</span>
                  </div>
                )}
                {waitSeconds != null && (
                  <div className="flex justify-between gap-3">
                    <span className="text-white/50" title="Waiting for a free GPU plus starting workers and loading models">
                      Queue &amp; model loading
                    </span>
                    <span>{formatDuration(waitSeconds)}</span>
                  </div>
                )}
                {timings?.total_s != null && (
                  <div className="flex justify-between gap-3 font-semibold text-white">
                    <span>Total</span>
                    <span>{formatDuration(timings.total_s)}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* V3 Per-Stage Provenance Chain (§11.3) */}
          {generation.provenance && Object.keys(generation.provenance).length > 0 && (
            <div className="space-y-2">
              <div className="text-[11px] font-mono uppercase tracking-wider text-white/50">
                Provenance
              </div>
              <div className="p-3 rounded-xl bg-white/5 border border-white/10 font-mono text-[11px] space-y-2 text-white/80">
                {(['enhance', 'encode', 'denoise', 'decode'] as const).map((stage) => {
                  const prov = generation.provenance?.[stage];
                  if (!prov) return null;
                  const shortRev = prov.model?.rev ? prov.model.rev.slice(0, 7) : '';
                  const stageSeconds = prov.timings?.stage_s;
                  return (
                    <div key={stage} className="space-y-0.5 pb-1.5 last:pb-0 border-b last:border-b-0 border-white/5">
                      <div className="flex items-center justify-between text-white">
                        <span
                          className="uppercase text-[10px] tracking-wider text-white/60"
                          title={
                            prov.timings?.shared
                              ? 'Reused from another image of this send: same prompt and references, encoded once'
                              : undefined
                          }
                        >
                          {stage}
                          {prov.timings?.shared ? <span className="normal-case tracking-normal"> (reused)</span> : null}
                        </span>
                        <span>
                          {prov.agent}/{prov.executor}{' '}
                          <span className="text-white/50">({prov.backend_kind})</span>
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-white/50">
                        <span className="truncate max-w-[200px]" title={prov.model?.id}>
                          {prov.model?.id}
                          {shortRev ? `@${shortRev}` : ''}
                        </span>
                        <span>
                          {[
                            stageSeconds != null && formatDuration(stageSeconds),
                            prov.peak_mem_bytes && formatGiB(prov.peak_mem_bytes),
                          ]
                            .filter(Boolean)
                            .join(' · ')}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Technical Section (Collapsible) */}
          <div className="border-t border-white/10 pt-3">
            <button
              onClick={() => setShowTechnical((v) => !v)}
              className="w-full flex items-center justify-between text-[11px] font-mono text-white/60 hover:text-white"
            >
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" /> Pipeline Architecture
              </span>
              <ChevronDown
                className={`w-3.5 h-3.5 transition-transform ${showTechnical ? 'rotate-180' : ''}`}
              />
            </button>

            {showTechnical && (
              <div className="mt-2.5 p-3 rounded-xl bg-white/5 border border-white/10 font-mono text-[11px] space-y-1.5 text-white/75">
                <div className="flex justify-between">
                  <span className="text-white/50">Pipeline</span>
                  <span>QwenImage21Pipeline</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Stages</span>
                  <span>{generation.params.enhance ? 'enhance → ' : ''}encode → denoise → decode</span>
                </div>
                {timings?.decode_mode && (
                  <div className="flex justify-between">
                    <span className="text-white/50">Decode</span>
                    <span className="capitalize">{timings.decode_mode} VAE</span>
                  </div>
                )}
                {(generation.executor || generation.card != null) && (
                  <div className="flex justify-between">
                    <span className="text-white/50">Decoded on</span>
                    <span>{generation.executor || `GPU ${generation.card}`}</span>
                  </div>
                )}
                {generation.peakVram != null && (
                  <div className="flex justify-between">
                    <span className="text-white/50">Peak VRAM</span>
                    <span>{formatGiB(generation.peakVram)}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-white/50">Created</span>
                  <span>{new Date(generation.createdAt).toLocaleString()}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
};
