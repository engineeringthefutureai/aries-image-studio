import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Generation,
  JobParams,
  UserPreferences,
  WorkerHealth,
  CapabilitiesInfo,
  AspectRatio,
  ComposerSource,
} from './types';
import { ASPECT_RATIOS, qualityForSteps } from './types';
import { DEFAULT_PREFERENCES } from './services/preferences';
import {
  fetchWorkerHealth,
  fetchCapabilitiesApi,
  submitBatchJobsApi,
  fetchJobsApi,
  cancelJobApi,
  retryJobApi,
  deleteJobApi,
  setFavoriteApi,
  fetchPreferencesApi,
  savePreferencesApi,
  type JobSubmission,
} from './services/api';
import { sanitizeEnhancedPrompt } from './utils/prompt';
import { generateUlid } from './utils/ulid';

import { Header } from './components/Header';
import { groupTurns } from './utils/turns';
import { Composer } from './components/Composer';
import { Timeline } from './components/Timeline';
import { Gallery } from './components/Gallery';
import { Lightbox } from './components/Lightbox';
import { StatusModal } from './components/StatusModal';
import { ShortcutsModal } from './components/ShortcutsModal';
import { ErrorBoundary } from './components/ErrorBoundary';

const newestFirst = (a: Generation, b: Generation) => b.createdAt - a.createdAt;

export function App() {
  // Everything shown comes from Studio: jobs (with images, favorites and UI metadata) and
  // preferences. Nothing is kept in browser storage.
  const [generations, setGenerations] = useState<Generation[]>([]);
  // False until Studio's first job list arrives (the views show a loader, not the starter).
  const [jobsLoaded, setJobsLoaded] = useState(false);
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [activeView, setActiveView] = useState<'timeline' | 'gallery'>('timeline');

  // Latest generations for the poll loop, without re-subscribing the effect
  const generationsRef = useRef<Generation[]>(generations);
  useEffect(() => {
    generationsRef.current = generations;
  }, [generations]);

  // Ref to trigger job poll immediately on submission
  const pollJobsRef = useRef<(() => Promise<void>) | null>(null);

  // Backend telemetry, capabilities & connectivity
  const [workerHealth, setWorkerHealth] = useState<WorkerHealth>({ state: 'offline' });
  const [capabilities, setCapabilities] = useState<CapabilitiesInfo | null>(null);

  // Active generation tracking
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Modals & Overlays
  const [lightboxGenId, setLightboxGenId] = useState<string | null>(null);
  const [isStatusOpen, setIsStatusOpen] = useState(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);

  // Composer prefill trigger
  const [prefilledPrompt, setPrefilledPrompt] = useState<string | null>(null);
  const [prefilledAspect, setPrefilledAspect] = useState<AspectRatio | null>(null);
  const [composerSource, setComposerSource] = useState<ComposerSource | null>(null);

  // Preferences: load once from Studio, then save each change (debounced). Nothing is
  // saved before the load, so defaults never overwrite what Studio has.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const saved = await fetchPreferencesApi();
      if (cancelled) return;
      if (saved === null) {
        setTimeout(load, 3000); // Studio unreachable: retry
        return;
      }
      setPreferences({ ...DEFAULT_PREFERENCES, ...saved });
      setPreferencesLoaded(true);
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', preferences.theme === 'dark');
    if (!preferencesLoaded) return;
    const timer = setTimeout(() => savePreferencesApi(preferences), 300);
    return () => clearTimeout(timer);
  }, [preferences, preferencesLoaded]);

  // Periodic Health & Capabilities Polling
  const checkHealth = useCallback(async () => {
    const [health, caps] = await Promise.all([
      fetchWorkerHealth(),
      fetchCapabilitiesApi(),
    ]);
    setWorkerHealth(health);
    if (caps) setCapabilities(caps);
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Update specific generation helper
  const updateGeneration = useCallback((id: string, updates: Partial<Generation>) => {
    setGenerations((prev) =>
      prev.map((g) => (g.id === id ? { ...g, ...updates } : g))
    );
  }, []);

  // Job sync (ARCHITECTURE_V3.md §8): a full load first, then only what changed since the
  // last version (plus deletions); every 1 s while anything is active, every 10 s when
  // idle, and on tab focus.
  useEffect(() => {
    let isCancelled = false;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let isPolling = false;
    let version: number | undefined;

    const pollJobs = async () => {
      if (isPolling || isCancelled) return;
      isPolling = true;

      try {
        const page = await fetchJobsApi(version);
        if (isCancelled) return;
        const full = version === undefined;
        version = page.version;
        const changed = new Map(page.jobs.map((g) => [g.id, g]));
        const deleted = new Set(page.deleted);
        setGenerations((prev) => {
          // A full load replaces everything except jobs just submitted from this tab that
          // Studio has not listed yet; an incremental one patches changed jobs in.
          const kept = prev.filter((g) => {
            if (deleted.has(g.id) || changed.has(g.id)) return false;
            return full ? g.status === 'queued' && !g.completedAt && Date.now() - g.createdAt < 30_000 : true;
          });
          return [...kept, ...changed.values()].sort(newestFirst);
        });
        setJobsLoaded(true);
      } catch {
        // Studio offline or network hiccup; retry next interval
      } finally {
        isPolling = false;
        if (!isCancelled) {
          const hasActive = generationsRef.current.some(
            (g) => g.status === 'queued' || g.status === 'running'
          );
          const intervalMs = hasActive ? 1000 : 10000;
          if (timerId) clearTimeout(timerId);
          timerId = setTimeout(pollJobs, intervalMs);
        }
      }
    };

    pollJobsRef.current = pollJobs;
    pollJobs();

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        pollJobs();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      isCancelled = true;
      if (timerId) clearTimeout(timerId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Submit one turn: `count` jobs with consecutive seeds, sent as one batch
  /** Submits one turn: every job the composer built (rows × count, see utils/composer.ts). */
  const handleSubmitTurn = async (prompt: string, jobs: JobParams[]) => {
    if (isSubmitting || jobs.length === 0) return;
    setIsSubmitting(true);
    setComposerSource(null);

    const batchId = `batch-${Date.now()}`;
    const now = Date.now();

    const newGenerations: Generation[] = [];
    const payloads: JobSubmission[] = [];

    for (const [index, params] of jobs.entries()) {
      const jobId = generateUlid(now);
      const seed = params.seed;

      newGenerations.push({
        id: jobId,
        batchId,
        batchIndex: index,
        prompt,
        params: { ...params, seed },
        status: 'queued',
        phase: 'queued',
        stage: params.enhance ? 'enhance' : 'encode',
        createdAt: now,
      });

      payloads.push({
        id: jobId,
        prompt,
        width: params.width,
        height: params.height,
        steps: params.steps,
        seed,
        enhance: params.enhance || false,
        tiling: params.tiling || false,
        meta: {
          batch_id: batchId,
          aspect: params.aspect,
          size_multiplier: params.sizeMultiplier,
          quality: params.quality,
          index,
        },
      });
    }

    setGenerations((prev) => [...newGenerations, ...prev]);
    setActiveView('timeline');
    // The new turn is at the top of the timeline: show it.
    window.scrollTo({ top: 0, behavior: 'smooth' });
    pollJobsRef.current?.();

    try {
      const res = await submitBatchJobsApi(payloads);
      if (!res.ok) {
        console.error('Batch submission failed:', res.error);
        for (const gen of newGenerations) {
          updateGeneration(gen.id, {
            status: 'failed',
            error: res.error || 'Failed to submit job',
          });
        }
      } else {
        pollJobsRef.current?.();
      }
    } catch (err) {
      console.error('Error submitting batch:', err);
      for (const gen of newGenerations) {
        updateGeneration(gen.id, {
          status: 'failed',
          error: err instanceof Error ? err.message : 'Error submitting job',
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    await cancelJobApi(jobId);
    updateGeneration(jobId, { status: 'cancelled' });
    pollJobsRef.current?.();
  };

  // Toggle Favorite
  const handleToggleFavorite = async (id: string) => {
    const gen = generationsRef.current.find((g) => g.id === id);
    if (!gen) return;
    const favorite = !gen.favorite;
    updateGeneration(id, { favorite });
    if (!(await setFavoriteApi(id, favorite))) {
      updateGeneration(id, { favorite: !favorite });
    }
  };

  /**
   * Rerun and Vary: fill the composer from an image (prompt, Enhance, aspect, size, steps)
   * instead of submitting. Rerun locks the image's seed; Vary leaves the seed random. The
   * composer then offers the original or the enhanced prompt (ComposerSource).
   */
  const handleToComposer = (gen: Generation, { newSeed }: { newSeed: boolean }) => {
    const enhanced = sanitizeEnhancedPrompt(gen.enhancedPrompt) || undefined;
    setComposerSource({
      seed: gen.params.seed,
      original: gen.prompt,
      enhanced,
      originalEnhance: !!gen.params.enhance,
      mode: 'original',
    });
    setPrefilledPrompt(gen.prompt);
    const quality = qualityForSteps(gen.params.steps);
    const tiling = !!gen.params.tiling;
    setPreferences((prev) => ({
      ...prev,
      // Sizes outside the presets (older images) keep the composer's current aspect.
      aspect: gen.params.aspect in ASPECT_RATIOS ? gen.params.aspect : prev.aspect,
      sizeMultiplier: gen.params.sizeMultiplier || 'full',
      quality,
      steps: gen.params.steps,
      enhance: !!gen.params.enhance,
      tiling,
      isSeedLocked: !newSeed,
      lockedSeed: newSeed ? prev.lockedSeed : gen.params.seed,
      // A fixed seed, custom steps and tiling are advanced options: open them when the
      // image needs one, so they apply. Extra rows are cleared to match this one image.
      advancedOpen: prev.advancedOpen || !newSeed || quality === 'custom' || tiling,
      extraRows: [],
    }));
    setLightboxGenId(null);
  };
  const handleRerun = (gen: Generation) => handleToComposer(gen, { newSeed: false });
  const handleVary = (gen: Generation) => handleToComposer(gen, { newSeed: true });

  // Retry Failed Job (V3: re-queues failed task in place if intermediate artifacts still exist, else re-submits)
  const handleRetry = async (gen: Generation) => {
    const result = await retryJobApi(gen.id);
    if (result === 'retried') {
      updateGeneration(gen.id, { status: 'queued', phase: 'queued', error: undefined });
      pollJobsRef.current?.();
    } else if (result === 'refused') {
      // Not failed or cancelled (any more): show what Studio has instead of guessing.
      pollJobsRef.current?.();
    } else {
      // Gone from Studio (or unreachable): offer it in the composer instead.
      handleRerun(gen);
    }
  };

  // Delete a single generation card
  const handleDeleteGeneration = async (id: string) => {
    if (!(await deleteJobApi(id))) return;
    setGenerations((prev) => prev.filter((g) => g.id !== id));
    if (lightboxGenId === id) {
      setLightboxGenId(null);
    }
  };

  // Delete an entire turn / batch
  const handleDeleteTurn = async (batchId: string) => {
    const toDelete = generations.filter((g) => g.batchId === batchId);
    const deleted = new Set<string>();
    for (const g of toDelete) {
      if (await deleteJobApi(g.id)) deleted.add(g.id);
    }
    setGenerations((prev) => prev.filter((g) => !deleted.has(g.id)));
    if (lightboxGenId && deleted.has(lightboxGenId)) {
      setLightboxGenId(null);
    }
  };

  // Select Sample Prompt
  const handleSelectSamplePrompt = (prompt: string, aspect?: AspectRatio) => {
    setComposerSource(null);
    setPrefilledPrompt(prompt);
    if (aspect) setPrefilledAspect(aspect);
  };

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing inside an input or textarea
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      if (e.key === 't' || e.key === 'T') {
        setActiveView('timeline');
      } else if (e.key === 'g' || e.key === 'G') {
        setActiveView('gallery');
      } else if (e.key === '/') {
        e.preventDefault();
        const composerTextarea = document.querySelector('textarea') as HTMLTextAreaElement | null;
        composerTextarea?.focus();
      } else if (e.key === '?') {
        setIsShortcutsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Lightbox navigation helpers
  // ←/→ follow the open view: the timeline's order (turns newest first, each left to right),
  // or the gallery's (newest first).
  const completedList = (activeView === 'timeline' ? groupTurns(generations).flat() : generations).filter(
    (g) => g.status === 'done' && g.imageUrl,
  );
  const currentLightboxIndex = completedList.findIndex((g) => g.id === lightboxGenId);
  const lightboxGen = currentLightboxIndex !== -1 ? completedList[currentLightboxIndex] : null;

  const handlePrevLightbox = () => {
    if (currentLightboxIndex > 0) {
      setLightboxGenId(completedList[currentLightboxIndex - 1].id);
    }
  };

  const handleNextLightbox = () => {
    if (currentLightboxIndex < completedList.length - 1) {
      setLightboxGenId(completedList[currentLightboxIndex + 1].id);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#fcf9f2] dark:bg-[#121210] text-[#1c1c18] dark:text-[#f3f0e9] font-['Inter'] antialiased transition-colors duration-200">
      {/* Top Navigation Header */}
      <Header
        activeView={activeView}
        onViewChange={setActiveView}
        workerHealth={workerHealth}
        onOpenStatus={() => setIsStatusOpen(true)}
        theme={preferences.theme}
        onToggleTheme={() =>
          setPreferences((p) => ({ ...p, theme: p.theme === 'light' ? 'dark' : 'light' }))
        }
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col pb-24">
        <ErrorBoundary fallbackTitle="Timeline rendering error">
          {activeView === 'timeline' ? (
            <Timeline
              generations={generations}
              onOpenLightbox={(gen) => setLightboxGenId(gen.id)}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onRetry={handleRetry}
              onRerun={handleRerun}
              onCancel={handleCancelJob}
              onSelectPrompt={handleSelectSamplePrompt}
              onDelete={handleDeleteGeneration}
              onDeleteTurn={handleDeleteTurn}
              loading={!jobsLoaded}
            />
          ) : (
            <Gallery
              generations={generations}
              onOpenLightbox={(gen) => setLightboxGenId(gen.id)}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onDelete={handleDeleteGeneration}
              loading={!jobsLoaded}
            />
          )}
        </ErrorBoundary>
      </main>

      {/* Docked Composer Bar */}
      <Composer
        preferences={preferences}
        onUpdatePreferences={(newPrefs) => setPreferences((p) => ({ ...p, ...newPrefs }))}
        onSubmit={handleSubmitTurn}
        isSubmitting={isSubmitting}
        workerState={workerHealth.state}
        capabilities={capabilities}
        prefilledPrompt={prefilledPrompt}
        prefilledAspect={prefilledAspect}
        source={composerSource}
        onSourceChange={setComposerSource}
        onClearPrefill={() => {
          setPrefilledPrompt(null);
          setPrefilledAspect(null);
        }}
      />

      {/* Telemetry Status Panel Modal */}
      <StatusModal
        isOpen={isStatusOpen}
        onClose={() => setIsStatusOpen(false)}
        health={workerHealth}
        onRefresh={checkHealth}
      />

      {/* Keyboard Shortcuts Cheat-sheet Modal */}
      <ShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />

      {/* Lightbox Artwork Inspector */}
      {lightboxGen && (
        <Lightbox
          generation={lightboxGen}
          onClose={() => setLightboxGenId(null)}
          onPrev={handlePrevLightbox}
          onNext={handleNextLightbox}
          hasPrev={currentLightboxIndex > 0}
          hasNext={currentLightboxIndex < completedList.length - 1}
          onToggleFavorite={handleToggleFavorite}
          onVary={handleVary}
          onRerun={handleRerun}
          onDelete={handleDeleteGeneration}
        />
      )}
    </div>
  );
}

export default App;
