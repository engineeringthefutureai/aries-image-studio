import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Generation,
  JobParams,
  UserPreferences,
  WorkerHealth,
  AspectRatio,
} from './types';
import {
  loadGenerations,
  saveGenerations,
  loadPreferences,
  savePreferences,
  storeImageBlob,
  deleteImageBlob,
  restoreGenerationImageUrls,
} from './services/storage';
import {
  fetchWorkerHealth,
  submitBatchJobsApi,
  fetchCurrentJobsApi,
  cancelJobApi,
  fetchJobImageBlob,
  acknowledgeJobImageApi,
  mapStatusToPhase,
  fetchJobDetailApi,
  type JobSubmission,
} from './services/api';
import { sanitizeEnhancedPrompt } from './utils/prompt';
import { generateUlid } from './utils/ulid';

import { Header } from './components/Header';
import { Composer } from './components/Composer';
import { Timeline } from './components/Timeline';
import { Gallery } from './components/Gallery';
import { Lightbox } from './components/Lightbox';
import { StatusModal } from './components/StatusModal';
import { ShortcutsModal } from './components/ShortcutsModal';
import { ErrorBoundary } from './components/ErrorBoundary';

export function App() {
  const [generations, setGenerations] = useState<Generation[]>(() => loadGenerations());
  const [preferences, setPreferences] = useState<UserPreferences>(() => loadPreferences());
  const [activeView, setActiveView] = useState<'timeline' | 'gallery'>('timeline');

  // Latest generations for the poll loop, without re-subscribing the effect
  const generationsRef = useRef<Generation[]>(generations);
  useEffect(() => {
    generationsRef.current = generations;
  }, [generations]);

  // In-flight image download tracker
  const fetchingImageIdsRef = useRef<Set<string>>(new Set());

  // Ref to trigger job poll immediately on submission
  const pollJobsRef = useRef<(() => Promise<void>) | null>(null);

  // Backend telemetry & connectivity
  const [workerHealth, setWorkerHealth] = useState<WorkerHealth>({ state: 'offline' });

  // Active generation tracking
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Modals & Overlays
  const [lightboxGenId, setLightboxGenId] = useState<string | null>(null);
  const [isStatusOpen, setIsStatusOpen] = useState(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);

  // Composer prefill trigger
  const [prefilledPrompt, setPrefilledPrompt] = useState<string | null>(null);
  const [prefilledAspect, setPrefilledAspect] = useState<AspectRatio | null>(null);

  // Sync generations & preferences to local storage
  useEffect(() => {
    saveGenerations(generations);
  }, [generations]);

  useEffect(() => {
    savePreferences(preferences);
    if (preferences.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [preferences]);

  // Hydrate full-resolution image blobs from local IndexedDB across page refreshes
  useEffect(() => {
    let isCancelled = false;
    restoreGenerationImageUrls(generationsRef.current).then((hydrated) => {
      if (isCancelled) return;
      const urlMap = new Map<string, string>();
      for (const item of hydrated) {
        if (item.imageUrl) {
          urlMap.set(item.id, item.imageUrl);
        }
      }
      setGenerations((prev) =>
        prev.map((g) => {
          const restoredUrl = urlMap.get(g.id);
          if (restoredUrl && !g.imageUrl) {
            return { ...g, imageUrl: restoredUrl };
          }
          return g;
        })
      );
    });
    return () => {
      isCancelled = true;
    };
  }, []);

  // Periodic Health Polling
  const checkHealth = useCallback(async () => {
    const health = await fetchWorkerHealth();
    setWorkerHealth(health);
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

  // Job polling (ARCHITECTURE.md §7): every 1 s while anything is active, every 10 s
  // when idle, and on tab focus. Finished images are fetched, stored locally, then
  // acknowledged so the manager can delete its copy.
  useEffect(() => {
    let isCancelled = false;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let isPolling = false;

    const pollJobs = async () => {
      if (isPolling || isCancelled) return;
      isPolling = true;

      try {
        const activeJobs = await fetchCurrentJobsApi();
        if (isCancelled) return;

        const managerJobMap = new Map();
        for (const job of activeJobs) {
          managerJobMap.set(job.id, job);
        }

        const currentGenerations = generationsRef.current;
        for (const gen of currentGenerations) {
          const mJob = managerJobMap.get(gen.id);

          if (mJob) {
            const rawEnhanced = mJob.enhanced_prompt || gen.enhancedPrompt;
            const enhancedPrompt = rawEnhanced ? sanitizeEnhancedPrompt(rawEnhanced) : undefined;
            const thinking = mJob.thinking || gen.thinking;

            if (mJob.status === 'stored') {
              if (gen.status !== 'done' && !fetchingImageIdsRef.current.has(mJob.id)) {
                fetchingImageIdsRef.current.add(mJob.id);
                (async () => {
                  try {
                    const { blob, url } = await fetchJobImageBlob(mJob.id);
                    await storeImageBlob(mJob.id, blob);
                    await acknowledgeJobImageApi(mJob.id);
                    updateGeneration(mJob.id, {
                      status: 'done',
                      phase: 'done',
                      imageUrl: url,
                      completedAt: mJob.completed_at ? mJob.completed_at * 1000 : Date.now(),
                      timings: mJob.timings,
                      enhancedPrompt,
                      thinking,
                      enhanceStats: mJob.enhance_stats || undefined,
                    });
                  } catch (err) {
                    console.warn(`Failed to pull stored image for ${mJob.id}:`, err);
                  } finally {
                    fetchingImageIdsRef.current.delete(mJob.id);
                  }
                })();
              } else if (
                gen.status === 'done' &&
                ((enhancedPrompt && gen.enhancedPrompt !== enhancedPrompt) ||
                  (thinking && gen.thinking !== thinking))
              ) {
                updateGeneration(mJob.id, {
                  enhancedPrompt,
                  thinking,
                });
              }
            } else if (mJob.status === 'failed') {
              if (gen.status !== 'failed') {
                updateGeneration(mJob.id, {
                  status: 'failed',
                  error: mJob.error || 'Stage execution failed',
                  crashClass: mJob.crash_class || undefined,
                  enhancedPrompt,
                  thinking,
                });
              }
            } else if (mJob.status === 'cancelled') {
              if (gen.status !== 'cancelled') {
                updateGeneration(mJob.id, { status: 'cancelled' });
              }
            } else {
              const phase = mapStatusToPhase(mJob.status);
              const common = {
                phase,
                stage: mJob.stage,
                subStage: mJob.sub_stage || undefined,
                totalSteps: mJob.steps || gen.params.steps,
                enhancedPrompt,
                thinking,
                enhanceStats: mJob.enhance_stats || undefined,
              };

              if (phase === 'queued') {
                updateGeneration(mJob.id, {
                  ...common,
                  status: 'queued',
                  card: undefined,
                  step: 0,
                  secondsPerStep: undefined,
                  etaSeconds: undefined,
                });
              } else {
                updateGeneration(mJob.id, {
                  ...common,
                  status: 'running',
                  step: mJob.step || 0,
                  secondsPerStep: mJob.s_per_step ?? undefined,
                  etaSeconds: mJob.eta_s ?? undefined,
                  card: mJob.card ?? undefined,
                });
              }
            }
          }
        }
      } catch {
        // Manager offline or network hiccup; retry next interval
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

    // Refresh immediately on tab focus per ARCHITECTURE.md §7
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
  }, [updateGeneration]);

  // Backfill enhanced prompt & thinking for historical generations if missing
  const checkedEnhancedIdsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const missingEnhanced = generations.filter(
      (g) =>
        g.params.enhance &&
        !g.enhancedPrompt &&
        (g.status === 'done' || g.status === 'failed') &&
        !checkedEnhancedIdsRef.current.has(g.id)
    );
    if (missingEnhanced.length === 0) return;

    for (const gen of missingEnhanced) {
      checkedEnhancedIdsRef.current.add(gen.id);
      fetchJobDetailApi(gen.id)
        .then((job) => {
          if (job && (job.enhanced_prompt || job.thinking)) {
            const clean = job.enhanced_prompt ? sanitizeEnhancedPrompt(job.enhanced_prompt) : undefined;
            updateGeneration(gen.id, {
              enhancedPrompt: clean,
              thinking: job.thinking || undefined,
            });
          }
        })
        .catch(() => {});
    }
  }, [generations, updateGeneration]);

  // Submit one turn: `count` jobs with consecutive seeds, sent as one batch
  const handleSubmitTurn = async (prompt: string, params: JobParams) => {
    if (isSubmitting) return;
    setIsSubmitting(true);

    const batchId = `batch-${Date.now()}`;
    const count = preferences.count || 1;
    const now = Date.now();

    const newGenerations: Generation[] = [];
    const payloads: JobSubmission[] = [];

    for (let i = 0; i < count; i++) {
      const jobId = generateUlid(now);
      const seed = params.seed + i;

      newGenerations.push({
        id: jobId,
        batchId,
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
        aspect: params.aspect,
        width: params.width,
        height: params.height,
        steps: params.steps,
        seed,
        enhance: params.enhance || false,
        decode_mode: 'untiled',
      });
    }

    setGenerations((prev) => [...newGenerations, ...prev]);
    setActiveView('timeline');
    pollJobsRef.current?.();

    try {
      const res = await submitBatchJobsApi(payloads);
      if (!res.ok) {
        console.error('Batch submission failed:', res.error);
        for (const gen of newGenerations) {
          updateGeneration(gen.id, {
            status: 'failed',
            error: res.error || 'Failed to submit job to Manager',
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

  // Cancel Job on Manager
  const handleCancelJob = async (jobId: string) => {
    await cancelJobApi(jobId);
    updateGeneration(jobId, { status: 'cancelled' });
  };

  // Toggle Favorite
  const handleToggleFavorite = (id: string) => {
    setGenerations((prev) =>
      prev.map((g) => (g.id === id ? { ...g, favorite: !g.favorite } : g))
    );
  };

  // Vary: Same prompt & params, new random seed
  const handleVary = (gen: Generation) => {
    const newSeed = Math.floor(Math.random() * 1000000);
    handleSubmitTurn(gen.prompt, {
      ...gen.params,
      seed: newSeed,
    });
  };

  // Rerun Exact: Same prompt, exact same seed
  const handleRerunExact = (gen: Generation) => {
    handleSubmitTurn(gen.prompt, { ...gen.params });
  };

  // Rerun Enhanced: Rerun generation only, keeping the enhanced prompt as is (skips re-enhancement)
  const handleRerunEnhanced = (gen: Generation) => {
    const promptText = sanitizeEnhancedPrompt(gen.enhancedPrompt);
    if (!promptText) return;
    handleSubmitTurn(promptText, {
      ...gen.params,
      enhance: false, // Run generation directly with enhanced prompt
    });
    if (lightboxGenId) setLightboxGenId(null);
  };

  // Reuse Enhanced Settings: Load enhanced prompt into Composer with enhance disabled
  const handleReuseEnhanced = (gen: Generation) => {
    const promptText = sanitizeEnhancedPrompt(gen.enhancedPrompt);
    if (!promptText) return;
    setPrefilledPrompt(promptText);
    setPrefilledAspect(gen.params.aspect);
    setPreferences((prev) => ({
      ...prev,
      quality: gen.params.quality === 'draft' ? 'draft' : 'standard',
      steps: gen.params.steps,
      sizeMultiplier: gen.params.sizeMultiplier || 'full',
      isSeedLocked: true,
      lockedSeed: gen.params.seed,
      enhance: false,
    }));
    if (lightboxGenId) setLightboxGenId(null);
  };

  // Retry Failed Job
  const handleRetry = (gen: Generation) => {
    handleSubmitTurn(gen.prompt, { ...gen.params });
  };

  // Delete a single generation card
  const handleDeleteGeneration = async (id: string) => {
    await deleteImageBlob(id);
    setGenerations((prev) => prev.filter((g) => g.id !== id));
    if (lightboxGenId === id) {
      setLightboxGenId(null);
    }
  };

  // Delete an entire turn / batch
  const handleDeleteTurn = async (batchId: string) => {
    const toDelete = generations.filter((g) => g.batchId === batchId);
    for (const g of toDelete) {
      await deleteImageBlob(g.id);
    }
    setGenerations((prev) => prev.filter((g) => g.batchId !== batchId));
    if (toDelete.some((g) => g.id === lightboxGenId)) {
      setLightboxGenId(null);
    }
  };

  // Reuse Settings: Load into Composer
  const handleReuseSettings = (gen: Generation) => {
    setPrefilledPrompt(gen.prompt);
    setPrefilledAspect(gen.params.aspect);
    setPreferences((prev) => ({
      ...prev,
      quality: gen.params.quality === 'draft' ? 'draft' : 'standard',
      steps: gen.params.steps,
      sizeMultiplier: gen.params.sizeMultiplier || 'full',
      isSeedLocked: true,
      lockedSeed: gen.params.seed,
    }));
    if (lightboxGenId) setLightboxGenId(null);
  };

  // Select Sample Prompt
  const handleSelectSamplePrompt = (prompt: string, aspect?: AspectRatio) => {
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
  const completedList = generations.filter((g) => g.status === 'done' && g.imageUrl);
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
      <main className="flex-1 flex flex-col pb-48 sm:pb-44">
        <ErrorBoundary fallbackTitle="Timeline rendering error">
          {activeView === 'timeline' ? (
            <Timeline
              generations={generations}
              onOpenLightbox={(gen) => setLightboxGenId(gen.id)}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onRetry={handleRetry}
              onCancel={handleCancelJob}
              onSelectPrompt={handleSelectSamplePrompt}
              onRerunEnhanced={handleRerunEnhanced}
              onReuseEnhanced={handleReuseEnhanced}
              onDelete={handleDeleteGeneration}
              onDeleteTurn={handleDeleteTurn}
            />
          ) : (
            <Gallery
              generations={generations}
              onOpenLightbox={(gen) => setLightboxGenId(gen.id)}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onDelete={handleDeleteGeneration}
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
        prefilledPrompt={prefilledPrompt}
        prefilledAspect={prefilledAspect}
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
          onReuseSettings={handleReuseSettings}
          onRerunExact={handleRerunExact}
          onRerunEnhanced={handleRerunEnhanced}
          onReuseEnhanced={handleReuseEnhanced}
          onDelete={handleDeleteGeneration}
        />
      )}
    </div>
  );
}

export default App;
