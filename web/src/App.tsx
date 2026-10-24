import { useState, useEffect, useLayoutEffect, useCallback, useMemo, useRef } from 'react';
import type {
  Generation,
  JobParams,
  UserPreferences,
  WorkerHealth,
  CapabilitiesInfo,
  AspectRatio,
  ComposerSource,
  ComposerReference,
} from './types';
import { ASPECT_RATIOS, qualityForSteps } from './types';
import { DEFAULT_PREFERENCES } from './services/preferences';
import {
  fetchWorkerHealth,
  fetchCapabilitiesApi,
  submitBatchJobsApi,
  fetchJobChangesApi,
  fetchJobApi,
  fetchGalleryFacetsApi,
  cancelJobApi,
  retryJobApi,
  deleteJobApi,
  setFavoriteApi,
  fetchPreferencesApi,
  savePreferencesApi,
  type GalleryFacets,
  type JobQuery,
  type JobSubmission,
} from './services/api';
import { usePagedJobs } from './hooks/usePagedJobs';
import {
  useRoute,
  navigate,
  leavePage,
  restoreScroll,
  viewerFrom,
  galleryHref,
  imageHref,
  type GalleryFilters,
} from './utils/router';
import { sanitizeEnhancedPrompt } from './utils/prompt';
import { fromGeneration, fromJobReference, referencePayload, releasePreview } from './utils/references';
import { generateUlid } from './utils/ulid';

import { Header } from './components/Header';
import { groupTurns } from './utils/turns';
import { Composer } from './components/Composer';
import { Timeline } from './components/Timeline';
import { Gallery } from './components/Gallery';
import { Viewer } from './components/Viewer';
import { StatusPage } from './components/StatusPage';
import { PageLoader } from './components/PageLoader';
import { ShortcutsModal } from './components/ShortcutsModal';
import { ConfirmDialog, type ConfirmRequest } from './components/ConfirmDialog';
import { ErrorBoundary } from './components/ErrorBoundary';

/** The timeline lists every job. */
const TIMELINE_QUERY: JobQuery = {};
const NO_FILTERS: GalleryFilters = { favorites: false, aspect: '', q: '' };

/** The gallery's Studio query: an aspect label stands for its preset's exact size ratio. */
function galleryQuery(f: GalleryFilters): JobQuery {
  const preset = ASPECT_RATIOS[f.aspect as AspectRatio];
  const ratio = preset ? `${preset.width}:${preset.height}` : f.aspect || undefined;
  return { doneOnly: true, favorites: f.favorites || undefined, q: f.q.trim() || undefined, ratio };
}

/** Studio's system view and capabilities, polled together. */
const fetchStatus = () => Promise.all([fetchWorkerHealth(), fetchCapabilitiesApi()]);

const sameFilters = (a: GalleryFilters, b: GalleryFilters) =>
  a.favorites === b.favorites && a.aspect === b.aspect && a.q === b.q;

export function App() {
  // Everything shown comes from Studio: jobs (with images, favorites and UI metadata) and
  // preferences. Nothing is kept in browser storage.
  const route = useRoute();
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);

  // Each page is its own URL (utils/router.ts). The gallery keeps the filters it was left
  // with, so the viewer's ←/→ and Back return to the same list.
  const [galleryFilters, setGalleryFilters] = useState<GalleryFilters | null>(null);
  if (route.name === 'gallery' && (!galleryFilters || !sameFilters(galleryFilters, route.filters))) {
    setGalleryFilters(route.filters);
  }
  const galleryJobQuery = useMemo(() => (galleryFilters ? galleryQuery(galleryFilters) : null), [galleryFilters]);
  const galleryLink = galleryHref(galleryFilters || NO_FILTERS);

  // Job lists, a page at a time (infinite scroll). The timeline's first page also gives
  // Studio's version, where polling for changes starts.
  const versionRef = useRef<number | undefined>(undefined);
  const timeline = usePagedJobs(TIMELINE_QUERY, (version) => {
    versionRef.current ??= version;
  });
  const gallery = usePagedJobs(galleryJobQuery);
  const [facets, setFacets] = useState<GalleryFacets | null>(null);

  // Latest timeline for the poll loop, without re-subscribing the effect
  const timelineItemsRef = useRef<Generation[]>(timeline.items);
  useEffect(() => {
    timelineItemsRef.current = timeline.items;
  }, [timeline.items]);

  // A job opened by its link that is in neither list (yet): fetched on its own.
  const [linkedJob, setLinkedJob] = useState<{ id: string; gen: Generation | null } | null>(null);

  // Ref to trigger job poll immediately on submission
  const pollJobsRef = useRef<(() => Promise<void>) | null>(null);

  // Backend telemetry, capabilities & connectivity
  const [workerHealth, setWorkerHealth] = useState<WorkerHealth>({ state: 'offline' });
  const [capabilities, setCapabilities] = useState<CapabilitiesInfo | null>(null);

  // Active generation tracking
  const [isSubmitting, setIsSubmitting] = useState(false);

  // The shortcuts sheet is the one dialog; the viewer and the system view are pages.
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);
  // A delete waiting for the user's yes (SPEC_UI §4.5): what to ask, and what to run then.
  const [pendingDelete, setPendingDelete] = useState<(ConfirmRequest & { run: () => void }) | null>(null);
  // Stable, so the open dialog keeps its focus while the lists refresh underneath.
  const closeConfirm = useCallback(() => setPendingDelete(null), []);

  // Composer prefill trigger
  const [prefilledPrompt, setPrefilledPrompt] = useState<string | null>(null);
  const [prefilledAspect, setPrefilledAspect] = useState<AspectRatio | null>(null);
  const [composerSource, setComposerSource] = useState<ComposerSource | null>(null);
  // Reference images attached in the composer, in prompt order (docs/REFERENCE_IMAGES.md).
  const [references, setReferences] = useState<ComposerReference[]>([]);
  const maxReferences = capabilities?.max_references_ever ?? 0;

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
  const applyStatus = useCallback(([health, caps]: Awaited<ReturnType<typeof fetchStatus>>) => {
    setWorkerHealth(health);
    if (caps) setCapabilities(caps);
  }, []);
  const refreshStatus = useCallback(() => fetchStatus().then(applyStatus), [applyStatus]);
  useEffect(() => {
    let alive = true;
    const poll = () => fetchStatus().then((status) => alive && applyStatus(status));
    poll();
    const interval = setInterval(poll, 4000);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, [applyStatus]);

  // A change to one job, in every list that has it.
  const { patch: patchTimeline, remove: removeTimeline, apply: applyTimeline } = timeline;
  const { patch: patchGallery, remove: removeGallery, apply: applyGallery } = gallery;
  const updateGeneration = useCallback(
    (id: string, updates: Partial<Generation>) => {
      patchTimeline(id, updates);
      patchGallery(id, updates);
      setLinkedJob((j) => (j?.gen?.id === id ? { ...j, gen: { ...j.gen, ...updates } } : j));
    },
    [patchTimeline, patchGallery],
  );
  const removeGenerations = useCallback(
    (ids: Set<string>) => {
      removeTimeline(ids);
      removeGallery(ids);
    },
    [removeTimeline, removeGallery],
  );

  // The gallery's counts: when it opens, and after changes to finished images.
  const galleryVisited = galleryFilters !== null;
  const refreshFacets = useCallback(async () => {
    const f = await fetchGalleryFacetsApi();
    if (f) setFacets(f);
  }, []);
  useEffect(() => {
    if (!galleryVisited) return;
    let cancelled = false;
    fetchGalleryFacetsApi().then((f) => {
      if (f && !cancelled) setFacets(f);
    });
    return () => {
      cancelled = true;
    };
  }, [galleryVisited]);
  const galleryVisitedRef = useRef(galleryVisited);
  useEffect(() => {
    galleryVisitedRef.current = galleryVisited;
  }, [galleryVisited]);

  // Job sync (ARCHITECTURE_V3.md §8): the lists load page by page; this polls what changed
  // since Studio's version (plus deletions) and patches every list. Every 1 s while
  // anything is active, every 10 s when idle, and on tab focus.
  useEffect(() => {
    let isCancelled = false;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let isPolling = false;

    const pollJobs = async () => {
      if (isPolling || isCancelled) return;
      isPolling = true;
      let waitForFirstPage = false;

      try {
        const version = versionRef.current;
        if (version === undefined) {
          waitForFirstPage = true;
          return;
        }
        const page = await fetchJobChangesApi(version);
        if (isCancelled) return;
        versionRef.current = page.version;
        const deleted = new Set(page.deleted);
        if (page.jobs.length || deleted.size) {
          applyTimeline(page.jobs, deleted);
          applyGallery(page.jobs, deleted);
          setLinkedJob((j) => {
            const fresh = j && page.jobs.find((g) => g.id === j.id);
            return fresh ? { id: fresh.id, gen: fresh } : j;
          });
          if (galleryVisitedRef.current && (deleted.size || page.jobs.some((g) => g.status === 'done'))) {
            refreshFacets();
          }
        }
      } catch {
        // Studio offline or network hiccup; retry next interval
      } finally {
        isPolling = false;
        if (!isCancelled) {
          const hasActive = timelineItemsRef.current.some(
            (g) => g.status === 'queued' || g.status === 'running'
          );
          const intervalMs = waitForFirstPage || hasActive ? 1000 : 10000;
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
  }, [applyTimeline, applyGallery, refreshFacets]);

  // Submit one turn: `count` jobs with consecutive seeds, sent as one batch
  /** Submits one turn: every job the composer built (rows × count, see utils/composer.ts). */
  const handleSubmitTurn = async (prompt: string, jobs: JobParams[], refs: ComposerReference[]) => {
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
        ...(params.enhanceMaxTokens ? { enhance_max_tokens: params.enhanceMaxTokens } : {}),
        ...(params.enhance && params.enhancer ? { enhancer: params.enhancer } : {}),
        ...(refs.length ? { references: referencePayload(refs) } : {}),
        meta: {
          batch_id: batchId,
          aspect: params.aspect,
          size_multiplier: params.sizeMultiplier,
          quality: params.quality,
          index,
        },
      });
    }

    timeline.prepend(newGenerations);
    // The new turn is at the top of the timeline: show it.
    if (route.name !== 'timeline') navigate('/');
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
    // Studio's answer, not 'cancelled' on trust: a job that just finished stays done, and
    // the poll would not send it again.
    const job = await cancelJobApi(jobId);
    if (job) updateGeneration(jobId, job);
    pollJobsRef.current?.();
  };

  const findGeneration = (id: string) =>
    [...timeline.items, ...gallery.items, ...(linkedJob?.gen ? [linkedJob.gen] : [])].find((g) => g.id === id);

  // Toggle Favorite
  const handleToggleFavorite = async (id: string) => {
    const gen = findGeneration(id);
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
    // The image's references come along, in the same order.
    setReferences((prev) => {
      prev.forEach(releasePreview);
      return (gen.references || []).map(fromJobReference);
    });
    const quality = qualityForSteps(gen.params.steps);
    const tiling = !!gen.params.tiling;
    const enhanceMaxTokens = gen.params.enhanceMaxTokens || 0; // 0: the send's default
    setPreferences((prev) => ({
      ...prev,
      // Sizes outside the presets (older images) keep the composer's current aspect.
      aspect: gen.params.aspect in ASPECT_RATIOS ? gen.params.aspect : prev.aspect,
      sizeMultiplier: gen.params.sizeMultiplier || 'full',
      quality,
      steps: gen.params.steps,
      enhance: !!gen.params.enhance,
      // The same rewriter as the image, when it had one.
      ...(gen.params.enhance ? { enhancer: gen.params.enhancer || 'builtin' } : {}),
      tiling,
      enhanceMaxTokens,
      isSeedLocked: !newSeed,
      lockedSeed: newSeed ? prev.lockedSeed : gen.params.seed,
      // A fixed seed, custom steps, tiling and a token limit are advanced options: open them when the
      // image needs one, so they apply. Extra rows are cleared to match this one image.
      advancedOpen:
        prev.advancedOpen || !newSeed || quality === 'custom' || tiling || enhanceMaxTokens > 0,
      extraRows: [],
    }));
    if (route.name === 'image') leavePage();
  };
  const handleRerun = (gen: Generation) => handleToComposer(gen, { newSeed: false });
  const handleVary = (gen: Generation) => handleToComposer(gen, { newSeed: true });

  /** Attach an earlier result as the next reference image (the composer opens to show it). */
  const handleUseAsReference = (gen: Generation) => {
    setReferences((prev) =>
      prev.length >= maxReferences || prev.some((r) => r.image === gen.id) ? prev : [...prev, fromGeneration(gen)],
    );
    if (route.name === 'image') leavePage();
  };
  const canUseAsReference = maxReferences > 0 && references.length < maxReferences;

  // Retry Failed Job (V3: re-queues failed task in place if intermediate artifacts still exist, else re-submits)
  const handleRetry = async (gen: Generation, enhancer?: string) => {
    const result = await retryJobApi(gen.id, enhancer);
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

  const deleteGenerations = async (ids: string[]) => {
    const deleted = new Set<string>();
    for (const id of ids) {
      if (await deleteJobApi(id)) deleted.add(id);
    }
    removeGenerations(deleted);
  };

  // Deleting a finished image is permanent, so it asks first; a failed or cancelled card has
  // no image to lose and goes at once. `after` runs once the delete is under way.
  const requestDelete = (gens: Generation[], what: 'image' | 'turn', after?: () => void) => {
    const run = () => {
      void deleteGenerations(gens.map((g) => g.id));
      after?.();
    };
    const images = gens.filter((g) => g.status === 'done' && g.imageUrl).length;
    if (images === 0) {
      run();
      return;
    }
    setPendingDelete({
      title: what === 'turn' ? 'Delete this turn?' : 'Delete this image?',
      message:
        images > 1
          ? `Its ${images} images are removed for good, with their settings.`
          : 'The image is removed for good, with its settings.',
      confirmLabel: 'Delete',
      run,
    });
  };

  // Delete a single generation card
  const handleDeleteGeneration = (id: string, after?: () => void) => {
    const gen = findGeneration(id);
    if (gen) requestDelete([gen], 'image', after);
  };

  // Delete an entire turn / batch
  const handleDeleteTurn = (batchId: string) => {
    requestDelete(timeline.items.filter((g) => g.batchId === batchId), 'turn');
  };

  // Select Sample Prompt
  const handleSelectSamplePrompt = (prompt: string, aspect?: AspectRatio) => {
    setComposerSource(null);
    setPrefilledPrompt(prompt);
    if (aspect) setPrefilledAspect(aspect);
  };

  const galleryLinkRef = useRef(galleryLink);
  useEffect(() => {
    galleryLinkRef.current = galleryLink;
  }, [galleryLink]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing inside an input or textarea
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 't' || e.key === 'T') {
        if (window.location.pathname !== '/') navigate('/');
      } else if (e.key === 'g' || e.key === 'G') {
        if (window.location.pathname !== '/gallery') navigate(galleryLinkRef.current);
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

  // The viewer (`/images/{id}`): ←/→ follow the list it was opened from, the timeline's order
  // (turns newest first, each left to right) or the gallery's; a shared link steps through
  // the timeline when the image is in it.
  const viewerId = route.name === 'image' ? route.id : null;
  const from = viewerId ? viewerFrom() : undefined;
  const timelineImages = useMemo(
    () => groupTurns(timeline.items).flat().filter((g) => g.status === 'done' && g.imageUrl),
    [timeline.items],
  );
  const viewerList = from === 'gallery' ? gallery : timeline;
  const viewerImages = from === 'gallery' ? gallery.items : timelineImages;
  const viewerIndex = viewerId ? viewerImages.findIndex((g) => g.id === viewerId) : -1;
  const viewerGen: Generation | null = viewerId
    ? viewerImages[viewerIndex] ??
      timeline.items.find((g) => g.id === viewerId) ??
      gallery.items.find((g) => g.id === viewerId) ??
      (linkedJob?.id === viewerId ? linkedJob.gen : null)
    : null;
  const viewerMissing = viewerId !== null && linkedJob?.id === viewerId && linkedJob.gen === null && !viewerGen;

  // An image outside the loaded lists (a shared link, an old image): fetch it alone.
  const viewerKnown = !!viewerGen;
  useEffect(() => {
    if (!viewerId || viewerKnown) return;
    let cancelled = false;
    fetchJobApi(viewerId)
      .then((gen) => !cancelled && setLinkedJob({ id: viewerId, gen }))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [viewerId, viewerKnown]);

  // Near the end of the loaded images, the next page comes in, so → keeps going.
  const { hasMore: viewerHasMore, loadMore: viewerLoadMore } = viewerList;
  useEffect(() => {
    if (viewerIndex >= 0 && viewerIndex >= viewerImages.length - 3 && viewerHasMore) viewerLoadMore();
  }, [viewerIndex, viewerImages.length, viewerHasMore, viewerLoadMore]);

  const stepViewer = (delta: number) => {
    const next = viewerImages[viewerIndex + delta];
    if (next) navigate(imageHref(next.id), { replace: true, from });
  };

  // Back/Forward: the list comes back where it was left, once its content is there.
  const listReady =
    route.name === 'timeline' ? timeline.loaded : route.name === 'gallery' ? gallery.loaded : true;
  useLayoutEffect(() => {
    if (listReady) restoreScroll();
  }, [route, listReady]);

  // The tab's title names the page (and the image, for shared links).
  useEffect(() => {
    const base = 'Aries Image Studio';
    const title =
      route.name === 'gallery'
        ? `Gallery · ${base}`
        : route.name === 'status'
        ? `System · ${base}`
        : route.name === 'image' && viewerGen
        ? `${viewerGen.prompt.slice(0, 60)}${viewerGen.prompt.length > 60 ? '…' : ''} · ${base}`
        : base;
    document.title = title;
  }, [route, viewerGen]);

  const openViewer = useCallback((gen: Generation) => navigate(imageHref(gen.id), { from: 'timeline' }), []);
  const onGalleryFilters = useCallback(
    (f: GalleryFilters, replace?: boolean) => navigate(galleryHref(f), { replace }),
    [],
  );

  // Keyed: the viewer's layout puts it at another position under the root div, and without
  // the key React would remount it on the way back, losing what Rerun/Vary filled in.
  const composer = (
    <Composer
      key="composer"
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
      references={references}
      onReferencesChange={setReferences}
      maxReferences={maxReferences}
      onClearPrefill={() => {
        setPrefilledPrompt(null);
        setPrefilledAspect(null);
      }}
      hidden={route.name === 'image' || route.name === 'status'}
    />
  );

  const shortcuts = (
    <>
      <ShortcutsModal isOpen={isShortcutsOpen} onClose={() => setIsShortcutsOpen(false)} />
      <ConfirmDialog
        request={pendingDelete}
        onConfirm={() => {
          pendingDelete?.run();
          setPendingDelete(null);
        }}
        onCancel={closeConfirm}
      />
    </>
  );

  // The viewer fills the window on its own: no header, no list underneath.
  if (route.name === 'image') {
    return (
      <div className="min-h-dvh bg-[#0b0b0a] text-white font-['Inter'] antialiased">
        {viewerGen?.imageUrl ? (
          <Viewer
            generation={viewerGen}
            onClose={() => leavePage()}
            onPrev={() => stepViewer(-1)}
            onNext={() => stepViewer(1)}
            hasPrev={viewerIndex > 0}
            hasNext={viewerIndex >= 0 && viewerIndex < viewerImages.length - 1}
            onToggleFavorite={handleToggleFavorite}
            onVary={handleVary}
            onRerun={handleRerun}
            onUseAsReference={canUseAsReference ? handleUseAsReference : undefined}
            // After the delete the viewer has nothing left to show: back to the list.
            onDelete={(id) => handleDeleteGeneration(id, () => leavePage())}
          />
        ) : viewerMissing || (viewerGen && !viewerGen.imageUrl) ? (
          <div className="min-h-dvh flex flex-col items-center justify-center gap-4 p-8 text-center">
            <p className="text-sm text-white/80">
              {viewerGen ? 'This image is not finished yet.' : 'This image does not exist (any more).'}
            </p>
            <button
              onClick={() => leavePage()}
              className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90"
            >
              Back to the timeline
            </button>
          </div>
        ) : (
          <div className="min-h-dvh flex">
            <PageLoader />
          </div>
        )}
        {composer}
        {shortcuts}
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#fcf9f2] dark:bg-[#121210] text-[#1c1c18] dark:text-[#f3f0e9] font-['Inter'] antialiased transition-colors duration-200">
      {/* Top Navigation Header */}
      <Header
        activeView={route.name === 'timeline' || route.name === 'gallery' ? route.name : null}
        galleryHref={galleryLink}
        workerHealth={workerHealth}
        theme={preferences.theme}
        onToggleTheme={() =>
          setPreferences((p) => ({ ...p, theme: p.theme === 'light' ? 'dark' : 'light' }))
        }
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* Main Content Area */}
      <main className={`flex-1 flex flex-col ${route.name === 'status' ? 'pb-8' : 'pb-24'}`}>
        <ErrorBoundary fallbackTitle="Page rendering error">
          {route.name === 'status' ? (
            <StatusPage onClose={() => leavePage()} health={workerHealth} onRefresh={refreshStatus} />
          ) : route.name === 'timeline' ? (
            <Timeline
              generations={timeline.items}
              onOpenLightbox={openViewer}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onRetry={handleRetry}
              onRerun={handleRerun}
              onCancel={handleCancelJob}
              onSelectPrompt={handleSelectSamplePrompt}
              onDelete={handleDeleteGeneration}
              onDeleteTurn={handleDeleteTurn}
              onUseAsReference={canUseAsReference ? handleUseAsReference : undefined}
              loading={!timeline.loaded}
              hasMore={timeline.hasMore}
              loadingMore={timeline.loadingMore}
              onLoadMore={timeline.loadMore}
            />
          ) : (
            <Gallery
              generations={gallery.items}
              filters={route.filters}
              onFiltersChange={onGalleryFilters}
              facets={facets}
              onToggleFavorite={handleToggleFavorite}
              onVary={handleVary}
              onDelete={handleDeleteGeneration}
              loading={!gallery.loaded}
              hasMore={gallery.hasMore}
              loadingMore={gallery.loadingMore}
              onLoadMore={gallery.loadMore}
            />
          )}
        </ErrorBoundary>
      </main>

      {composer}
      {shortcuts}
    </div>
  );
}

export default App;
