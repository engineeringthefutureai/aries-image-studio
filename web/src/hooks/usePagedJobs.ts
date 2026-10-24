import { useCallback, useEffect, useRef, useState } from 'react';
import type { Generation } from '../types';
import { fetchJobsPageApi, type JobQuery } from '../services/api';

const PAGE_SIZE = 40;

/** Newest first; jobs created in the same instant by id, as Studio pages them. */
const newestFirst = (a: Generation, b: Generation) => b.createdAt - a.createdAt || b.id.localeCompare(a.id);

/** Whether a job belongs in a list with this query (the client side of Studio's filters). */
export function matchesQuery(g: Generation, q: JobQuery): boolean {
  if (q.doneOnly && !(g.status === 'done' && g.imageUrl)) return false;
  if (q.favorites && !g.favorite) return false;
  if (q.ratio) {
    const [w, h] = q.ratio.split(':').map(Number);
    if (!(w > 0 && h > 0) || Math.abs(g.params.width / g.params.height - w / h) >= 0.01) return false;
  }
  const text = q.q?.trim().toLowerCase();
  if (text) {
    const hit =
      g.prompt.toLowerCase().includes(text) ||
      !!g.enhancedPrompt?.toLowerCase().includes(text) ||
      String(g.params.seed).includes(text);
    if (!hit) return false;
  }
  return true;
}

export interface PagedJobs {
  /** The loaded pages, newest first. */
  items: Generation[];
  /** False until the first page has arrived. */
  loaded: boolean;
  hasMore: boolean;
  loadingMore: boolean;
  /** Load the next page (no-op while one loads or when there is none). */
  loadMore: () => void;
  /**
   * Jobs that changed or were deleted (the poll loop): updated in place, dropped when they
   * no longer match, added when they match and fall inside the loaded pages (older ones
   * arrive with their page).
   */
  apply: (changed: Generation[], deleted: Set<string>) => void;
  /** Jobs just submitted from this tab, before Studio lists them. */
  prepend: (gens: Generation[]) => void;
  patch: (id: string, updates: Partial<Generation>) => void;
  remove: (ids: Set<string>) => void;
}

interface State {
  key: string;
  items: Generation[];
  next: string | null;
  loaded: boolean;
  loadingMore: boolean;
}

/**
 * A list of Studio's jobs loaded a page at a time (infinite scroll). `query` null: nothing
 * loads (a page not visited yet). `onFirstPage` gets Studio's version with each first page,
 * where the poll loop starts.
 */
export function usePagedJobs(query: JobQuery | null, onFirstPage?: (version: number) => void): PagedJobs {
  const key = query ? JSON.stringify(query) : '';
  const [state, setState] = useState<State>({ key, items: [], next: null, loaded: false, loadingMore: false });
  const stateRef = useRef(state);
  stateRef.current = state;
  const queryRef = useRef(query);
  queryRef.current = query;
  const onFirstPageRef = useRef(onFirstPage);
  onFirstPageRef.current = onFirstPage;

  // A new query starts over from its first page; responses for an older query are dropped.
  useEffect(() => {
    if (!query) return;
    let cancelled = false;
    setState({ key, items: [], next: null, loaded: false, loadingMore: true });
    const load = async () => {
      try {
        const page = await fetchJobsPageApi(query, PAGE_SIZE);
        if (cancelled) return;
        onFirstPageRef.current?.(page.version);
        setState((s) => {
          // Keep jobs submitted from this tab meanwhile that Studio did not list yet.
          const listed = new Set(page.jobs.map((g) => g.id));
          const local = s.key === key ? s.items.filter((g) => !listed.has(g.id)) : [];
          return { key, items: [...local, ...page.jobs].sort(newestFirst), next: page.next, loaded: true, loadingMore: false };
        });
      } catch {
        if (!cancelled) setTimeout(load, 3000); // Studio unreachable: retry
      }
    };
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` stands for `query`
  }, [key]);

  const loadMore = useCallback(() => {
    const s = stateRef.current;
    const q = queryRef.current;
    if (!q || !s.loaded || s.loadingMore || !s.next) return;
    const startKey = s.key;
    setState((prev) => ({ ...prev, loadingMore: true }));
    fetchJobsPageApi(q, PAGE_SIZE, s.next)
      .then((page) => {
        setState((prev) => {
          if (prev.key !== startKey) return prev;
          const have = new Set(prev.items.map((g) => g.id));
          const items = [...prev.items, ...page.jobs.filter((g) => !have.has(g.id))].sort(newestFirst);
          return { ...prev, items, next: page.next, loadingMore: false };
        });
      })
      .catch(() => setState((prev) => (prev.key === startKey ? { ...prev, loadingMore: false } : prev)));
  }, []);

  const apply = useCallback((changed: Generation[], deleted: Set<string>) => {
    const q = queryRef.current;
    if (!q) return;
    setState((s) => {
      if (!s.loaded) return s; // the first page brings them
      let items = deleted.size ? s.items.filter((g) => !deleted.has(g.id)) : s.items;
      if (changed.length === 0) return items === s.items ? s : { ...s, items };
      items = [...items];
      // With more pages to come, only jobs inside the loaded ones; older ones come with
      // their page (adding them now would leave a gap above them).
      const oldest = s.next && items.length ? items[items.length - 1] : null;
      let sort = false;
      for (const g of changed) {
        const at = items.findIndex((x) => x.id === g.id);
        if (!matchesQuery(g, q)) {
          if (at >= 0) items.splice(at, 1);
        } else if (at >= 0) {
          items[at] = g;
        } else if (!oldest || newestFirst(g, oldest) <= 0) {
          items.push(g);
          sort = true;
        }
      }
      if (sort) items.sort(newestFirst);
      return { ...s, items };
    });
  }, []);

  const prepend = useCallback((gens: Generation[]) => {
    setState((s) => ({ ...s, items: [...gens, ...s.items].sort(newestFirst) }));
  }, []);

  const patch = useCallback((id: string, updates: Partial<Generation>) => {
    setState((s) =>
      s.items.some((g) => g.id === id)
        ? { ...s, items: s.items.map((g) => (g.id === id ? { ...g, ...updates } : g)) }
        : s,
    );
  }, []);

  const remove = useCallback((ids: Set<string>) => {
    setState((s) => ({ ...s, items: s.items.filter((g) => !ids.has(g.id)) }));
  }, []);

  const current = state.key === key;
  return {
    items: current ? state.items : [],
    loaded: current && state.loaded,
    hasMore: current && !!state.next,
    loadingMore: current && state.loadingMore,
    loadMore,
    apply,
    prepend,
    patch,
    remove,
  };
}
