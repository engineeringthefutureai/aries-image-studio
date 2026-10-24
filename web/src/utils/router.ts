import { useSyncExternalStore } from 'react';

/**
 * The app's pages, one per URL (docs/SPEC_UI.md §Routes). Every page is a page of its own,
 * never a layer over another: the viewer and the status panel replace the list instead of
 * covering it, so scrolling one can never scroll the other.
 */
export type Route =
  | { name: 'timeline' }
  | { name: 'gallery'; filters: GalleryFilters }
  | { name: 'image'; id: string }
  | { name: 'status' };

export interface GalleryFilters {
  favorites: boolean;
  /** Aspect label (`16:9`, or `w:h` for sizes outside the presets); empty: all. */
  aspect: string;
  q: string;
}

/** Where the viewer was opened from: its ←/→ follow that list, and closing goes back to it. */
export type ViewerFrom = 'timeline' | 'gallery';

interface HistoryState {
  key: string;
  from?: ViewerFrom;
  /** Reached by a link inside the app: Back stays in it (closing a page goes back). */
  inApp?: boolean;
}

export function parseRoute(path: string, search: string): Route {
  const image = path.match(/^\/images\/([^/]+)\/?$/);
  if (image) return { name: 'image', id: decodeURIComponent(image[1]) };
  if (/^\/gallery\/?$/.test(path)) {
    const p = new URLSearchParams(search);
    return {
      name: 'gallery',
      filters: { favorites: p.get('fav') === '1', aspect: p.get('aspect') || '', q: p.get('q') || '' },
    };
  }
  if (/^\/status\/?$/.test(path)) return { name: 'status' };
  return { name: 'timeline' };
}

export function galleryHref(f: GalleryFilters): string {
  const p = new URLSearchParams();
  if (f.favorites) p.set('fav', '1');
  if (f.aspect) p.set('aspect', f.aspect);
  if (f.q.trim()) p.set('q', f.q.trim());
  const qs = p.toString();
  return qs ? `/gallery?${qs}` : '/gallery';
}

export const imageHref = (id: string) => `/images/${encodeURIComponent(id)}`;

// Scroll positions by history entry, so Back lands where the list was left. In memory only:
// the app keeps nothing in browser storage.
const scrollByKey = new Map<string, number>();
let counter = 0;
const newKey = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`;

function currentState(): HistoryState | null {
  const s = window.history.state as HistoryState | null;
  return s && typeof s.key === 'string' ? s : null;
}

if (typeof window !== 'undefined') {
  window.history.scrollRestoration = 'manual';
  if (!currentState()) window.history.replaceState({ key: newKey() } satisfies HistoryState, '');
}

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

if (typeof window !== 'undefined') {
  window.addEventListener('popstate', () => {
    pendingRestore = true;
    notify();
  });
}

/** Set by Back/Forward: the page restores its saved scroll position once it has rendered. */
let pendingRestore = false;

/**
 * Go to `href`. `replace` swaps the current entry (stepping through images in the viewer,
 * typing in the gallery search) instead of adding one.
 */
export function navigate(href: string, opts: { replace?: boolean; from?: ViewerFrom } = {}): void {
  const here = currentState();
  const from = opts.from ? { from: opts.from } : {};
  if (opts.replace) {
    // The same history entry: it keeps its key (and saved scroll) and how it was reached.
    const state: HistoryState = { key: here?.key ?? newKey(), inApp: here?.inApp, ...from };
    window.history.replaceState(state, '', href);
  } else {
    if (here) scrollByKey.set(here.key, window.scrollY);
    window.history.pushState({ key: newKey(), inApp: true, ...from } satisfies HistoryState, '', href);
    window.scrollTo(0, 0);
  }
  notify();
}

/**
 * Leave a page such as the viewer: Back when it was reached from inside the app (the list
 * comes back where it was left), else to `fallback` (a shared link opened in a new tab).
 */
export function leavePage(fallback = '/'): void {
  if (currentState()?.inApp) window.history.back();
  else navigate(fallback, { replace: true });
}

/** The viewer's list, when it was opened from one in this tab. */
export const viewerFrom = (): ViewerFrom | undefined => currentState()?.from;

/**
 * After a Back/Forward, scrolls to where the page was left (call once the page has its
 * content). Returns whether there was a position to restore.
 */
export function restoreScroll(): boolean {
  if (!pendingRestore) return false;
  pendingRestore = false;
  const y = scrollByKey.get(currentState()?.key ?? '');
  if (y === undefined) return false;
  window.scrollTo(0, y);
  return true;
}

let snapshot = { path: '', search: '', route: { name: 'timeline' } as Route };
function getSnapshot() {
  const { pathname, search } = window.location;
  if (pathname !== snapshot.path || search !== snapshot.search) {
    snapshot = { path: pathname, search, route: parseRoute(pathname, search) };
  }
  return snapshot.route;
}

export function useRoute(): Route {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getSnapshot,
  );
}

/**
 * An in-app link: a real `href` (copy, open in a new tab, middle-click), navigated without a
 * page load on a plain click.
 */
export function onLinkClick(e: React.MouseEvent<HTMLElement>, href: string, opts?: { from?: ViewerFrom }) {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  navigate(href, opts);
}
