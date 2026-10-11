import type { Generation, UserPreferences } from '../types';

const GENERATIONS_KEY = 'aries_studio_generations_v1';
const PREFS_KEY = 'aries_studio_prefs_v1';

export const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'light',
  aspect: '1:1',
  sizeMultiplier: 'full',
  quality: 'standard',
  steps: 40,
  count: 1,
  isSeedLocked: false,
  lockedSeed: 42,
  enhance: false,
};

/** Blob URLs die with the page; images are restored from IndexedDB on load instead. */
function withoutBlobUrl(item: Generation): Generation {
  if (item?.imageUrl?.startsWith('blob:')) {
    return { ...item, imageUrl: undefined };
  }
  return item;
}

export function loadGenerations(): Generation[] {
  try {
    const raw = localStorage.getItem(GENERATIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(withoutBlobUrl);
  } catch (err) {
    console.error('Failed to load generations from storage:', err);
    return [];
  }
}

export function saveGenerations(items: Generation[]): void {
  try {
    const sanitized = items.map(withoutBlobUrl);
    localStorage.setItem(GENERATIONS_KEY, JSON.stringify(sanitized));
  } catch (err) {
    console.error('Failed to persist generations to storage:', err);
  }
}

export function loadPreferences(): UserPreferences {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return DEFAULT_PREFERENCES;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_PREFERENCES, ...parsed };
  } catch (err) {
    console.error('Failed to load preferences from storage:', err);
    return DEFAULT_PREFERENCES;
  }
}

export function savePreferences(prefs: UserPreferences): void {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch (err) {
    console.error('Failed to persist preferences to storage:', err);
  }
}

// --- IndexedDB Binary Image Storage ---
// Stores full-resolution generated PNG blobs locally so images persist across refreshes
// without quota limits or reliance on ephemeral worker backend cache.

const IDB_NAME = 'aries_studio_db';
const IDB_STORE = 'images';

function openImageDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      return reject(new Error('IndexedDB not supported'));
    }
    const request = indexedDB.open(IDB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function storeImageBlob(jobId: string, blob: Blob): Promise<void> {
  try {
    const db = await openImageDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      const store = tx.objectStore(IDB_STORE);
      const req = store.put(blob, jobId);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn('[Storage] Failed to store image blob in IndexedDB:', err);
  }
}

export async function getImageBlob(jobId: string): Promise<Blob | null> {
  try {
    const db = await openImageDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const store = tx.objectStore(IDB_STORE);
      const req = store.get(jobId);
      req.onsuccess = () => resolve(req.result instanceof Blob ? req.result : null);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn('[Storage] Failed to get image blob from IndexedDB:', err);
    return null;
  }
}

export async function deleteImageBlob(jobId: string): Promise<void> {
  try {
    const db = await openImageDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      const store = tx.objectStore(IDB_STORE);
      const req = store.delete(jobId);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn('[Storage] Failed to delete image blob from IndexedDB:', err);
  }
}

export async function restoreGenerationImageUrls(items: Generation[]): Promise<Generation[]> {
  try {
    const db = await openImageDb();
    const hydrated = await Promise.all(
      items.map(async (gen) => {
        if (gen.status !== 'done') return gen;
        return new Promise<Generation>((resolve) => {
          const tx = db.transaction(IDB_STORE, 'readonly');
          const store = tx.objectStore(IDB_STORE);
          const req = store.get(gen.id);
          req.onsuccess = () => {
            const blob = req.result;
            if (blob instanceof Blob) {
              resolve({ ...gen, imageUrl: URL.createObjectURL(blob) });
            } else {
              resolve(gen);
            }
          };
          req.onerror = () => resolve(gen);
        });
      })
    );
    return hydrated;
  } catch (err) {
    console.warn('[Storage] Failed to restore image URLs from IndexedDB:', err);
    return items;
  }
}

