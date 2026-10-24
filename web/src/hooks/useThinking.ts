import { useEffect, useState } from 'react';
import type { Generation } from '../types';
import { fetchJobApi } from '../services/api';

// Reasoning never changes once written: fetched once per job and tab.
const cache = new Map<string, Promise<string | undefined>>();

/**
 * The enhancer's reasoning for `gen`, fetched from Studio when `open` (job lists leave it out:
 * it is most of their size). Undefined while it loads.
 */
export function useThinking(gen: Generation | null | undefined, open: boolean): string | undefined {
  const [text, setText] = useState<{ id: string; value?: string } | null>(null);
  const id = gen?.id;
  const inline = gen?.thinking;
  const wanted = open && !!gen?.hasThinking && !inline;

  useEffect(() => {
    if (!wanted || !id) return;
    let cancelled = false;
    let request = cache.get(id);
    if (!request) {
      request = fetchJobApi(id).then((g) => g?.thinking);
      request.catch(() => cache.delete(id));
      cache.set(id, request);
    }
    request.then(
      (value) => !cancelled && setText({ id, value }),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [id, wanted]);

  if (inline) return inline;
  return text && text.id === id ? text.value : undefined;
}
