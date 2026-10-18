import type { ComposerReference, Generation, JobReference } from '../types';
import { generateUlid } from './ulid';

/** Image types Studio accepts as references (`POST /api/uploads`). */
export const REFERENCE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

export const uploadThumbUrl = (sha256: string, size = 512) =>
  `/api/uploads/${encodeURIComponent(sha256)}/thumb?size=${size}`;

/** A job's stored reference, back in the composer (Rerun, Vary). */
export const fromJobReference = (ref: JobReference): ComposerReference => ({
  key: generateUlid(),
  previewUrl: uploadThumbUrl(ref.sha256),
  status: 'ready',
  upload: ref.sha256,
});

/** An earlier result attached as a reference ("Use as reference"). */
export const fromGeneration = (gen: Generation): ComposerReference => ({
  key: generateUlid(),
  previewUrl: gen.thumbUrl || gen.imageUrl || '',
  status: 'ready',
  image: gen.id,
});

/** What `POST /api/jobs` takes for each attached reference, in order. */
export const referencePayload = (refs: ComposerReference[]) =>
  refs.map((r) => (r.upload ? { upload: r.upload } : { image: r.image as string }));

/** Moves the reference at `index` one place earlier ("image 2" becomes "image 1"). */
export function moveEarlier<T>(list: T[], index: number): T[] {
  if (index <= 0 || index >= list.length) return list;
  const next = [...list];
  [next[index - 1], next[index]] = [next[index], next[index - 1]];
  return next;
}

/** Frees the browser's preview of a local file once its reference is gone. */
export function releasePreview(ref: ComposerReference) {
  if (ref.previewUrl.startsWith('blob:')) URL.revokeObjectURL(ref.previewUrl);
}
