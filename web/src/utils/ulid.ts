const CROCKFORD_BASE32 = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

/**
 * ULID: 26 Crockford Base32 characters, a 48-bit millisecond timestamp followed by
 * 80 random bits. Job ids are generated here so resubmitting a job is idempotent.
 */
export function generateUlid(now: number = Date.now()): string {
  let time = '';
  let t = now;
  for (let i = 0; i < 10; i++) {
    time = CROCKFORD_BASE32[t % 32] + time;
    t = Math.floor(t / 32);
  }

  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let random = '';
  for (const byte of bytes) {
    random += CROCKFORD_BASE32[byte % 32];
  }
  return time + random;
}
