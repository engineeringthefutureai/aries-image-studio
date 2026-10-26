/**
 * Who is signed in (docs/AUTH.md). Studio answers `GET /api/me` with the account, with
 * `{auth: false}` when sign-in is off, or 401 when signed out.
 */

export interface Account {
  email: string;
  name: string;
  picture: string;
  /** The owner may lend, disable and enable cards. */
  owner: boolean;
}

/** `open`: sign-in is off, everyone sees everything. */
export type SessionState =
  | { kind: 'loading' }
  | { kind: 'open' }
  | { kind: 'signed-out' }
  | { kind: 'signed-in'; account: Account };

export const SIGNED_OUT_EVENT = 'aries:signed-out';

export async function fetchSession(): Promise<SessionState> {
  try {
    const res = await fetch('/api/me', { cache: 'no-store' });
    if (res.status === 401) return { kind: 'signed-out' };
    if (!res.ok) return { kind: 'open' };
    const me = await res.json();
    if (!me.auth) return { kind: 'open' };
    return {
      kind: 'signed-in',
      account: { email: me.email, name: me.name ?? '', picture: me.picture ?? '', owner: !!me.owner },
    };
  } catch {
    // Studio unreachable: the app shows it as offline.
    return { kind: 'open' };
  }
}

/**
 * Any API answer of 401 (the session expired or was signed out elsewhere) raises
 * SIGNED_OUT_EVENT, so the app returns to the sign-in screen instead of failing call by
 * call. Installed once, before the app renders.
 */
export function watchForSignOut() {
  const original = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const res = await original(input, init);
    if (res.status === 401) {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      const path = new URL(url, window.location.origin).pathname;
      if (path.startsWith('/api/') && path !== '/api/me') {
        window.dispatchEvent(new Event(SIGNED_OUT_EVENT));
      }
    }
    return res;
  };
}

export function signInHref(): string {
  const here = window.location.pathname + window.location.search;
  return `/auth/login?next=${encodeURIComponent(here)}`;
}

export async function signOut() {
  try {
    await fetch('/auth/logout', { method: 'POST' });
  } finally {
    window.location.assign('/');
  }
}
