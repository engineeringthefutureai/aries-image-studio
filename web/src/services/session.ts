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
  /** This Studio has a read-only demo for visitors; the owner chooses what it shows. */
  demoMode: boolean;
}

/** `open`: sign-in is off, everyone sees everything. `signed-out`: the sign-in screen, which offers
 * the read-only demo (docs/DEMO.md) when this Studio has one. `visitor`: in the demo. */
export type SessionState =
  | { kind: 'loading' }
  | { kind: 'open' }
  | { kind: 'signed-out'; demo: boolean }
  | { kind: 'visitor' }
  | { kind: 'signed-in'; account: Account };

/** Why the last sign-in came back without a session (`?auth_error=` from Studio). */
export const SIGN_IN_ERRORS: Record<string, string> = {
  not_allowed: "That Google account doesn't have access to this Studio.",
  expired: 'The sign-in took too long. Try again.',
  cancelled: 'Sign-in was cancelled.',
  failed: "Sign-in didn't work. Try again.",
};

export const SIGNED_OUT_EVENT = 'aries:signed-out';

/** Studio last said this is a visitor: a visitor's answer is then no news. */
let isVisitor = false;

/** Who this is, as Studio says; null when it could not say (unreachable, or failing). */
export async function fetchSession(): Promise<SessionState | null> {
  const session = await askStudio();
  if (session) isVisitor = session.kind === 'visitor';
  return session;
}

async function askStudio(): Promise<SessionState | null> {
  try {
    const res = await fetch('/api/me', { cache: 'no-store' });
    if (res.status === 401) {
      const body = await res.json().catch(() => null);
      return { kind: 'signed-out', demo: !!body?.demo_mode };
    }
    if (!res.ok) return null;
    const me = await res.json();
    if (!me.auth) return { kind: 'open' };
    if (me.visitor) return { kind: 'visitor' };
    return {
      kind: 'signed-in',
      account: { email: me.email, name: me.name ?? '', picture: me.picture ?? '', owner: !!me.owner, demoMode: !!me.demo_mode },
    };
  } catch {
    return null;
  }
}

/**
 * Raises SIGNED_OUT_EVENT when an API answer shows this page is out of date, so the app asks Studio
 * who this is now instead of failing call by call: a 401 (a session or guest id ended), a visitor's
 * answer to a page that thinks it is signed in, or an account's answer to a visitor's page (signed in
 * from another tab: the account's images must not land in the demo's page). Installed once, before
 * the app renders.
 */
export function watchForSignOut() {
  const original = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const res = await original(input, init);
    const fromVisitor = res.headers.has('x-aries-visitor');
    const stale = res.status === 401 || (isVisitor ? res.ok && !fromVisitor : fromVisitor);
    if (stale) {
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

/** Accepts the terms and opens the demo with a new id; what went wrong, if it did not open. */
export async function openDemo(): Promise<string | null> {
  try {
    const res = await fetch('/auth/guest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accept_terms: true }),
    });
    if (res.ok) return null;
    if (res.status === 429) return 'Too many tries from your address. Try again in a minute.';
  } catch {
    // The same as a refusal: nothing changed, and the button is still there.
  }
  return "The demo didn't open. Try again.";
}

/** Signs out, or leaves the demo: both come back to the sign-in screen. */
export async function signOut() {
  try {
    await fetch('/auth/logout', { method: 'POST' });
  } finally {
    window.location.assign('/');
  }
}
