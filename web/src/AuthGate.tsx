import React, { useEffect, useState } from 'react';
import App from './App';
import { SignIn } from './components/SignIn';
import { fetchSession, openDemo, SIGNED_OUT_EVENT, type SessionState } from './services/session';

/** The app when signed in (or when sign-in is off), read-only for a demo guest; the sign-in
 * screen otherwise, which is also where a guest's visit starts. */
export const AuthGate: React.FC = () => {
  const [session, setSession] = useState<SessionState>({ kind: 'loading' });

  useEffect(() => {
    let live = true;
    // Studio unreachable at the start: the app shows it as offline.
    fetchSession().then((s) => live && setSession(s ?? { kind: 'open' }));
    // The session or the guest's id ended: Studio says what that leaves. Until it can say, the
    // page stays as it is, rather than guessing.
    let checking = false;
    const signedOut = async () => {
      if (checking) return;
      checking = true;
      let session = await fetchSession();
      while (live && !session) {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        session = await fetchSession();
      }
      checking = false;
      if (live && session) setSession(session);
    };
    window.addEventListener(SIGNED_OUT_EVENT, signedOut);
    return () => {
      live = false;
      window.removeEventListener(SIGNED_OUT_EVENT, signedOut);
    };
  }, []);

  if (session.kind === 'loading') return null;
  if (session.kind === 'signed-out') {
    const enterDemo = async () => {
      const error = await openDemo();
      if (error) return error;
      const now = await fetchSession();
      if (!now) return "The demo didn't open. Try again.";
      // Accepted, yet Studio doesn't know this browser: its cookie was not kept.
      if (now.kind === 'signed-out') return 'The demo needs a cookie. Allow cookies for this site and try again.';
      setSession(now);
      return null;
    };
    return <SignIn demo={session.demo} onEnterDemo={enterDemo} />;
  }
  // A new app for a new kind of session: nothing of the account's stays on the page.
  return (
    <App
      key={session.kind}
      account={session.kind === 'signed-in' ? session.account : null}
      visitor={session.kind === 'visitor'}
    />
  );
};
