import React, { useEffect, useState } from 'react';
import App from './App';
import { SignIn } from './components/SignIn';
import { fetchSession, SIGNED_OUT_EVENT, type SessionState } from './services/session';

/** The app when signed in (or when sign-in is off), the sign-in screen otherwise. */
export const AuthGate: React.FC = () => {
  const [session, setSession] = useState<SessionState>({ kind: 'loading' });

  useEffect(() => {
    let live = true;
    fetchSession().then((s) => live && setSession(s));
    const signedOut = () => setSession({ kind: 'signed-out' });
    window.addEventListener(SIGNED_OUT_EVENT, signedOut);
    return () => {
      live = false;
      window.removeEventListener(SIGNED_OUT_EVENT, signedOut);
    };
  }, []);

  if (session.kind === 'loading') return null;
  if (session.kind === 'signed-out') return <SignIn />;
  return <App account={session.kind === 'signed-in' ? session.account : null} />;
};
