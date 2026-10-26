import React, { useEffect, useState } from 'react';
import { Logo } from './Logo';
import { signInHref } from '../services/session';

/** Why the last sign-in came back without a session (`?auth_error=` from Studio). */
const ERRORS: Record<string, string> = {
  not_allowed: "That Google account doesn't have access to this Studio.",
  expired: 'The sign-in took too long. Try again.',
  cancelled: 'Sign-in was cancelled.',
  failed: "Sign-in didn't work. Try again.",
};

/** Shown instead of the app while signed out. Google is the only way in. */
export const SignIn: React.FC = () => {
  const [error] = useState(() => {
    const code = new URLSearchParams(window.location.search).get('auth_error');
    return code ? (ERRORS[code] ?? ERRORS.failed) : null;
  });

  useEffect(() => {
    // The page has no preferences yet: follow the system theme.
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.classList.toggle('dark', dark);
    // Drop ?auth_error= so a retry starts from a clean address.
    const url = new URL(window.location.href);
    if (url.searchParams.has('auth_error')) {
      url.searchParams.delete('auth_error');
      window.history.replaceState(null, '', url.pathname + url.search);
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-[#fcf9f2] dark:bg-[#121210] text-[#1c1c18] dark:text-[#fcf9f2]">
      <div className="w-full max-w-sm mt-8 rounded-2xl border border-[#e5e2db] dark:border-[#272724] bg-white/60 dark:bg-[#1c1c19] p-8 shadow-sm text-center">
        <Logo className="w-14 h-14 mx-auto drop-shadow-sm" />
        <h1 className="mt-4 font-['Plus_Jakarta_Sans'] font-bold text-xl tracking-tight">Aries Image Studio</h1>
        <p className="mt-1 text-sm text-[#58423a] dark:text-[#a09088]">Sign in to see your images.</p>
        {error && (
          <p
            role="alert"
            className="mt-5 rounded-lg px-3 py-2 text-sm bg-rose-500/10 text-rose-800 dark:text-rose-300 border border-rose-500/20"
          >
            {error}
          </p>
        )}
        <a
          href={signInHref()}
          className="mt-6 inline-flex w-full items-center justify-center gap-2.5 rounded-full px-4 py-2.5 text-sm font-medium bg-[#a53705] text-white hover:bg-[#8f2f04] transition-colors shadow-sm"
        >
          <GoogleMark />
          Sign in with Google
        </a>
        <p className="mt-5 text-xs text-[#8c7169] dark:text-[#a09088]">
          By signing in you agree to the{' '}
          <a href="/terms" className="underline hover:text-[#a53705]">terms</a> and the{' '}
          <a href="/privacy" className="underline hover:text-[#a53705]">privacy policy</a>.
        </p>
      </div>
      <About />
    </div>
  );
};

/** What this is, under the card, for whoever lands here first; visible without a click. */
const About: React.FC = () => (
  <p className="w-full max-w-md mt-6 mb-8 text-center text-[11px] leading-relaxed text-[#6b554d] dark:text-[#a09088]">
    Aries Image Studio is a private, self-hosted image generator, by invitation only. Images are
    made by open models on the host's own computers. Google sign-in only checks that you are
    invited; no Google API is used to make or edit images.
  </p>
);

/** Google's "G", on a white disc so it reads on the accent button. */
const GoogleMark: React.FC = () => (
  <span className="inline-flex w-5 h-5 items-center justify-center rounded-full bg-white">
    <svg viewBox="0 0 48 48" className="w-3.5 h-3.5" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.6 5.4 2.7 13.3l7.9 6.1C12.5 13.6 17.8 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.4c-.5 2.9-2.2 5.3-4.6 6.9l7.3 5.7c4.3-3.9 7-9.8 7-17.1z" />
      <path fill="#FBBC05" d="M10.6 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-7.9-6.1C1 16.6 0 20.2 0 24s1 7.4 2.7 10.7l7.9-6.1z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.3-5.7c-2 1.4-4.7 2.3-8.6 2.3-6.2 0-11.5-4.1-13.4-9.9l-7.9 6.1C6.6 42.6 14.6 48 24 48z" />
    </svg>
  </span>
);
