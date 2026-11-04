import React from 'react';
import { Globe } from 'lucide-react';
import { useSignInError } from '../hooks/useSignInError';
import { signInHref, signOut } from '../services/session';

/** Under a demo visitor's header: what this is, and the way in for people who have an account. */
export const DemoBanner: React.FC = () => {
  const error = useSignInError();

  return (
    <div className="border-b border-sky-500/20 bg-sky-500/10 text-sky-900 dark:text-sky-200 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-1.5 font-semibold">
          <Globe className="w-3.5 h-3.5" />
          Read-only demo
        </span>
        <span className="flex-1 min-w-48">Browse what this Studio has made. Making images is by invitation.</span>
        {error && <span className="w-full order-last text-rose-700 dark:text-rose-300">{error}</span>}
        <a href="/privacy" className="underline underline-offset-2 hover:no-underline">
          Privacy
        </a>
        <a href="/terms" className="underline underline-offset-2 hover:no-underline">
          Terms
        </a>
        <button type="button" onClick={() => signOut()} className="underline underline-offset-2 hover:no-underline">
          Leave demo
        </button>
        <a href={signInHref()} className="font-semibold underline underline-offset-2 hover:no-underline">
          Sign in
        </a>
      </div>
    </div>
  );
};
