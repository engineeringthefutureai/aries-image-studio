import { useEffect, useState } from 'react';
import { SIGN_IN_ERRORS } from '../services/session';

/** Why the last sign-in came back without a session (`?auth_error=`), once: a reload starts clean. */
export function useSignInError(): string | null {
  const [error] = useState(() => {
    const code = new URLSearchParams(window.location.search).get('auth_error');
    return code ? (SIGN_IN_ERRORS[code] ?? SIGN_IN_ERRORS.failed) : null;
  });

  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.has('auth_error')) {
      url.searchParams.delete('auth_error');
      window.history.replaceState(window.history.state, '', url);
    }
  }, []);

  return error;
}
