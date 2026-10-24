import { useEffect } from 'react';

/**
 * Keeps the page from scrolling under a dialog while it is open (the shortcuts sheet and the
 * delete confirmation; the viewer and the status panel are pages of their own).
 */
export function useBodyScrollLock(locked: boolean): void {
  useEffect(() => {
    if (!locked) return;
    const root = document.documentElement;
    const previous = root.style.overflow;
    root.style.overflow = 'hidden';
    return () => {
      root.style.overflow = previous;
    };
  }, [locked]);
}
