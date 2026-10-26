import React, { useEffect, useRef, useState } from 'react';
import { LogOut } from 'lucide-react';
import { signOut, type Account } from '../services/session';

/** The signed-in account in the header: its picture, and a menu to sign out. */
export const AccountMenu: React.FC<{ account: Account }> = ({ account }) => {
  const [open, setOpen] = useState(false);
  const [pictureFailed, setPictureFailed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const label = account.name || account.email;
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-[#f1eee7] dark:bg-[#20201d] text-[#58423a] dark:text-[#a09088] text-xs font-semibold ring-1 ring-[#e5e2db] dark:ring-[#2d2d2a] hover:ring-[#a53705]/50 transition"
        title={account.email}
        aria-label="Account"
        aria-expanded={open}
      >
        {account.picture && !pictureFailed ? (
          // Google's avatar host refuses requests that carry a referrer.
          <img
            src={account.picture}
            alt=""
            referrerPolicy="no-referrer"
            className="w-full h-full object-cover"
            onError={() => setPictureFailed(true)}
          />
        ) : (
          label.slice(0, 1).toUpperCase()
        )}
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 mt-2 w-64 max-w-[calc(100vw-2rem)] rounded-xl border border-[#e5e2db] dark:border-[#2d2d2a] bg-[#fcf9f2] dark:bg-[#1c1c19] shadow-lg p-1.5 z-50"
        >
          <div className="px-3 py-2">
            {account.name && (
              <div className="text-sm font-medium text-[#1c1c18] dark:text-[#fcf9f2] truncate">{account.name}</div>
            )}
            <div className="text-xs text-[#8c7169] dark:text-[#a09088] truncate">{account.email}</div>
          </div>
          <button
            role="menuitem"
            onClick={() => signOut()}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-[#58423a] dark:text-[#a09088] hover:bg-black/5 dark:hover:bg-white/10 hover:text-[#1c1c18] dark:hover:text-[#fcf9f2] transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign out
          </button>
          <div className="mt-1 pt-1.5 px-3 pb-1 border-t border-[#e5e2db] dark:border-[#2d2d2a] flex gap-3 text-xs text-[#8c7169] dark:text-[#a09088]">
            <a href="/privacy" className="hover:text-[#1c1c18] dark:hover:text-[#fcf9f2]">Privacy</a>
            <a href="/terms" className="hover:text-[#1c1c18] dark:hover:text-[#fcf9f2]">Terms</a>
          </div>
        </div>
      )}
    </div>
  );
};
