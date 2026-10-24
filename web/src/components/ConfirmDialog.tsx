import React, { useEffect, useRef } from 'react';
import { Trash2 } from 'lucide-react';
import { useBodyScrollLock } from '../hooks/useBodyScrollLock';

export interface ConfirmRequest {
  title: string;
  message: string;
  confirmLabel: string;
}

interface ConfirmDialogProps {
  request: ConfirmRequest | null;
  onConfirm: () => void;
  onCancel: () => void;
}

/** A yes/no question over the page, e.g. before a delete. Enter confirms, Esc cancels. */
export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({ request, onConfirm, onCancel }) => {
  const open = request !== null;
  const confirmRef = useRef<HTMLButtonElement>(null);
  useBodyScrollLock(open);

  useEffect(() => {
    if (!open) return;
    confirmRef.current?.focus();
    // Capture phase, so the page's own shortcuts (Esc leaves the viewer, F favorites, ←/→)
    // never see a key meant for the dialog. Enter and Space still press the focused button.
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onCancel]);

  if (!request) return null;

  return (
    <div
      onClick={onCancel}
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-message"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-sm rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl p-6 select-none"
      >
        <h3 id="confirm-title" className="font-['Plus_Jakarta_Sans'] font-bold text-base">
          {request.title}
        </h3>
        <p id="confirm-message" className="mt-2 text-sm text-[#58423a] dark:text-[#a09088]">
          {request.message}
        </p>
        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#f1eee7] dark:bg-[#2c2c28] hover:bg-[#e5e2db] dark:hover:bg-[#363632] transition-colors"
          >
            Cancel
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-medium bg-rose-600 text-white hover:bg-rose-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 focus-visible:ring-offset-2 focus-visible:ring-offset-[#fcf9f2] dark:focus-visible:ring-offset-[#1a1a17]"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {request.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};
