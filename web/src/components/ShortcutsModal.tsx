import React from 'react';
import { X, Keyboard } from 'lucide-react';

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ isOpen, onClose }) => {
  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const shortcuts = [
    { key: '/', description: 'Focus composer prompt input' },
    { key: 'Enter', description: 'Submit prompt for generation' },
    { key: 'Shift + Enter', description: 'Insert newline in prompt' },
    { key: 'G', description: 'Switch to Gallery view' },
    { key: 'T', description: 'Switch to Timeline view' },
    { key: '← / →', description: 'Previous / next image in Lightbox (pan when zoomed in)' },
    { key: '+ / −', description: 'Zoom in / out in Lightbox' },
    { key: '0 / 1', description: 'Fit to screen / actual size (1:1) in Lightbox' },
    { key: 'F', description: 'Toggle favorite on active image' },
    { key: 'Esc', description: 'Close Lightbox or modal' },
  ];

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl overflow-hidden flex flex-col"
      >
        <div className="px-6 py-4 border-b border-[#e5e2db] dark:border-[#2e2e2a] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Keyboard className="w-5 h-5 text-[#a53705]" />
            <h3 className="font-['Plus_Jakarta_Sans'] font-bold text-base">Keyboard Shortcuts</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 divide-y divide-[#e5e2db]/50 dark:divide-[#2e2e2a]">
          {shortcuts.map((sc, i) => (
            <div key={i} className="py-2.5 flex items-center justify-between text-xs">
              <span className="text-[#58423a] dark:text-[#a09088]">{sc.description}</span>
              <kbd className="px-2 py-1 rounded bg-[#f1eee7] dark:bg-[#252522] border border-[#e5e2db] dark:border-[#333330] font-mono font-medium shadow-2xs">
                {sc.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="px-6 py-3 border-t border-[#e5e2db] dark:border-[#2e2e2a] bg-[#f6f3ec] dark:bg-[#232320] text-right">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90 transition-opacity"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
