import React from 'react';
import type { WorkerHealth } from '../types';
import { Grid, Clock, Moon, Sun, HelpCircle, AlertTriangle } from 'lucide-react';
import { Logo } from './Logo';

interface HeaderProps {
  activeView: 'timeline' | 'gallery';
  onViewChange: (view: 'timeline' | 'gallery') => void;
  workerHealth: WorkerHealth;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onOpenStatus: () => void;
  onOpenShortcuts: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeView,
  onViewChange,
  workerHealth,
  theme,
  onToggleTheme,
  onOpenStatus,
  onOpenShortcuts,
}) => {
  const isDark = theme === 'dark';
  const getStatusBadge = () => {
    if (workerHealth.state === 'offline') {
      return {
        label: 'Backend Offline',
        dotClass: 'bg-rose-500',
        bgClass: 'bg-rose-500/10 text-rose-800 dark:text-rose-300 border-rose-500/30',
        ping: true,
      };
    }

    if (workerHealth.state === 'error') {
      return {
        label: 'Hardware Error',
        dotClass: 'bg-red-500',
        bgClass: 'bg-red-500/10 text-red-800 dark:text-red-300 border-red-500/30',
        ping: false,
      };
    }

    const runningCards = (workerHealth.cards || []).filter((c) => c.state === 'running');
    if (runningCards.length > 0) {
      const summary = runningCards.map((c) => `GPU ${c.card}: ${c.stage}`).join(', ');
      return {
        label: `Busy (${summary})`,
        dotClass: 'bg-amber-500',
        bgClass: 'bg-amber-500/10 text-amber-800 dark:text-amber-300 border-amber-500/20',
        ping: true,
      };
    }

    if (workerHealth.state === 'busy') {
      return {
        label: 'Busy',
        dotClass: 'bg-amber-500',
        bgClass: 'bg-amber-500/10 text-amber-800 dark:text-amber-300 border-amber-500/20',
        ping: true,
      };
    }

    return {
      label: 'Ready',
      dotClass: 'bg-emerald-500',
      bgClass: 'bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 border-emerald-500/20',
      ping: false,
    };
  };

  const statusInfo = getStatusBadge();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-[#e5e2db] dark:border-[#272724] bg-[#fcf9f2]/85 dark:bg-[#151512]/85 backdrop-blur-md transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <Logo className="w-9 h-9 shrink-0 drop-shadow-sm" />
          <div className="flex items-center gap-2">
            <span className="font-['Plus_Jakarta_Sans'] font-bold text-lg text-[#1c1c18] dark:text-[#fcf9f2] tracking-tight">
              Aries Image Studio
            </span>
            <span className="font-mono text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded-full bg-[#ebe8e1] dark:bg-[#252522] text-[#8c7169] dark:text-[#a09088] leading-none hidden sm:inline-flex items-center">
              v{__APP_VERSION__}
            </span>
          </div>
        </div>

        {/* View Switcher: Timeline / Gallery */}
        <div className="flex items-center bg-[#f1eee7] dark:bg-[#20201d] p-1 rounded-full border border-[#e5e2db]/50 dark:border-[#2d2d2a]">
          <button
            onClick={() => onViewChange('timeline')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
              activeView === 'timeline'
                ? 'bg-[#a53705] text-white shadow-sm'
                : 'text-[#58423a] dark:text-[#a09088] hover:text-[#1c1c18] dark:hover:text-[#fcf9f2]'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Timeline</span>
            <kbd className="hidden sm:inline-block font-mono text-[10px] opacity-70 px-1 py-0.2 rounded bg-black/10 dark:bg-white/10">
              T
            </kbd>
          </button>
          <button
            onClick={() => onViewChange('gallery')}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
              activeView === 'gallery'
                ? 'bg-[#a53705] text-white shadow-sm'
                : 'text-[#58423a] dark:text-[#a09088] hover:text-[#1c1c18] dark:hover:text-[#fcf9f2]'
            }`}
          >
            <Grid className="w-3.5 h-3.5" />
            <span>Gallery</span>
            <kbd className="hidden sm:inline-block font-mono text-[10px] opacity-70 px-1 py-0.2 rounded bg-black/10 dark:bg-white/10">
              G
            </kbd>
          </button>
        </div>

        {/* Right Actions: Status Pill, Theme, Shortcuts */}
        <div className="flex items-center gap-2">
          {/* Status Pill */}
          <button
            onClick={onOpenStatus}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors shadow-2xs hover:opacity-90 ${statusInfo.bgClass}`}
            title="Click to view hardware telemetry and queue status"
          >
            <span className="relative flex h-2 w-2">
              {statusInfo.ping && (
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${statusInfo.dotClass}`}
                />
              )}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${statusInfo.dotClass}`} />
            </span>
            <span>{statusInfo.label}</span>
            {workerHealth.state === 'offline' && <AlertTriangle className="w-3.5 h-3.5 ml-0.5" />}
          </button>

          {/* Theme Toggle */}
          <button
            onClick={onToggleTheme}
            className="w-8 h-8 rounded-full flex items-center justify-center bg-[#f1eee7] dark:bg-[#20201d] text-[#58423a] dark:text-[#a09088] hover:text-[#1c1c18] dark:hover:text-[#fcf9f2] transition-colors"
            title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Keyboard Shortcuts Dialog */}
          <button
            onClick={onOpenShortcuts}
            className="w-8 h-8 rounded-full flex items-center justify-center bg-[#f1eee7] dark:bg-[#20201d] text-[#58423a] dark:text-[#a09088] hover:text-[#1c1c18] dark:hover:text-[#fcf9f2] transition-colors"
            title="Keyboard shortcuts (?)"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
