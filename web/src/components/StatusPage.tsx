import React, { useState, useEffect } from 'react';
import type { WorkerHealth, SystemExecutorInfo } from '../types';
import {
  X,
  Cpu,
  HardDrive,
  RefreshCw,
  Layers,
  Activity,
  AlertCircle,
  Server,
} from 'lucide-react';
import { controlExecutorApi } from '../services/api';

const SPILL_WARN_BYTES = 1024 ** 3;

interface StatusPageProps {
  /** Leave the page (Esc, the close button): back to where it was opened from. */
  onClose: () => void;
  health: WorkerHealth;
  onRefresh: () => void;
  /** Lend / Disable buttons: only for the owner when Studio has sign-in. */
  canControl: boolean;
}

/** The system view at `/status`: a page of its own, not a layer over the timeline. */
export const StatusPage: React.FC<StatusPageProps> = ({
  onClose,
  health,
  onRefresh,
  canControl,
}) => {
  const [busyExecutorId, setBusyExecutorId] = useState<string | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  const handleExecutorControl = async (
    executorId: string,
    action: 'lend' | 'return' | 'disable' | 'enable'
  ) => {
    setBusyExecutorId(executorId);
    try {
      await controlExecutorApi(executorId, action);
      onRefresh();
    } finally {
      setBusyExecutorId(null);
    }
  };

  const renderExecutorStatePill = (ex: SystemExecutorInfo) => {
    let label = ex.state;
    if ((ex.state === 'ready' || ex.state === 'starting') && ex.current_stage) {
      label = `${ex.state}: ${ex.current_stage}`;
    } else if (ex.state === 'busy' && ex.current_stage) {
      const shortTask = ex.current_task_id ? ` (${ex.current_task_id.slice(-6)})` : '';
      label = `busy: ${ex.current_stage}${shortTask}`;
    } else if (ex.state === 'disabled' && ex.disabled_reason) {
      label = `disabled: ${ex.disabled_reason}`;
    } else if (ex.pending_lend) {
      label = `${ex.state} (lending after task)`;
    }

    const isBusy = ex.state === 'busy' || ex.state === 'starting';
    const isUnavail = ex.state === 'offline' || ex.state === 'disabled' || ex.state === 'lent';

    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-medium ${
          isBusy
            ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
            : isUnavail
            ? 'bg-rose-500/15 text-rose-700 dark:text-rose-300'
            : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
        }`}
      >
        {label}
      </span>
    );
  };

  const agents = health.agents || [];
  const queues = health.queues || [];

  return (
    <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-6">
      <div className="w-full rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-xs overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-[#e5e2db] dark:border-[#2e2e2a] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-[#a53705]" />
            <h3 className="font-['Plus_Jakarta_Sans'] font-bold text-base">
              System View · Agents &amp; Executors
            </h3>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={onRefresh}
              className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088] transition-colors cursor-pointer"
              title="Refresh telemetry"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088] transition-colors cursor-pointer"
              title="Close (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="p-6 space-y-6">
          {/* Queue strip (§11.1): 4 stage boxes with depth, oldest wait, and blocked reasons */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-[#a53705]" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088]">
                Stage Queues
              </h4>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
              {(['enhance', 'encode', 'denoise', 'decode'] as const).map((stg) => {
                const qInfo = queues.find((q) => q.stage === stg);
                const depth = qInfo ? qInfo.queued : 0;
                const running = qInfo?.running ?? 0;
                const oldestWait = qInfo?.oldest_wait_s ?? 0;
                const blockedReasons = qInfo?.blocked_reasons || [];

                return (
                  <div
                    key={stg}
                    className="p-3 rounded-xl bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db]/60 dark:border-[#2e2e2a] flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[#8c7169] dark:text-[#a09088] font-semibold capitalize">
                        {stg}
                      </span>
                      {running > 0 && (
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300">
                          {running} active
                        </span>
                      )}
                    </div>
                    <div className="text-xl font-bold mt-1 font-mono">{depth}</div>
                    {oldestWait > 0.5 && (
                      <div className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088] mt-0.5">
                        wait: {oldestWait.toFixed(1)}s
                      </div>
                    )}
                    {blockedReasons.length > 0 && (
                      <div
                        className="mt-1.5 text-[10px] text-amber-700 dark:text-amber-300 bg-amber-500/10 px-1.5 py-1 rounded leading-tight"
                        title={blockedReasons.join('; ')}
                      >
                        ⚠️ {blockedReasons[0]}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Agent cards (§11.1) */}
          {agents.length > 0 ? (
            <div className="space-y-4">
              {agents.map((ag) => {
                const ramTotal = ag.host_ram_total_bytes || 0;
                const ramAvail = ag.host_ram_available_bytes || 0;
                const ramUsed = Math.max(0, ramTotal - ramAvail);
                const ramPct = ramTotal > 0 ? Math.min(100, Math.round((ramUsed / ramTotal) * 100)) : 0;

                return (
                  <div
                    key={ag.id}
                    className="p-4 rounded-xl bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db] dark:border-[#2e2e2a] space-y-4"
                  >
                    {/* Agent Header */}
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#e5e2db]/60 dark:border-[#2e2e2a]">
                      <div className="flex items-center gap-2.5">
                        <Server className="w-4 h-4 text-[#a53705]" />
                        <span className="font-bold text-sm">{ag.id}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase font-semibold bg-[#a53705]/15 text-[#a53705] dark:text-[#f38d58]">
                          {ag.backend_kind}
                        </span>
                        <span className="flex items-center gap-1 text-xs font-mono">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              ag.online ? 'bg-emerald-500' : 'bg-rose-500'
                            }`}
                          />
                          <span>{ag.online ? 'online' : 'offline'}</span>
                        </span>
                      </div>

                      <div className="flex items-center gap-3 text-xs font-mono text-[#8c7169] dark:text-[#a09088]">
                        <span>v{ag.agent_version}</span>
                        <span
                          className={
                            ag.scrape_stale
                              ? 'text-rose-600 dark:text-rose-400 font-semibold'
                              : ''
                          }
                        >
                          {ag.scrape_age_s != null
                            ? ag.scrape_stale
                              ? `stale (${ag.scrape_age_s.toFixed(1)}s ago)`
                              : `scraped ${ag.scrape_age_s.toFixed(1)}s ago`
                            : 'no scrape yet'}
                        </span>
                      </div>
                    </div>

                    {/* Host RAM bar */}
                    {ramTotal > 0 && (
                      <div>
                        <div className="flex items-center justify-between text-xs font-mono mb-1">
                          <span className="flex items-center gap-1.5 text-[#8c7169] dark:text-[#a09088] font-sans text-[11px]">
                            <HardDrive className="w-3.5 h-3.5 text-[#a53705]" />
                            Host RAM ({formatBytes(ramAvail)} free of {formatBytes(ramTotal)})
                          </span>
                          <span>{ramPct}% used</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-[#e5e2db] dark:bg-[#333330] overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              ramPct > 85 ? 'bg-amber-500' : 'bg-[#a53705]'
                            }`}
                            style={{ width: `${ramPct}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Executors List */}
                    <div className="space-y-2.5">
                      {ag.executors.map((ex) => {
                        const used = ex.mem_used_bytes || 0;
                        const total = ex.mem_total_bytes || 0;
                        const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
                        const maxPx = Math.max(
                          0,
                          ...ex.stages.map((s) => s.max_px || 0)
                        );
                        const maxSide = maxPx > 0 ? Math.round(Math.sqrt(maxPx)) : 0;
                        const stageList = ex.stages.map((s) => s.stage).join(', ');
                        const isControlBusy = busyExecutorId === ex.id;
                        const isLent = ex.state === 'lent' || ex.lent;
                        const isDisabled = ex.state === 'disabled' || Boolean(ex.disabled_reason);

                        return (
                          <div
                            key={ex.key}
                            className="p-3 rounded-xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db]/80 dark:border-[#2e2e2a] space-y-2"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="flex items-center gap-2">
                                <Cpu className="w-3.5 h-3.5 text-[#a53705]" />
                                <span className="font-mono font-bold text-xs">{ex.id}</span>
                                <span className="font-mono text-[11px] text-[#8c7169] dark:text-[#a09088]">
                                  {ex.device}
                                </span>
                                {renderExecutorStatePill(ex)}
                              </div>

                              {/* Operator controls: [Lend]/[Return] and [Disable]/[Enable] */}
                              {canControl && (
                                <div className="flex items-center gap-1.5">
                                  {isLent ? (
                                    <button
                                      disabled={isControlBusy}
                                      onClick={() => handleExecutorControl(ex.id, 'return')}
                                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-800 dark:text-emerald-300 transition-colors cursor-pointer"
                                    >
                                      Return
                                    </button>
                                  ) : (
                                    <button
                                      disabled={isControlBusy || isDisabled}
                                      onClick={() => handleExecutorControl(ex.id, 'lend')}
                                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/15 text-[#58423a] dark:text-[#a09088] disabled:opacity-40 transition-colors cursor-pointer"
                                      title="Lend GPU for external work (stops resident worker)"
                                    >
                                      Lend
                                    </button>
                                  )}

                                  {isDisabled ? (
                                    <button
                                      disabled={isControlBusy}
                                      onClick={() => handleExecutorControl(ex.id, 'enable')}
                                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-500/15 hover:bg-amber-500/25 text-amber-800 dark:text-amber-300 transition-colors cursor-pointer"
                                    >
                                      Enable
                                    </button>
                                  ) : (
                                    <button
                                      disabled={isControlBusy}
                                      onClick={() => handleExecutorControl(ex.id, 'disable')}
                                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-rose-500/10 hover:bg-rose-500/20 text-rose-700 dark:text-rose-300 transition-colors cursor-pointer"
                                    >
                                      Disable
                                    </button>
                                  )}
                                </div>
                              )}
                            </div>

                            {/* Memory bar + capabilities */}
                            <div>
                              <div className="flex items-center justify-between text-[11px] font-mono text-[#8c7169] dark:text-[#a09088] mb-1">
                                <span>
                                  stages: {stageList}
                                  {maxSide > 0 ? ` (max ${maxSide}²)` : ''}
                                </span>
                                <span>
                                  {formatBytes(used)} / {formatBytes(total)} ({pct}%)
                                </span>
                              </div>
                              <div className="w-full h-1.5 rounded-full bg-[#e5e2db] dark:bg-[#333330] overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    pct > 90
                                      ? 'bg-rose-500'
                                      : pct > 70
                                      ? 'bg-amber-500'
                                      : 'bg-[#a53705]'
                                  }`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              {ex.overflow_host_bytes > 0 && (
                                <div className="mt-1 text-[10px] font-mono text-amber-600 dark:text-amber-400">
                                  {ex.overflow_host_bytes >= SPILL_WARN_BYTES ? '⚠️ ' : ''}
                                  Driver overflow to host RAM: {formatBytes(ex.overflow_host_bytes)}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db]/60 dark:border-[#2e2e2a] flex items-center gap-2 text-xs text-[#8c7169] dark:text-[#a09088]">
              <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
              <span>No agents currently registered with the backend.</span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
