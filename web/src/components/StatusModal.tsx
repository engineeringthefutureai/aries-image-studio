import React, { useState, useEffect } from 'react';
import type { WorkerHealth, StageRunItem } from '../types';
import { X, Cpu, HardDrive, RefreshCw, Layers, Activity, AlertCircle, History } from 'lucide-react';
import { fetchStageRunsApi } from '../services/api';

interface StatusModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: WorkerHealth;
  onRefresh: () => void;
}

export const StatusModal: React.FC<StatusModalProps> = ({
  isOpen,
  onClose,
  health,
  onRefresh,
}) => {
  const [stageRuns, setStageRuns] = useState<StageRunItem[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);

  const loadRuns = async () => {
    setLoadingRuns(true);
    try {
      const runs = await fetchStageRunsApi(6);
      setStageRuns(runs);
    } catch {
      // ignore
    } finally {
      setLoadingRuns(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadRuns();
    }
  }, [isOpen]);

  const handleRefresh = () => {
    onRefresh();
    loadRuns();
  };

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const formatBytes = (bytes: number) => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-[#e5e2db] dark:border-[#2e2e2a] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-[#a53705]" />
            <h3 className="font-['Plus_Jakarta_Sans'] font-bold text-base">Generator Status & Hardware</h3>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={handleRefresh}
              className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088] transition-colors"
              title="Refresh telemetry"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="p-6 space-y-6 overflow-y-auto max-h-[75vh]">
          {/* Overall state */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db]/60 dark:border-[#2e2e2a]">
            <div>
              <div className="text-xs text-[#8c7169] dark:text-[#a09088] font-medium uppercase tracking-wider">
                Current Worker State
              </div>
              <div className="text-lg font-bold capitalize mt-0.5 flex items-center gap-2">
                <span>{health.state}</span>
                {health.state === 'offline' && (
                  <span className="text-xs font-normal text-rose-500 font-sans flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" /> Reconnection polling active
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Cards: current stage and VRAM */}
          {health.cards && health.cards.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[#a53705]" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088]">
                  GPUs & VRAM
                </h4>
              </div>

              <div className="space-y-3">
                {health.cards.map((card) => {
                  const cardIdx = card.card;
                  const isRunning = card.state === 'running';
                  const isUnavail = card.state === 'unavailable';
                  const used = card.vram_used ?? 0;
                  const total = card.vram_total ?? 0;
                  const peak = card.vram_peak ?? 0;
                  const reserved = card.vram_reserved ?? 0;
                  const overflow = card.vram_overflow ?? 0;
                  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;

                  return (
                    <div
                      key={cardIdx}
                      className="p-3.5 rounded-xl bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db]/60 dark:border-[#2e2e2a]"
                    >
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="font-semibold text-sm flex items-center gap-2">
                          <span>GPU {cardIdx}</span>
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium capitalize ${
                              isRunning
                                ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                                : isUnavail
                                ? 'bg-rose-500/15 text-rose-700 dark:text-rose-300'
                                : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
                            }`}
                          >
                            {card.state}
                          </span>
                        </span>
                        <span className="font-mono text-xs font-medium text-[#8c7169] dark:text-[#a09088]">
                          PCI {card.pci || 'N/A'}
                        </span>
                      </div>

                      {isRunning && (
                        <div className="text-xs font-mono text-[#a53705] dark:text-[#e06d3b] my-1">
                          Running stage: <span className="font-bold uppercase">{card.stage}</span>
                          {card.sub_stage && (
                            <span className="ml-1.5 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 font-semibold normal-case">
                              ({card.sub_stage})
                            </span>
                          )}
                          <span className="opacity-75 ml-1.5">(PID {card.pid})</span>
                          {card.job_id && <span className="opacity-75 ml-2">Job {card.job_id.slice(-8)}</span>}
                        </div>
                      )}

                      {isUnavail && card.unavailable_reason && (
                        <div className="text-xs text-rose-600 dark:text-rose-400 my-1">
                          Reason: {card.unavailable_reason}
                        </div>
                      )}

                      {!isRunning && !isUnavail && (
                        <div className="text-xs text-[#8c7169] dark:text-[#a09088] my-1">
                          Idle · VRAM reclaimed by process exit
                        </div>
                      )}

                      {/* VRAM Allocation Progress Bar & Detailed Telemetry */}
                      <div className="mt-2.5 pt-2.5 border-t border-[#e5e2db]/40 dark:border-[#2e2e2a]/60">
                        <div className="flex items-center justify-between text-xs mb-1 font-mono">
                          <span className="text-[#8c7169] dark:text-[#a09088] text-[11px] font-sans">
                            VRAM Allocation
                          </span>
                          <span className="font-medium text-xs">
                            {formatBytes(used)} / {formatBytes(total)} ({pct}%)
                          </span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-[#e5e2db] dark:bg-[#333330] overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              pct > 90
                                ? 'bg-rose-500'
                                : pct > 70
                                ? 'bg-amber-500'
                                : pct > 0
                                ? 'bg-[#a53705]'
                                : 'bg-transparent'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <div className="flex justify-between items-center text-[10px] text-[#8c7169] dark:text-[#a09088] font-mono mt-1.5">
                          <span>Peak: {formatBytes(peak)}</span>
                          <span>Reserved: {formatBytes(reserved)}</span>
                          {overflow > 0 ? (
                            <span className="text-amber-600 dark:text-amber-400 font-medium">
                              ⚠️ {formatBytes(overflow)} spill to host RAM
                            </span>
                          ) : (
                            <span className="hidden sm:inline">No host RAM spill</span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Architecture v2 Queue Depths */}
          {health.queue_depths && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Layers className="w-4 h-4 text-[#a53705]" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088]">
                  Stage Queue Depths
                </h4>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs">
                {(['enhance', 'encode', 'denoise', 'decode'] as const).map((stg) => (
                  <div key={stg} className="p-3 rounded-lg bg-[#f6f3ec] dark:bg-[#232320] text-center">
                    <div className="text-[#8c7169] dark:text-[#a09088] capitalize">{stg}</div>
                    <div className="text-lg font-bold mt-0.5">
                      {health.queue_depths?.[stg] ?? 0}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Architecture v2 Host RAM & Decode Floor */}
          {health.host_available_ram_bytes !== undefined && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <HardDrive className="w-4 h-4 text-[#a53705]" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088]">
                  Host Memory & Untiled 2048² Floor
                </h4>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-3 rounded-lg bg-[#f6f3ec] dark:bg-[#232320]">
                  <div className="text-[#8c7169] dark:text-[#a09088]">Available Host RAM</div>
                  <div className="font-semibold text-sm mt-0.5">
                    {formatBytes(health.host_available_ram_bytes)}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#f6f3ec] dark:bg-[#232320]">
                  <div className="text-[#8c7169] dark:text-[#a09088]">Untiled Decode Floor</div>
                  <div className="font-semibold text-sm mt-0.5">
                    {health.decode_ram_floor_bytes ? formatBytes(health.decode_ram_floor_bytes) : '6.0 GB'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Recent Worker Runs & Eviction Audit Logs */}
          {stageRuns && stageRuns.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-[#a53705]" />
                  <h4 className="text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088]">
                    Recent Worker Runs & Eviction History
                  </h4>
                </div>
                {loadingRuns && (
                  <span className="text-[10px] text-[#8c7169] dark:text-[#a09088] animate-pulse">
                    Refreshing...
                  </span>
                )}
              </div>
              <div className="space-y-2">
                {stageRuns.map((run) => {
                  const durationS = run.ended_at && run.started_at ? Math.max(0, run.ended_at - run.started_at) : 0;
                  const isClean = run.exit_code === 0;
                  return (
                    <div
                      key={run.id}
                      className="p-2.5 rounded-lg bg-[#f6f3ec] dark:bg-[#232320] border border-[#e5e2db]/40 dark:border-[#2e2e2a]/40 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-bold uppercase text-[11px] px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 font-mono">
                            {run.stage}
                          </span>
                          <span className="font-medium text-[11px] text-[#8c7169] dark:text-[#a09088]">
                            GPU {run.card} · PID {run.pid}
                          </span>
                          <span className="text-[10px] font-mono text-[#8c7169] dark:text-[#a09088]">
                            {durationS.toFixed(1)}s · {run.jobs_processed} job{run.jobs_processed === 1 ? '' : 's'}
                          </span>
                        </div>
                        <span
                          className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded ${
                            isClean
                              ? 'text-emerald-700 dark:text-emerald-300 bg-emerald-500/10'
                              : 'text-rose-700 dark:text-rose-300 bg-rose-500/10'
                          }`}
                        >
                          exit {run.exit_code ?? '?'}
                        </span>
                      </div>
                      {run.exit_reason && (
                        <div className="mt-1 text-[11px] text-[#58423a] dark:text-[#c4b5af] font-mono truncate">
                          Reason: <span className="opacity-90">{run.exit_reason}</span>
                        </div>
                      )}
                      {run.queue_depths && (
                        <div className="mt-1 flex items-center gap-2 text-[10px] font-mono text-[#8c7169] dark:text-[#a09088]">
                          <span>Queue at exit:</span>
                          <span>enh: {run.queue_depths.enhance ?? 0}</span>
                          <span>enc: {run.queue_depths.encode ?? 0}</span>
                          <span>den: {run.queue_depths.denoise ?? 0}</span>
                          <span>dec: {run.queue_depths.decode ?? 0}</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[#e5e2db] dark:border-[#2e2e2a] bg-[#f6f3ec] dark:bg-[#232320] flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90 transition-opacity"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
