import React from 'react';
import type { JobReference } from '../types';
import { uploadThumbUrl } from '../utils/references';

/** A job's reference images in prompt order, numbered as the prompt names them ("image 1"). */
export const ReferenceStrip: React.FC<{ references: JobReference[]; className?: string; dark?: boolean }> = ({
  references,
  className = '',
  dark = false,
}) => (
  <div className={`flex flex-wrap gap-1.5 ${className}`} role="list" aria-label="Reference images">
    {references.map((ref, i) => (
      <a
        key={`${ref.sha256}-${i}`}
        href={`/api/uploads/${encodeURIComponent(ref.sha256)}`}
        target="_blank"
        rel="noreferrer"
        role="listitem"
        title={`Image ${i + 1} · ${ref.width}×${ref.height}`}
        className={`relative w-10 h-10 rounded-md overflow-hidden ${dark ? 'bg-white/10' : 'bg-[#f1eee7] dark:bg-[#232320]'}`}
      >
        <img src={uploadThumbUrl(ref.sha256)} alt={`Reference image ${i + 1}`} loading="lazy" className="w-full h-full object-cover" />
        <span className="absolute bottom-0 left-0 px-1 rounded-tr bg-black/60 text-white font-mono text-[9px] leading-3.5">
          {i + 1}
        </span>
      </a>
    ))}
  </div>
);
