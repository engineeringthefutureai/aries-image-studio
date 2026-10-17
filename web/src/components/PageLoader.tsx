import React from 'react';
import { Loader2 } from 'lucide-react';

/** Shown until Studio's first job list arrives, so the empty-state starter never flashes. */
export const PageLoader: React.FC = () => (
  <div className="flex-1 flex items-center justify-center p-16" role="status" aria-label="Loading">
    <Loader2 className="w-6 h-6 animate-spin text-[#8c7169] dark:text-[#a09088]" />
  </div>
);
