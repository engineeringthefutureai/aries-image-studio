import React from 'react';
import type { Generation } from '../types';
import { useThinking } from '../hooks/useThinking';

/** The enhancer's reasoning for an open reasoning panel, fetched on demand. */
export const ReasoningText: React.FC<{ generation: Generation; className?: string }> = ({ generation, className }) => {
  const text = useThinking(generation, true);
  return <div className={className}>{text ?? <span className="opacity-60">Loading reasoning…</span>}</div>;
};
