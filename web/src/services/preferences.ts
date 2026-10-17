import type { UserPreferences } from '../types';

/** Composer and theme defaults until Studio returns the saved preferences. */
export const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'light',
  aspect: '1:1',
  sizeMultiplier: 'full',
  quality: 'standard',
  steps: 20,
  count: 1,
  isSeedLocked: false,
  lockedSeed: 42,
  enhance: false,
  advancedOpen: false,
  tiling: false,
  extraRows: [],
};
