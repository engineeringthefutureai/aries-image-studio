import {
  type ComposerRow,
  type JobParams,
  type UserPreferences,
  QUALITY_STEPS,
  calculateDimensions,
  qualityForSteps,
  stepsFor,
} from '../types';

/**
 * The composer's rules in one place (SPEC_UI §4.2). With Advanced closed a send is one
 * row from the main settings: a preset's steps (custom counts as Standard), random seed,
 * no tiling. With Advanced open, custom steps, a fixed seed, tiling and the extra rows
 * apply too.
 */
export function mainRow(prefs: UserPreferences): ComposerRow {
  const advanced = prefs.advancedOpen;
  return {
    aspect: prefs.aspect,
    sizeMultiplier: prefs.sizeMultiplier || 'full',
    steps: advanced
      ? stepsFor(prefs.quality, prefs.steps)
      : prefs.quality === 'custom'
        ? QUALITY_STEPS.standard
        : stepsFor(prefs.quality, prefs.steps),
    enhance: prefs.enhance,
    tiling: advanced && prefs.tiling,
  };
}

/** Every row a send uses: the main settings, then the extra rows (Advanced open only). */
export function sendRows(prefs: UserPreferences): ComposerRow[] {
  return prefs.advancedOpen ? [mainRow(prefs), ...prefs.extraRows] : [mainRow(prefs)];
}

export const imagesPerSend = (prefs: UserPreferences) => sendRows(prefs).length * Math.max(1, prefs.count || 1);

/**
 * One seed per image of a row (count of them): a fixed seed counts up from itself
 * (Advanced open), otherwise each is drawn at random, all different.
 */
export function sendSeeds(prefs: UserPreferences, random: () => number = Math.random): number[] {
  const count = Math.max(1, prefs.count || 1);
  if (prefs.advancedOpen && prefs.isSeedLocked) {
    return Array.from({ length: count }, (_, k) => prefs.lockedSeed + k);
  }
  const seeds = new Set<number>();
  while (seeds.size < count) seeds.add(Math.floor(random() * 1000000));
  return [...seeds];
}

/**
 * The jobs one send makes: each row × count. Rows share the seeds so they compare like
 * for like: image k of every row gets seeds[k].
 */
export function buildJobs(prefs: UserPreferences, seeds: number[]): JobParams[] {
  return sendRows(prefs).flatMap((row) => {
    const { width, height } = calculateDimensions(row.aspect, row.sizeMultiplier);
    return seeds.map((seed) => ({
      width,
      height,
      aspect: row.aspect,
      sizeMultiplier: row.sizeMultiplier,
      steps: row.steps,
      seed,
      quality: qualityForSteps(row.steps),
      enhance: row.enhance,
      tiling: row.tiling,
    }));
  });
}
