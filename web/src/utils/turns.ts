import type { Generation } from '../types';

/** Within a turn: the order of the send, left to right (older jobs without a position by creation time). */
const inSendOrder = (a: Generation, b: Generation) =>
  (a.batchIndex ?? Infinity) - (b.batchIndex ?? Infinity) || a.createdAt - b.createdAt || a.id.localeCompare(b.id);

/** Generations grouped into turns (one per batchId) as the timeline shows them: newest turn first, each in send order. */
export function groupTurns(generations: Generation[]): Generation[][] {
  const map = new Map<string, Generation[]>();
  for (const gen of generations) {
    const arr = map.get(gen.batchId) || [];
    arr.push(gen);
    map.set(gen.batchId, arr);
  }
  return Array.from(map.values())
    .map((turn) => turn.sort(inSendOrder))
    .sort((a, b) => Math.max(...b.map((g) => g.createdAt)) - Math.max(...a.map((g) => g.createdAt)));
}
