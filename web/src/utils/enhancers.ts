/**
 * A cloud model id for people, as Studio labels it (studio/src/cloud.rs `model_label`):
 * `claude-sonnet-5-5` → "Claude Sonnet 5.5", `gemini-3.5-flash-lite` → "Gemini 3.5 Flash Lite".
 */
export function modelLabel(model: string): string {
  const words: string[] = [];
  let prevNumeric = false;
  for (const part of model.split(/[-_]/)) {
    if (!part || part === 'it' || /^\d{8,}$/.test(part)) continue;
    const numeric = /^\d+$/.test(part);
    if (numeric && prevNumeric && words.length) {
      words[words.length - 1] += `.${part}`;
      continue;
    }
    prevNumeric = numeric;
    words.push(/\d/.test(part) ? part.toUpperCase() : part[0].toUpperCase() + part.slice(1));
  }
  return words.join(' ');
}
