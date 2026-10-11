/**
 * Clean up and extract human-readable enhanced prompt text,
 * stripping any raw JSON wrapper, markdown blocks, truncated JSON tokens,
 * or misspelled keys like "enhanted_prompt" or "enhritten_prompt".
 */
export function sanitizeEnhancedPrompt(text?: string | null): string {
  if (!text) return '';
  let str = text.trim();

  // If already clean prose without JSON delimiters, return directly
  if (!str.startsWith('{') && !str.startsWith('```') && !str.includes('prompt"')) {
    return str;
  }

  // 1. Try full JSON parse
  try {
    const cleanJson = str.replace(/^```json\s*|^```\s*|```$/g, '').trim();
    const parsed = JSON.parse(cleanJson);
    if (parsed && typeof parsed === 'object') {
      const candidate =
        parsed.rewritten_prompt ||
        parsed.enhanced_prompt ||
        parsed.enhritten_prompt ||
        parsed.enhanted_prompt;

      if (typeof candidate === 'string' && candidate.trim()) {
        return candidate.trim();
      }

      // If key had a different name/spelling, find first non-ratio string
      for (const [key, val] of Object.entries(parsed)) {
        if (key !== 'wh_ratio' && typeof val === 'string' && val.trim().length > 0) {
          return val.trim();
        }
      }
    }
  } catch {
    // Continue to regex fallback
  }

  // 2. Resilient regex extraction for complete or truncated JSON:
  // Matches "enhanced_prompt", "rewritten_prompt", "enhritten_prompt", "enhanted_prompt", etc.
  const match =
    str.match(/"(?:rewritten_prompt|enhanced_prompt|enhritten_prompt|enhanted_prompt|[^"]*prompt[^"]*)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)/i) ||
    str.match(/\{\s*"[^"]+"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)/);

  if (match && match[1]) {
    return match[1]
      .replace(/\\"/g, '"')
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '')
      .replace(/\\t/g, ' ')
      .trim();
  }

  // 3. Fallback: strip markdown and leftover wrapper characters
  str = str.replace(/^```json\s*|^```\s*|```$/g, '');
  str = str.replace(/^\s*\{\s*"[^"]+"\s*:\s*"?/i, '');
  str = str.replace(/"?\s*,?\s*"wh_ratio".*$/is, '');
  str = str.replace(/"?\s*\}?\s*$/, '');

  return str.trim();
}
