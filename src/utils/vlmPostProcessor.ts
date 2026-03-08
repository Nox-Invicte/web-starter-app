/**
 * VLM Output Post-Processor
 * 
 * Cleans up VLM output artifacts: repetition, hallucination,
 * formatting noise, and normalizes the text for reliable pattern detection.
 */

/**
 * Full post-processing pipeline for VLM output
 */
export function postProcessVLMOutput(raw: string): string {
  let text = raw;

  // 1. Remove common VLM preamble / commentary
  text = stripCommentary(text);

  // 2. Remove repeated lines (hallucination artifact)
  text = deduplicateLines(text);

  // 3. Remove repeated phrases within lines
  text = removeRepeatedPhrases(text);

  // 4. Normalize whitespace
  text = normalizeWhitespace(text);

  // 5. Fix common OCR-style substitutions the VLM might make
  text = fixCommonSubstitutions(text);

  return text.trim();
}

/**
 * Strip VLM commentary / preamble that isn't actual extracted text
 */
function stripCommentary(text: string): string {
  const lines = text.split('\n');
  const filtered: string[] = [];

  const commentaryPatterns = [
    /^(here is|here are|sure,|okay,|let me|i can see|i see the|i'll)/i,
    /^(the following text|extracted text:|text:$)/i,
    /^(note:|disclaimer:|sorry|unfortunately|i cannot|i don't)/i,
    /^\s*[-*]\s*(the image shows|this is a|it appears to be)/i,
  ];

  let foundContent = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (foundContent) filtered.push('');
      continue;
    }

    // Skip commentary lines at the start
    if (!foundContent && commentaryPatterns.some(p => p.test(trimmed))) {
      continue;
    }

    foundContent = true;
    filtered.push(line);
  }

  // Also strip trailing commentary
  while (filtered.length > 0) {
    const last = filtered[filtered.length - 1].trim();
    if (!last || commentaryPatterns.some(p => p.test(last))) {
      filtered.pop();
    } else {
      break;
    }
  }

  return filtered.join('\n');
}

/**
 * Remove exact duplicate lines (common VLM hallucination)
 */
function deduplicateLines(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  const seen = new Set<string>();

  for (const line of lines) {
    const normalized = line.trim().toLowerCase();
    if (!normalized) {
      result.push(line);
      continue;
    }
    if (!seen.has(normalized)) {
      seen.add(normalized);
      result.push(line);
    }
  }

  return result.join('\n');
}

/**
 * Detect and remove repeated phrases within a single line
 * e.g., "Total Total Total 12.50" → "Total 12.50"
 */
function removeRepeatedPhrases(text: string): string {
  return text.split('\n').map(line => {
    // Remove word-level repetition (same word 3+ times in a row)
    let cleaned = line.replace(/\b(\w+)(\s+\1){2,}\b/gi, '$1');
    // Remove phrase-level repetition (2-4 word phrases repeated)
    cleaned = cleaned.replace(/(.{4,40}?)\1{2,}/g, '$1');
    return cleaned;
  }).join('\n');
}

/**
 * Normalize whitespace: collapse multiple spaces, trim lines
 */
function normalizeWhitespace(text: string): string {
  return text
    .split('\n')
    .map(line => line.replace(/\s{2,}/g, ' ').trim())
    .join('\n')
    // Collapse 3+ blank lines into 1
    .replace(/\n{3,}/g, '\n\n');
}

/**
 * Fix common character substitutions VLMs make
 */
function fixCommonSubstitutions(text: string): string {
  let fixed = text;

  // Common l/1/I confusions in specific contexts
  // Fix "0" vs "O" in contexts where digits are expected (card numbers, etc.)
  // Only in clearly numeric contexts
  fixed = fixed.replace(/\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?)(\d{0,4})\b/g, (match) => {
    return match.replace(/[oO]/g, '0').replace(/[lI]/g, '1');
  });

  // Fix common email domain typos
  fixed = fixed.replace(/@gmai1\.com/gi, '@gmail.com');
  fixed = fixed.replace(/@grnail\.com/gi, '@gmail.com');

  return fixed;
}

/**
 * Calculate a confidence score for VLM output quality (0-100)
 * Includes hallucination detection — if the output looks fabricated,
 * score drops to near zero.
 */
export function assessVLMOutputQuality(text: string): number {
  if (!text || text.trim().length === 0) return 0;

  // Check for hallucination FIRST — if detected, return 0 immediately
  if (isHallucinated(text)) return 0;

  let score = 50; // baseline

  const lines = text.split('\n').filter(l => l.trim());

  // More lines of content = likely better extraction
  if (lines.length >= 3) score += 10;
  if (lines.length >= 8) score += 5;

  // Contains numbers/dates = likely real content
  if (/\d/.test(text)) score += 10;

  // Contains recognizable patterns (emails, phones, etc.)
  if (/@/.test(text)) score += 5;
  if (/\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/.test(text)) score += 5;
  if (/\$|€|£|¥|RM|SGD|₹/.test(text)) score += 5;

  // Penalty for very short output (likely failed)
  if (text.trim().length < 20) score -= 20;

  // Penalty for excessive repetition
  const uniqueLines = new Set(lines.map(l => l.trim().toLowerCase()));
  const repetitionRatio = uniqueLines.size / Math.max(lines.length, 1);
  if (repetitionRatio < 0.5) score -= 15;

  // Penalty for VLM refusing / commentary taking over
  if (/sorry|cannot|unable|don't see|no text/i.test(text)) score -= 25;

  return Math.max(0, Math.min(100, score));
}

/**
 * Detect if VLM output is hallucinated (fabricated text not from the image).
 * 
 * Common hallucination patterns from tiny VLMs:
 * - Privacy policies, legal disclaimers, terms of service
 * - Generic descriptions instead of actual text extraction
 * - Model identity/capability statements
 * - Fictional narratives unrelated to OCR
 */
export function isHallucinated(text: string): boolean {
  if (!text || text.trim().length === 0) return false;

  const lower = text.toLowerCase();

  // Privacy policy / legal boilerplate hallucination
  const legalPatterns = [
    /privacy policy/i,
    /terms (of|and) (service|use|conditions)/i,
    /confidential(ity)?\s+(under|policy|agreement)/i,
    /lawful (and )?appropriate processing/i,
    /personal (data|details|information)\s+(can|will|may|is|are)\s+(not\s+)?(be\s+)?(used|shared|disclosed|collected|processed)/i,
    /data protection/i,
    /all (other )?records will (always )?remain/i,
    /cannot (otherwise )?reflectively represent/i,
    /does not mean that any (information|data)/i,
    /remain confidential/i,
    /kept private and secure/i,
  ];

  // Model self-description / capability statements
  const modelPatterns = [
    /as (an )?ai (language )?model/i,
    /i('m| am) (a |an )?(language |ai )?model/i,
    /trained (by|on|to)/i,
    /my (training|knowledge) (data|cutoff)/i,
  ];

  // Generic filler / narrative hallucination
  const fillerPatterns = [
    /for purposes of this (instance|example|demonstration)/i,
    /labeled and numbered for/i,
    /as per your data and format may vary/i,
    /it does not mean that/i,
    /full legal relationship/i,
    /the name shown above cannot/i,
  ];

  const allPatterns = [...legalPatterns, ...modelPatterns, ...fillerPatterns];

  // Count how many hallucination patterns match
  let matchCount = 0;
  for (const pattern of allPatterns) {
    if (pattern.test(lower)) matchCount++;
  }

  // If 2+ hallucination patterns match, it's fabricated
  if (matchCount >= 2) return true;

  // Also check: if the text is very long but has almost no numbers or
  // special characters, it's likely prose, not OCR content from a payment screen
  const wordCount = lower.split(/\s+/).length;
  const digitCount = (lower.match(/\d/g) || []).length;
  const avgWordLen = lower.replace(/\s+/g, '').length / Math.max(wordCount, 1);

  // Long prose with almost no digits and high average word length = likely hallucinated
  if (wordCount > 50 && digitCount < 3 && avgWordLen > 5) {
    // Check if it reads like natural language prose (many common English words)
    const commonWords = ['the', 'and', 'for', 'that', 'this', 'with', 'are', 'not', 'can', 'will', 'our', 'any', 'been', 'their', 'have', 'which', 'under', 'only'];
    const commonCount = commonWords.filter(w => lower.includes(` ${w} `)).length;
    if (commonCount >= 8) return true;
  }

  return false;
}

/**
 * Merge sensitive data from VLM targeted scan with regex-detected patterns
 */
export function parseVLMSensitiveFindings(vlmOutput: string): Array<{ type: string; value: string }> {
  const findings: Array<{ type: string; value: string }> = [];

  if (!vlmOutput || /^NONE$/i.test(vlmOutput.trim())) return findings;

  const lines = vlmOutput.split('\n').filter(l => l.trim());

  for (const line of lines) {
    const trimmed = line.trim();

    // Try to parse "Type: Value" or "- Type: Value" format
    const match = trimmed.match(/^[-*•]?\s*(?:(\w[\w\s/]+?):\s+)?(.+)$/);
    if (match) {
      const type = match[1]?.trim() || 'Unknown';
      const value = match[2]?.trim();
      if (value && value.length > 2) {
        findings.push({ type, value });
      }
    }
  }

  return findings;
}
