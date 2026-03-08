/**
 * Smart VLM Prompt Engineering System
 * 
 * Instead of one generic prompt, uses specialized prompts
 * for different content types and decomposed tasks.
 * This makes the small VLM (450M params) much more effective.
 */

export type ContentType = 'receipt' | 'screenshot' | 'document' | 'id_card' | 'payment' | 'unknown';

export interface VLMPrompt {
  id: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  description: string;
}

/**
 * Classification prompt — identify what type of image this is.
 * Kept very short for 450M model.
 */
export const CLASSIFY_PROMPT: VLMPrompt = {
  id: 'classify',
  prompt: 'What is this image? Answer: receipt, screenshot, document, ID card, or payment app.',
  maxTokens: 10,
  temperature: 0.5,
  description: 'Classify image content type',
};

/**
 * Extraction prompts — short and direct for small VLM.
 * The 450M model works best with concise instructions.
 */
export const EXTRACTION_PROMPTS: Record<ContentType, VLMPrompt> = {
  receipt: {
    id: 'extract_receipt',
    prompt: 'Read all text on this receipt. Include store name, items, prices, total, date, and any card numbers.',
    maxTokens: 300,
    temperature: 0.5,
    description: 'Extract receipt text',
  },
  screenshot: {
    id: 'extract_screenshot',
    prompt: 'Read all text visible on this screen. Include labels, messages, emails, passwords, codes, and URLs.',
    maxTokens: 300,
    temperature: 0.5,
    description: 'Extract screenshot text',
  },
  document: {
    id: 'extract_document',
    prompt: 'Read all text in this document. Include names, addresses, dates, and reference numbers.',
    maxTokens: 300,
    temperature: 0.5,
    description: 'Extract document text',
  },
  id_card: {
    id: 'extract_id',
    prompt: 'Read all text and numbers on this card or ID. Include name, ID number, dates, and address.',
    maxTokens: 200,
    temperature: 0.5,
    description: 'Extract ID card text',
  },
  payment: {
    id: 'extract_payment',
    prompt: 'Read all text on this payment screen. Include amount, name, phone number, transaction ID, date, and bank or app name.',
    maxTokens: 300,
    temperature: 0.5,
    description: 'Extract payment/transaction text',
  },
  unknown: {
    id: 'extract_generic',
    prompt: 'Read all visible text in this image. List every word, number, and symbol you see.',
    maxTokens: 300,
    temperature: 0.5,
    description: 'Extract all text (generic)',
  },
};

/**
 * Targeted sensitive data extraction prompt — kept short
 */
export const SENSITIVE_DATA_PROMPT: VLMPrompt = {
  id: 'sensitive_scan',
  prompt: 'List any passwords, credit card numbers, API keys, emails, phone numbers, or personal IDs visible in this image.',
  maxTokens: 200,
  temperature: 0.5,
  description: 'Targeted sensitive data extraction',
};

/**
 * Parse the classification response into a ContentType.
 * Very forgiving — the 450M model may respond with phrases, not just one word.
 */
export function parseClassification(response: string): ContentType {
  const text = response.toLowerCase().trim();

  // Check for keywords anywhere in the response
  if (/payment|gpay|paytm|venmo|paypal|upi|transfer|transaction|wallet|money\s*sent/i.test(text)) return 'payment';
  if (/receipt|bill|invoice|store|shop|total|price/i.test(text)) return 'receipt';
  if (/screenshot|screen|app|phone|computer|chat|notification|ui|interface/i.test(text)) return 'screenshot';
  if (/id.?card|passport|license|driver|identity|credit.?card|debit/i.test(text)) return 'id_card';
  if (/document|letter|form|contract|certificate|paper|memo/i.test(text)) return 'document';

  return 'unknown';
}

/**
 * Stored calibration: which prompts work best for this user's content
 */
export interface PromptCalibration {
  contentType: ContentType;
  bestPromptId: string;
  accuracy: number;
  samplesEvaluated: number;
  calibratedAt: string;
}

export function saveCalibration(calibrations: PromptCalibration[]): void {
  try {
    localStorage.setItem('vlm_prompt_calibration', JSON.stringify(calibrations));
  } catch { /* ignore */ }
}

export function loadCalibration(): PromptCalibration[] {
  try {
    const saved = localStorage.getItem('vlm_prompt_calibration');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
}
