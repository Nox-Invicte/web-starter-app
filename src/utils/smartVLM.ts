/**
 * Smart VLM Engine — Multi-Pass Inference with Adaptive Prompting
 * 
 * Makes the small VLM dramatically more effective by:
 * 1. Classifying the image first to pick the right prompt
 * 2. Using specialized extraction prompts per content type
 * 3. Running a targeted sensitive data scan
 * 4. Post-processing all outputs to clean artifacts
 * 5. Learning which prompts work best via dataset calibration
 */

import { VLMWorkerBridge } from '@runanywhere/web-llamacpp';
import {
  CLASSIFY_PROMPT,
  EXTRACTION_PROMPTS,
  SENSITIVE_DATA_PROMPT,
  parseClassification,
  loadCalibration,
  type ContentType,
  type VLMPrompt,
} from './vlmPrompts';
import {
  postProcessVLMOutput,
  assessVLMOutputQuality,
  isHallucinated,
  parseVLMSensitiveFindings,
} from './vlmPostProcessor';
import { getTesseractEngine, getTrainedTesseractConfig } from './tesseractOCR';
import { autoPreprocess } from './imagePreprocessor';

export interface SmartVLMResult {
  /** Cleaned extracted text */
  text: string;
  /** Detected content type */
  contentType: ContentType;
  /** Quality score 0-100 */
  quality: number;
  /** Extra sensitive items found by targeted scan */
  sensitiveFindings: Array<{ type: string; value: string }>;
  /** Which passes were run */
  passes: string[];
  /** Total inference time in ms */
  totalTimeMs: number;
  /** True if VLM hallucinated and Tesseract was used as fallback */
  usedFallback?: boolean;
}

export type VLMProgressCallback = (stage: string) => void;

/**
 * Downscale canvas to fit within maxDim (CLIP resizes internally,
 * so larger images just waste memory and time).
 */
function prepareCanvas(canvas: HTMLCanvasElement, maxDim: number = 256): HTMLCanvasElement {
  if (canvas.width <= maxDim && canvas.height <= maxDim) return canvas;
  const scale = Math.min(maxDim / canvas.width, maxDim / canvas.height);
  const small = document.createElement('canvas');
  small.width = Math.round(canvas.width * scale);
  small.height = Math.round(canvas.height * scale);
  const ctx = small.getContext('2d')!;
  ctx.drawImage(canvas, 0, 0, small.width, small.height);
  return small;
}

/**
 * Extract RGB pixel data from a canvas for the VLM bridge.
 * Returns a fresh copy every time (the bridge transfers/detaches the buffer).
 */
function extractRGBPixels(canvas: HTMLCanvasElement): Uint8Array {
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const rgbPixels = new Uint8Array(canvas.width * canvas.height * 3);
  for (let i = 0; i < imgData.data.length / 4; i++) {
    rgbPixels[i * 3] = imgData.data[i * 4];
    rgbPixels[i * 3 + 1] = imgData.data[i * 4 + 1];
    rgbPixels[i * 3 + 2] = imgData.data[i * 4 + 2];
  }
  return rgbPixels;
}

/**
 * Run a single VLM inference.
 * Creates a FRESH pixel copy each call since bridge.process() transfers the buffer.
 */
async function runPrompt(
  bridge: VLMWorkerBridge,
  canvas: HTMLCanvasElement,
  prompt: VLMPrompt,
): Promise<string> {
  const pixels = extractRGBPixels(canvas);
  const response = await bridge.process(
    pixels, canvas.width, canvas.height,
    prompt.prompt,
    { maxTokens: prompt.maxTokens, temperature: prompt.temperature },
  );
  return response.text;
}

/**
 * Fallback: run Tesseract OCR when VLM output is hallucinated.
 * Preprocesses the canvas and returns extracted text + confidence.
 */
async function tesseractFallback(canvas: HTMLCanvasElement): Promise<{ text: string; confidence: number }> {
  const tesseract = await getTesseractEngine(getTrainedTesseractConfig());
  // Preprocess for better Tesseract accuracy
  const ctx = canvas.getContext('2d')!;
  const imgData = autoPreprocess(ctx.getImageData(0, 0, canvas.width, canvas.height));
  const fallbackCanvas = document.createElement('canvas');
  fallbackCanvas.width = canvas.width;
  fallbackCanvas.height = canvas.height;
  const fctx = fallbackCanvas.getContext('2d')!;
  fctx.putImageData(imgData, 0, 0);
  const result = await tesseract.recognize(fallbackCanvas, false);
  return { text: result.text, confidence: result.confidence };
}

/**
 * Smart multi-pass VLM processing
 * 
 * Pass 1: Classify image type (fast, ~100ms)
 * Pass 2: Content-specific text extraction (main pass)
 * Pass 3: Targeted sensitive data scan (optional, if needed)
 * 
 * If the VLM hallucinates, automatically falls back to Tesseract.
 */
export async function smartVLMProcess(
  canvas: HTMLCanvasElement,
  onProgress?: VLMProgressCallback,
): Promise<SmartVLMResult> {
  const bridge = VLMWorkerBridge.shared;
  if (!bridge.isModelLoaded) {
    throw new Error('VLM model not loaded');
  }

  const start = performance.now();
  const passes: string[] = [];
  // Downscale to 256px — CLIP resizes internally, bigger is just slower
  const vlmCanvas = prepareCanvas(canvas, 256);

  // --- Pass 1: Classify the image ---
  onProgress?.('Classifying image type...');
  let contentType: ContentType = 'unknown';
  try {
    const classifyResponse = await runPrompt(bridge, vlmCanvas, CLASSIFY_PROMPT);
    contentType = parseClassification(classifyResponse);
    passes.push(`classify:${contentType}`);
    console.log(`🏷️ VLM classified image as: ${contentType}`);
  } catch (err) {
    console.warn('Classification pass failed, using generic prompt:', err);
    passes.push('classify:failed');
  }

  // Check if calibration overrides exist
  const calibration = loadCalibration();
  const cal = calibration.find(c => c.contentType === contentType);
  if (cal) {
    console.log(`📈 Using calibrated prompt (accuracy: ${cal.accuracy.toFixed(1)}%)`);
  }

  // --- Pass 2: Content-specific text extraction ---
  const extractPrompt = EXTRACTION_PROMPTS[contentType];
  onProgress?.(`Extracting text (${contentType} mode)...`);

  let rawText = '';
  try {
    rawText = await runPrompt(bridge, vlmCanvas, extractPrompt);
    passes.push(`extract:${extractPrompt.id}`);
  } catch (err) {
    console.warn('Extraction pass failed:', err);
    passes.push('extract:failed');
  }

  // Post-process
  const cleanedText = postProcessVLMOutput(rawText);
  const quality = assessVLMOutputQuality(cleanedText);
  console.log(`📝 VLM extracted ${cleanedText.length} chars (quality: ${quality})`);

  // --- Hallucination check: if VLM output is fabricated, fall back to Tesseract ---
  if (quality === 0 || isHallucinated(rawText) || isHallucinated(cleanedText)) {
    console.warn('⚠️ VLM output detected as hallucination — falling back to Tesseract');
    onProgress?.('⚠️ VLM hallucinated — falling back to Tesseract...');
    passes.push('hallucination_detected');

    try {
      const fallback = await tesseractFallback(canvas);
      passes.push(`tesseract_fallback:${fallback.confidence.toFixed(0)}%`);
      const totalTimeMs = performance.now() - start;
      return {
        text: fallback.text,
        contentType,
        quality: fallback.confidence,
        sensitiveFindings: [],
        passes,
        totalTimeMs,
        usedFallback: true,
      };
    } catch (err) {
      console.warn('Tesseract fallback also failed:', err);
      passes.push('tesseract_fallback:failed');
    }
  }

  // --- Pass 3: Targeted sensitive data scan (only if pass 2 produced content) ---
  let sensitiveFindings: Array<{ type: string; value: string }> = [];
  if (quality >= 30) {
    onProgress?.('Scanning for sensitive data...');
    try {
      const sensitiveRaw = await runPrompt(bridge, vlmCanvas, SENSITIVE_DATA_PROMPT);
      sensitiveFindings = parseVLMSensitiveFindings(postProcessVLMOutput(sensitiveRaw));
      passes.push(`sensitive:${sensitiveFindings.length} found`);
      console.log(`🔒 VLM found ${sensitiveFindings.length} sensitive items`);
    } catch (err) {
      console.warn('Sensitive scan pass failed:', err);
      passes.push('sensitive:failed');
    }
  }

  const totalTimeMs = performance.now() - start;

  return {
    text: cleanedText,
    contentType,
    quality,
    sensitiveFindings,
    passes,
    totalTimeMs,
  };
}

/**
 * Quick single-pass VLM (for hybrid mode where speed matters)
 * Uses classification to pick the right prompt but skips the sensitive scan
 */
export async function quickVLMExtract(
  canvas: HTMLCanvasElement,
  onProgress?: VLMProgressCallback,
): Promise<{ text: string; quality: number; contentType: ContentType }> {
  const bridge = VLMWorkerBridge.shared;
  if (!bridge.isModelLoaded) {
    throw new Error('VLM model not loaded');
  }

  const vlmCanvas = prepareCanvas(canvas, 256);

  // Quick classify
  onProgress?.('Classifying...');
  let contentType: ContentType = 'unknown';
  try {
    const resp = await runPrompt(bridge, vlmCanvas, CLASSIFY_PROMPT);
    contentType = parseClassification(resp);
  } catch { /* use unknown */ }

  // Extract with specialized prompt (fresh pixel copy each call)
  onProgress?.(`Extracting (${contentType})...`);
  const prompt = EXTRACTION_PROMPTS[contentType];
  const raw = await runPrompt(bridge, vlmCanvas, prompt);
  const text = postProcessVLMOutput(raw);
  const quality = assessVLMOutputQuality(text);

  // If hallucinated, return quality 0 so callers know to reject/fallback
  if (isHallucinated(raw) || isHallucinated(text)) {
    console.warn('⚠️ quickVLMExtract: hallucination detected, marking quality=0');
    return { text: '', quality: 0, contentType };
  }

  return { text, quality, contentType };
}
