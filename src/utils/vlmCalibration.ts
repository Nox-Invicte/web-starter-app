/**
 * VLM Calibration System
 * 
 * Uses the SROIE2019 dataset to evaluate VLM prompt effectiveness
 * and find the best prompting strategy. Results are saved to localStorage
 * so the VLM gets "smarter" after calibration.
 */

import { VLMWorkerBridge } from '@runanywhere/web-llamacpp';
import { loadSROIEDataset, type DatasetSample } from './datasetLoader';
import { autoPreprocess } from './imagePreprocessor';
import {
  EXTRACTION_PROMPTS,
  saveCalibration,
  type ContentType,
  type PromptCalibration,
} from './vlmPrompts';
import { postProcessVLMOutput, assessVLMOutputQuality } from './vlmPostProcessor';

export interface CalibrationProgress {
  stage: string;
  currentSample: number;
  totalSamples: number;
  currentPrompt: string;
  accuracy: number;
}

export type CalibrationProgressCallback = (progress: CalibrationProgress) => void;

export interface CalibrationResult {
  calibrations: PromptCalibration[];
  overallAccuracy: number;
  samplesEvaluated: number;
  bestContentType: ContentType;
  bestPromptId: string;
  timeMs: number;
}

/**
 * Levenshtein-based text similarity (0-100)
 */
function textAccuracy(predicted: string, groundTruth: string): number {
  const pred = predicted.toLowerCase().trim();
  const truth = groundTruth.toLowerCase().trim();
  if (pred === truth) return 100;
  if (!pred || !truth) return 0;

  // Word overlap approach (faster and more meaningful for OCR)
  const predWords = new Set(pred.split(/\s+/).filter(Boolean));
  const truthWords = new Set(truth.split(/\s+/).filter(Boolean));
  if (truthWords.size === 0) return predWords.size === 0 ? 100 : 0;

  let matches = 0;
  for (const w of truthWords) {
    if (predWords.has(w)) matches++;
  }

  return (matches / truthWords.size) * 100;
}

/**
 * Load an image from a path into an HTMLImageElement
 */
function loadImage(path: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const timeout = setTimeout(() => resolve(null), 10000);
    img.onload = () => { clearTimeout(timeout); resolve(img); };
    img.onerror = () => { clearTimeout(timeout); resolve(null); };
    img.src = path;
  });
}

/**
 * Run VLM calibration against the SROIE2019 dataset.
 * Tests the receipt prompt (since SROIE is receipts) and the generic prompt,
 * measures accuracy, and saves the best configuration.
 */
export async function calibrateVLM(
  sampleCount: number = 5,
  onProgress?: CalibrationProgressCallback,
): Promise<CalibrationResult> {
  const start = performance.now();

  const bridge = VLMWorkerBridge.shared;
  if (!bridge.isModelLoaded) {
    throw new Error('VLM model not loaded. Please load the Vision Model first.');
  }

  // Load dataset
  onProgress?.({ stage: 'Loading dataset...', currentSample: 0, totalSamples: sampleCount, currentPrompt: '', accuracy: 0 });
  const samples = await loadSROIEDataset(sampleCount);
  if (samples.length === 0) {
    throw new Error('No dataset samples loaded for calibration.');
  }

  // Test two prompts: receipt-specific and generic
  const promptsToTest: { type: ContentType; id: string }[] = [
    { type: 'receipt', id: EXTRACTION_PROMPTS.receipt.id },
    { type: 'unknown', id: EXTRACTION_PROMPTS.unknown.id },
  ];

  const results: Map<string, { totalAccuracy: number; totalQuality: number; count: number }> = new Map();

  for (const promptInfo of promptsToTest) {
    const prompt = EXTRACTION_PROMPTS[promptInfo.type];
    results.set(promptInfo.id, { totalAccuracy: 0, totalQuality: 0, count: 0 });

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i];

      onProgress?.({
        stage: `Testing ${promptInfo.type} prompt`,
        currentSample: i + 1,
        totalSamples: samples.length,
        currentPrompt: promptInfo.id,
        accuracy: results.get(promptInfo.id)!.count > 0
          ? results.get(promptInfo.id)!.totalAccuracy / results.get(promptInfo.id)!.count
          : 0,
      });

      try {
        // Load image
        const img = await loadImage(sample.imagePath);
        if (!img) continue;

        // Prepare canvas (downscale to 256 — CLIP resizes internally)
        const canvas = document.createElement('canvas');
        const scale = Math.min(256 / img.width, 256 / img.height);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Preprocess
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        imageData = autoPreprocess(imageData);
        ctx.putImageData(imageData, 0, 0);

        // Extract fresh RGB pixels (buffer gets transferred/detached by bridge.process)
        const finalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const rgbPixels = new Uint8Array(canvas.width * canvas.height * 3);
        for (let p = 0; p < finalData.data.length / 4; p++) {
          rgbPixels[p * 3] = finalData.data[p * 4];
          rgbPixels[p * 3 + 1] = finalData.data[p * 4 + 1];
          rgbPixels[p * 3 + 2] = finalData.data[p * 4 + 2];
        }

        // Run VLM
        const response = await bridge.process(
          rgbPixels, canvas.width, canvas.height,
          prompt.prompt,
          { maxTokens: prompt.maxTokens, temperature: prompt.temperature },
        );

        const cleaned = postProcessVLMOutput(response.text);
        const quality = assessVLMOutputQuality(cleaned);
        const accuracy = textAccuracy(cleaned, sample.text);

        const r = results.get(promptInfo.id)!;
        r.totalAccuracy += accuracy;
        r.totalQuality += quality;
        r.count++;

        console.log(`  ✓ [${promptInfo.type}] Sample ${i + 1}: accuracy=${accuracy.toFixed(1)}%, quality=${quality}`);
      } catch (err) {
        console.warn(`  ✗ [${promptInfo.type}] Sample ${i + 1} failed:`, err);
      }
    }
  }

  // Determine best prompt
  let bestId = '';
  let bestType: ContentType = 'unknown';
  let bestAccuracy = 0;

  const calibrations: PromptCalibration[] = [];

  for (const [id, r] of results) {
    const avgAccuracy = r.count > 0 ? r.totalAccuracy / r.count : 0;
    const type = promptsToTest.find(p => p.id === id)!.type;

    calibrations.push({
      contentType: type,
      bestPromptId: id,
      accuracy: avgAccuracy,
      samplesEvaluated: r.count,
      calibratedAt: new Date().toISOString(),
    });

    if (avgAccuracy > bestAccuracy) {
      bestAccuracy = avgAccuracy;
      bestId = id;
      bestType = type;
    }
  }

  // Save
  saveCalibration(calibrations);

  const totalSamples = [...results.values()].reduce((s, r) => s + r.count, 0);
  const overallAccuracy = totalSamples > 0
    ? [...results.values()].reduce((s, r) => s + r.totalAccuracy, 0) / totalSamples
    : 0;

  const timeMs = performance.now() - start;

  console.log(`\n✅ VLM Calibration complete in ${(timeMs / 1000).toFixed(1)}s`);
  console.log(`📈 Best prompt: ${bestId} (${bestType}) — ${bestAccuracy.toFixed(1)}% accuracy`);

  return {
    calibrations,
    overallAccuracy,
    samplesEvaluated: totalSamples,
    bestContentType: bestType,
    bestPromptId: bestId,
    timeMs,
  };
}
