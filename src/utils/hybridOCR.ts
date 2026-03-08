/**
 * Hybrid OCR Engine - Combines Tesseract and Smart VLM for enhanced accuracy
 * 
 * Strategy:
 * 1. Primary: Use Tesseract (fast, trained on datasets)
 * 2. Validation: Use Smart VLM (multi-pass, adaptive prompts) to verify/enhance
 * 3. Fusion: Word-level alignment merge for maximum accuracy
 */

import { getTesseractEngine, getTrainedTesseractConfig, type OCRResult } from './tesseractOCR';
import { autoPreprocess } from './imagePreprocessor';
import { quickVLMExtract } from './smartVLM';
import { postProcessVLMOutput } from './vlmPostProcessor';

export interface HybridOCRResult {
  text: string;
  confidence: number;
  method: 'tesseract' | 'vlm' | 'hybrid';
  tesseractResult?: OCRResult;
  vlmResult?: string;
  fused: boolean;
  contentType?: string;
}

export interface HybridOCRConfig {
  confidenceThreshold: number; // Use VLM if Tesseract confidence below this
  enableFusion: boolean; // Combine both results
  vlmValidation: boolean; // Use VLM to validate Tesseract
  preprocessEnabled: boolean;
}

const DEFAULT_CONFIG: HybridOCRConfig = {
  confidenceThreshold: 75, // Validate if confidence < 75%
  enableFusion: true,
  vlmValidation: true,
  preprocessEnabled: true,
};

/**
 * Hybrid OCR Engine class
 */
export class HybridOCREngine {
  private config: HybridOCRConfig;
  
  constructor(config: Partial<HybridOCRConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }
  
  /**
   * Process image with hybrid OCR approach (now using Smart VLM)
   */
  async recognize(
    canvas: HTMLCanvasElement,
    vlmAvailable: boolean = true
  ): Promise<HybridOCRResult> {
    console.log('🔬 Starting Hybrid OCR (Tesseract + Smart VLM)...');
    
    // Stage 1: Tesseract OCR (primary engine)
    console.log('  📝 Stage 1: Running Tesseract OCR...');
    const tesseract = await getTesseractEngine(getTrainedTesseractConfig());
    
    if (this.config.preprocessEnabled) {
      const ctx = canvas.getContext('2d')!;
      const imageData = autoPreprocess(ctx.getImageData(0, 0, canvas.width, canvas.height));
      ctx.putImageData(imageData, 0, 0);
    }
    
    const tesseractResult = await tesseract.recognize(canvas, !this.config.preprocessEnabled);
    console.log(`  ✅ Tesseract: confidence=${tesseractResult.confidence.toFixed(1)}%`);
    
    // Determine if VLM validation is needed
    const needsValidation = vlmAvailable && 
                           this.config.vlmValidation && 
                           tesseractResult.confidence < this.config.confidenceThreshold;
    
    if (!needsValidation) {
      return {
        text: tesseractResult.text,
        confidence: tesseractResult.confidence,
        method: 'tesseract',
        tesseractResult,
        fused: false,
      };
    }
    
    // Stage 2: Smart VLM validation (adaptive prompts, post-processing)
    console.log('  🧠 Stage 2: Running Smart VLM validation...');
    try {
      const vlmResult = await quickVLMExtract(canvas);
      const vlmText = vlmResult.text;
      console.log(`  ✅ Smart VLM: ${vlmText.length} chars, quality=${vlmResult.quality}, type=${vlmResult.contentType}`);

      // If VLM hallucinated (quality 0 or empty text), discard it and keep Tesseract only
      if (vlmResult.quality === 0 || !vlmText.trim()) {
        console.warn('  ⚠️ VLM output is hallucinated or empty — using Tesseract only');
        return {
          text: tesseractResult.text,
          confidence: tesseractResult.confidence,
          method: 'tesseract',
          tesseractResult,
          fused: false,
        };
      }
      
      // Stage 3: Smart Fusion
      if (this.config.enableFusion) {
        console.log('  🔗 Stage 3: Smart fusion (word-level alignment)...');
        const fusedText = smartFuseResults(tesseractResult.text, vlmText, tesseractResult.confidence, vlmResult.quality);
        const fusedConfidence = calculateFusedConfidence(
          tesseractResult.confidence,
          tesseractResult.text,
          vlmText
        );
        
        return {
          text: fusedText,
          confidence: fusedConfidence,
          method: 'hybrid',
          tesseractResult,
          vlmResult: vlmText,
          fused: true,
          contentType: vlmResult.contentType,
        };
      }
      
      return {
        text: vlmText,
        confidence: Math.max(vlmResult.quality, 70),
        method: 'vlm',
        tesseractResult,
        vlmResult: vlmText,
        fused: false,
        contentType: vlmResult.contentType,
      };
      
    } catch (error) {
      console.warn('Smart VLM validation failed:', error);
      return {
        text: tesseractResult.text,
        confidence: tesseractResult.confidence,
        method: 'tesseract',
        tesseractResult,
        fused: false,
      };
    }
  }
}

/**
 * Smart fusion: word-level alignment and merge
 * Uses word overlap to find matching lines, then picks the best version of each
 */
function smartFuseResults(
  tesseractText: string,
  vlmText: string,
  tessConfidence: number,
  vlmQuality: number,
): string {
  const tessLines = tesseractText.split('\n').filter(l => l.trim());
  const vlmLines = vlmText.split('\n').filter(l => l.trim());

  // If one source is empty, use the other
  if (tessLines.length === 0) return vlmText;
  if (vlmLines.length === 0) return tesseractText;

  // Build word sets per line for matching
  const tessWordSets = tessLines.map(l => new Set(l.toLowerCase().split(/\s+/).filter(Boolean)));
  const vlmWordSets = vlmLines.map(l => new Set(l.toLowerCase().split(/\s+/).filter(Boolean)));

  const fused: string[] = [];
  const usedVlmLines = new Set<number>();

  for (let ti = 0; ti < tessLines.length; ti++) {
    // Find the VLM line that best matches this Tesseract line
    let bestVlmIdx = -1;
    let bestOverlap = 0;

    for (let vi = 0; vi < vlmLines.length; vi++) {
      if (usedVlmLines.has(vi)) continue;
      const overlap = wordOverlap(tessWordSets[ti], vlmWordSets[vi]);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestVlmIdx = vi;
      }
    }

    if (bestVlmIdx >= 0 && bestOverlap >= 0.3) {
      usedVlmLines.add(bestVlmIdx);
      // Pick the better version of this line
      fused.push(pickBetterLine(tessLines[ti], vlmLines[bestVlmIdx], tessConfidence, vlmQuality));
    } else {
      // No good VLM match — keep Tesseract line
      fused.push(tessLines[ti]);
    }
  }

  // Add any VLM-only lines that had no Tesseract match (new content VLM found)
  for (let vi = 0; vi < vlmLines.length; vi++) {
    if (!usedVlmLines.has(vi) && vlmLines[vi].trim().length > 3) {
      fused.push(vlmLines[vi]);
    }
  }

  return fused.join('\n');
}

/**
 * Pick the better version of a line from two sources
 */
function pickBetterLine(tessLine: string, vlmLine: string, tessConf: number, vlmQuality: number): string {
  const tessLen = tessLine.trim().length;
  const vlmLen = vlmLine.trim().length;

  // If one is much longer, it likely captured more text
  if (vlmLen > tessLen * 1.4) return vlmLine;
  if (tessLen > vlmLen * 1.4) return tessLine;

  // If similar length, pick based on confidence/quality
  if (tessConf >= 80) return tessLine;
  if (vlmQuality >= 75) return vlmLine;

  // Default: pick the one with more alphanumeric content (less garbled)
  const tessAlpha = (tessLine.match(/[a-zA-Z0-9]/g) || []).length;
  const vlmAlpha = (vlmLine.match(/[a-zA-Z0-9]/g) || []).length;
  return vlmAlpha > tessAlpha ? vlmLine : tessLine;
}

/**
 * Word overlap ratio between two word sets (Jaccard on intersection / min size)
 */
function wordOverlap(set1: Set<string>, set2: Set<string>): number {
  if (set1.size === 0 || set2.size === 0) return 0;
  let intersection = 0;
  for (const w of set1) {
    if (set2.has(w)) intersection++;
  }
  return intersection / Math.min(set1.size, set2.size);
}

/**
 * Calculate confidence for fused result
 */
function calculateFusedConfidence(
  tesseractConf: number,
  tesseractText: string,
  vlmText: string
): number {
  const similarity = calculateTextSimilarity(tesseractText, vlmText);
  
  // High similarity = both agree = high confidence
  if (similarity > 0.8) {
    return Math.max(tesseractConf, 92);
  } else if (similarity > 0.6) {
    return Math.max((tesseractConf + 88) / 2, 80);
  } else if (similarity > 0.3) {
    return Math.max(tesseractConf, 75);
  } else {
    // Very different — take the higher of the two sources
    return Math.max(tesseractConf, 70);
  }
}

/**
 * Calculate text similarity (Jaccard on words)
 */
function calculateTextSimilarity(text1: string, text2: string): number {
  const words1 = text1.toLowerCase().split(/\s+/).filter(Boolean);
  const words2 = text2.toLowerCase().split(/\s+/).filter(Boolean);
  
  if (words1.length === 0 && words2.length === 0) return 1;
  if (words1.length === 0 || words2.length === 0) return 0;
  
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  
  return intersection.size / union.size;
}

/**
 * Get singleton hybrid OCR engine
 */
let hybridEngine: HybridOCREngine | null = null;

export function getHybridOCREngine(config?: Partial<HybridOCRConfig>): HybridOCREngine {
  if (!hybridEngine || config) {
    hybridEngine = new HybridOCREngine(config);
  }
  return hybridEngine;
}
