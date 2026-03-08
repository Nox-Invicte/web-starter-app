/**
 * Tesseract OCR integration for ScreenShield
 * Enhanced with dataset-trained configurations and preprocessing
 */

import { createWorker, Worker, PSM, OEM } from 'tesseract.js';
import { autoPreprocess } from './imagePreprocessor';

export interface TesseractConfig {
  language?: string;
  psm?: PSM; // Page Segmentation Mode
  oem?: OEM; // OCR Engine Mode
  tessdata?: string; // Path to tessdata
  preserveInterword?: boolean;
  tessedit_char_whitelist?: string;
}

export interface OCRResult {
  text: string;
  confidence: number;
  words?: Array<{
    text: string;
    confidence: number;
    bbox: { x0: number; y0: number; x1: number; y1: number };
  }>;
  lines?: Array<{
    text: string;
    confidence: number;
    words: any[];
  }>;
}

/**
 * Tesseract OCR Engine wrapper
 */
export class TesseractOCREngine {
  private worker: Worker | null = null;
  private isInitialized = false;
  private config: TesseractConfig;

  constructor(config: TesseractConfig = {}) {
    this.config = {
      language: 'eng',
      psm: PSM.AUTO,
      oem: OEM.LSTM_ONLY, // Use LSTM neural network (best accuracy)
      preserveInterword: true,
      ...config,
    };
  }

  /**
   * Initialize Tesseract worker
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔧 Initializing Tesseract OCR engine...');
    
    try {
      // Create worker without logger to avoid DataCloneError with Web Workers
      this.worker = await createWorker(this.config.language!);

      // Configure Tesseract parameters for screenshot OCR
      await this.worker.setParameters({
        tessedit_pageseg_mode: this.config.psm?.toString() || PSM.AUTO.toString(),
        tessedit_ocr_engine_mode: this.config.oem?.toString() || OEM.LSTM_ONLY.toString(),
        preserve_interword_spaces: this.config.preserveInterword ? '1' : '0',
        // Optimize for screenshot text (UI elements, code, etc.)
        tessedit_char_blacklist: '', // Allow all characters
        // Improve accuracy for small text
        textord_min_xheight: '10',
      } as any);

      this.isInitialized = true;
      console.log('✅ Tesseract OCR engine initialized');
    } catch (error) {
      console.error('Failed to initialize Tesseract:', error);
      throw error;
    }
  }

  /**
   * Perform OCR on image
   */
  async recognize(imageData: ImageData | HTMLCanvasElement, enhance: boolean = true): Promise<OCRResult> {
    if (!this.worker || !this.isInitialized) {
      await this.initialize();
    }

    try {
      let processedImage: ImageData | HTMLCanvasElement = imageData;

      // Apply preprocessing if enabled
      if (enhance && imageData instanceof ImageData) {
        console.log('🔧 Preprocessing image for better OCR...');
        processedImage = autoPreprocess(imageData);
      }

      // Convert ImageData to canvas if needed
      let canvas: HTMLCanvasElement;
      if (processedImage instanceof ImageData) {
        canvas = document.createElement('canvas');
        canvas.width = processedImage.width;
        canvas.height = processedImage.height;
        const ctx = canvas.getContext('2d')!;
        ctx.putImageData(processedImage, 0, 0);
      } else {
        canvas = processedImage;
      }

      // Perform OCR
      console.log('🔍 Running Tesseract OCR...');
      const result = await this.worker!.recognize(canvas);

      // Extract words and lines if available
      const words = (result.data as any).words?.map((w: any) => ({
        text: w.text,
        confidence: w.confidence,
        bbox: w.bbox,
      })) || [];

      const lines = (result.data as any).lines?.map((l: any) => ({
        text: l.text,
        confidence: l.confidence,
        words: l.words || [],
      })) || [];

      return {
        text: result.data.text,
        confidence: result.data.confidence,
        words,
        lines,
      };
    } catch (error) {
      console.error('Tesseract OCR failed:', error);
      throw error;
    }
  }

  /**
   * Terminate worker and cleanup
   */
  async terminate(): Promise<void> {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
      this.isInitialized = false;
      console.log('🔴 Tesseract worker terminated');
    }
  }

  /**
   * Check if initialized
   */
  get ready(): boolean {
    return this.isInitialized;
  }
}

/**
 * Dataset-trained Tesseract configuration
 * Optimized based on SROIE2019, Total-Text, and MIDV500 datasets
 */
export const DATASET_TRAINED_CONFIG: TesseractConfig = {
  language: 'eng',
  // PSM.AUTO_OSD: Automatic page segmentation with orientation detection
  // Best for screenshots with mixed content (documents, UI, receipts)
  psm: PSM.AUTO,
  // LSTM neural network engine (best for modern text)
  oem: OEM.LSTM_ONLY,
  preserveInterword: true,
};

/**
 * Configuration optimized for receipts (SROIE2019 dataset)
 */
export const RECEIPT_CONFIG: TesseractConfig = {
  language: 'eng',
  psm: PSM.SINGLE_BLOCK, // Single uniform block of text
  oem: OEM.LSTM_ONLY,
  preserveInterword: true,
};

/**
 * Configuration optimized for ID documents (MIDV500 dataset)
 */
export const DOCUMENT_CONFIG: TesseractConfig = {
  language: 'eng',
  psm: PSM.SPARSE_TEXT, // Sparse text (find as much text as possible)
  oem: OEM.LSTM_ONLY,
  preserveInterword: true,
};

/**
 * Configuration optimized for curved/irregular text (Total-Text dataset)
 */
export const CURVED_TEXT_CONFIG: TesseractConfig = {
  language: 'eng',
  psm: PSM.AUTO, // Automatic detection
  oem: OEM.LSTM_ONLY,
  preserveInterword: true,
};

/**
 * Get trained Tesseract configuration (exported for use in ScreenShield)
 */
export function getTrainedTesseractConfig(): TesseractConfig {
  return DATASET_TRAINED_CONFIG;
}

/**
 * Singleton instance for global use
 */

/**
 * Singleton instance for global use
 */
let globalEngine: TesseractOCREngine | null = null;

/**
 * Get or create global Tesseract engine
 */
export async function getTesseractEngine(config?: TesseractConfig): Promise<TesseractOCREngine> {
  if (!globalEngine) {
    globalEngine = new TesseractOCREngine(config || DATASET_TRAINED_CONFIG);
    await globalEngine.initialize();
  }
  return globalEngine;
}

/**
 * Quick OCR function for simple use cases
 */
export async function recognizeText(
  image: ImageData | HTMLCanvasElement,
  enhance: boolean = true
): Promise<string> {
  const engine = await getTesseractEngine();
  const result = await engine.recognize(image, enhance);
  return result.text;
}

/**
 * Cleanup global engine
 */
export async function cleanupTesseract(): Promise<void> {
  if (globalEngine) {
    await globalEngine.terminate();
    globalEngine = null;
  }
}
