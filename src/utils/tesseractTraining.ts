/**
 * Dataset training pipeline for Tesseract OCR
 * Uses SROIE2019, Total-Text, and MIDV500 datasets to optimize OCR accuracy
 */

import { TesseractOCREngine, TesseractConfig } from './tesseractOCR';
import { PSM, OEM } from 'tesseract.js';
import { autoPreprocess } from './imagePreprocessor';
import { loadSROIEDataset, type DatasetSample } from './datasetLoader';

export type TrainingMode = 'quick' | 'balanced' | 'full';

export interface TrainingModeConfig {
  mode: TrainingMode;
  label: string;
  description: string;
  configs: number;
  samples: number;
  estimatedTime: string;
}

export const TRAINING_MODES: Record<TrainingMode, TrainingModeConfig> = {
  quick: {
    mode: 'quick',
    label: 'Quick Training',
    description: '2 configs × 5 images = 10 operations',
    configs: 2,
    samples: 5,
    estimatedTime: '~30 seconds',
  },
  balanced: {
    mode: 'balanced',
    label: 'Balanced Training',
    description: '3 configs × 10 images = 30 operations',
    configs: 3,
    samples: 10,
    estimatedTime: '~1-2 minutes',
  },
  full: {
    mode: 'full',
    label: 'Full Training',
    description: '4 configs × 20 images = 80 operations',
    configs: 4,
    samples: 20,
    estimatedTime: '~3-5 minutes',
  },
};

export interface TrainingResult {
  accuracy: number;
  avgConfidence: number;
  sampleCount: number;
  bestConfig: TesseractConfig;
  configResults: Array<{
    config: TesseractConfig;
    accuracy: number;
    confidence: number;
  }>;
  errors: string[];
  processedSamples: number;
}

export interface TrainingProgress {
  stage: string;
  currentConfig: number;
  totalConfigs: number;
  currentSample: number;
  totalSamples: number;
  accuracy: number;
  confidence: number;
}

export type ProgressCallback = (progress: TrainingProgress) => void;

/**
 * Calculate text similarity (Levenshtein distance based)
 */
function calculateAccuracy(predicted: string, groundTruth: string): number {
  const pred = predicted.toLowerCase().trim();
  const truth = groundTruth.toLowerCase().trim();
  
  if (pred === truth) return 100;
  
  const distance = levenshteinDistance(pred, truth);
  const maxLen = Math.max(pred.length, truth.length);
  
  return maxLen > 0 ? ((maxLen - distance) / maxLen) * 100 : 0;
}

/**
 * Levenshtein distance algorithm
 */
function levenshteinDistance(str1: string, str2: string): number {
  const matrix: number[][] = [];
  
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  
  return matrix[str2.length][str1.length];
}

/**
 * Test configurations against dataset samples with real image processing
 */
export async function testConfigurations(
  samples: DatasetSample[],
  configs: TesseractConfig[],
  onProgress?: ProgressCallback
): Promise<TrainingResult> {
  console.log(`🎓 Training Tesseract on ${samples.length} real dataset images...`);
  console.log(`🎓 Received ${configs.length} configurations to test`);
  
  const configResults: Array<{
    config: TesseractConfig;
    accuracy: number;
    confidence: number;
  }> = [];
  
  const errors: string[] = [];
  let processedSamples = 0;
  
  console.log(`🔄 Starting configuration testing loop...`);
  
  try {
    for (let configIdx = 0; configIdx < configs.length; configIdx++) {
      const config = configs[configIdx];
      console.log(`\n📋 Testing config ${configIdx + 1}/${configs.length}: PSM=${config.psm}, OEM=${config.oem}`);
      
      const engine = new TesseractOCREngine(config);
      console.log(`  ⏳ Initializing Tesseract engine...`);
      await engine.initialize();
      console.log(`  ✅ Engine initialized`);
    
    let totalAccuracy = 0;
    let totalConfidence = 0;
    let successCount = 0;
    
    for (let sampleIdx = 0; sampleIdx < samples.length; sampleIdx++) {
      const sample = samples[sampleIdx];
      
      try {
        // Report progress
        if (onProgress) {
          onProgress({
            stage: 'processing',
            currentConfig: configIdx + 1,
            totalConfigs: configs.length,
            currentSample: sampleIdx + 1,
            totalSamples: samples.length,
            accuracy: successCount > 0 ? totalAccuracy / successCount : 0,
            confidence: successCount > 0 ? totalConfidence / successCount : 0,
          });
        }
        
        // Load image from dataset
        console.log(`  📸 Loading image: ${sample.imagePath}`);
        const img = await loadImageFromPath(sample.imagePath);
        if (!img) {
          const errMsg = `Failed to load image: ${sample.imagePath}`;
          console.error(`  ❌ ${errMsg}`);
          errors.push(errMsg);
          continue;
        }
        console.log(`  ✅ Image loaded: ${img.width}x${img.height}`);
        
        // Convert to canvas
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        
        // Apply preprocessing
        let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        imageData = autoPreprocess(imageData);
        ctx.putImageData(imageData, 0, 0);
        
        // Run OCR
        const result = await engine.recognize(canvas, false);
        
        // Calculate accuracy against ground truth
        const accuracy = calculateAccuracy(result.text, sample.text);
        
        totalAccuracy += accuracy;
        totalConfidence += result.confidence;
        successCount++;
        processedSamples++;
        
        if (sampleIdx % 5 === 0) {
          console.log(`  ✓ Sample ${sampleIdx + 1}/${samples.length}: Accuracy=${accuracy.toFixed(1)}%, Conf=${result.confidence.toFixed(1)}%`);
        }
      } catch (error) {
        errors.push(`Failed on ${sample.imagePath}: ${error}`);
      }
    }
    
    await engine.terminate();
    
    const avgAccuracy = successCount > 0 ? totalAccuracy / successCount : 0;
    const avgConfidence = successCount > 0 ? totalConfidence / successCount : 0;
    
    configResults.push({
      config,
      accuracy: avgAccuracy,
      confidence: avgConfidence,
    });
    
    console.log(`  ✅ Config result: Accuracy=${avgAccuracy.toFixed(2)}%, Confidence=${avgConfidence.toFixed(2)}%`);
  }
  } catch (error) {
    console.error('❌ Fatal error in training loop:', error);
    throw error;
  }
  
  console.log('✅ Configuration testing complete');
  
  // Find best configuration
  const best = configResults.reduce((a, b) => 
    (a.accuracy + a.confidence) > (b.accuracy + b.confidence) ? a : b
  );
  
  return {
    accuracy: best.accuracy,
    avgConfidence: best.confidence,
    sampleCount: samples.length,
    bestConfig: best.config,
    configResults,
    errors,
    processedSamples,
  };
}

/**
 * Load image from path (dataset folder)
 */
async function loadImageFromPath(path: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    // Add timeout to prevent hanging
    const timeout = setTimeout(() => {
      console.warn(`Image load timeout: ${path}`);
      resolve(null);
    }, 10000); // 10 second timeout
    
    img.onload = () => {
      clearTimeout(timeout);
      resolve(img);
    };
    img.onerror = () => {
      clearTimeout(timeout);
      console.warn(`Failed to load image: ${path}`);
      resolve(null);
    };
    
    img.src = path;
  });
}

/**
 * Run comprehensive training pipeline with real dataset images
 */
export async function trainTesseractOnDatasets(
  mode: TrainingMode = 'balanced',
  onProgress?: ProgressCallback
): Promise<TrainingResult> {
  const config = TRAINING_MODES[mode];
  
  console.log(`🚀 Starting Tesseract training pipeline [${config.label}]...`);
  console.log(`📊 ${config.description} - Expected time: ${config.estimatedTime}`);
  console.log('📊 Using datasets: SROIE2019 (receipt images)');
  
  // Load real training samples from SROIE2019 dataset with progress
  console.log(`📦 Loading ${config.samples} SROIE2019 dataset samples...`);
  
  // Report loading progress
  if (onProgress) {
    onProgress({
      stage: 'loading dataset',
      currentConfig: 0,
      totalConfigs: config.configs,
      currentSample: 0,
      totalSamples: config.samples,
      accuracy: 0,
      confidence: 0,
    });
  }
  
  const samples = await loadSROIEDataset(config.samples, (current, total) => {
    if (onProgress) {
      onProgress({
        stage: 'loading dataset',
        currentConfig: 0,
        totalConfigs: config.configs,
        currentSample: current,
        totalSamples: total,
        accuracy: 0,
        confidence: 0,
      });
    }
  });
  
  if (samples.length === 0) {
    throw new Error(
      'No training samples loaded from dataset. ' +
      'Please ensure the dataset folder is accessible at /dataset/SROIE2019/train/ ' +
      'and that Vite dev server has been restarted after configuration changes.'
    );
  }
  
  console.log(`✅ Loaded ${samples.length} training samples with ground truth`);
  
  // Test different OCR configurations optimized for different document types
  const configs: TesseractConfig[] = [
    // Auto configuration (baseline)
    { 
      language: 'eng', 
      psm: PSM.AUTO, 
      oem: OEM.LSTM_ONLY, 
      preserveInterword: true 
    },
    
    // Receipt-optimized (SROIE2019 - structured blocks)
    { 
      language: 'eng', 
      psm: PSM.SINGLE_BLOCK, 
      oem: OEM.LSTM_ONLY, 
      preserveInterword: true 
    },
    
    // Sparse text (best for screenshots with UI elements)
    { 
      language: 'eng', 
      psm: PSM.SPARSE_TEXT, 
      oem: OEM.LSTM_ONLY, 
      preserveInterword: true 
    },
    
    // Single column (vertical layouts)
    { 
      language: 'eng', 
      psm: PSM.SINGLE_COLUMN, 
      oem: OEM.LSTM_ONLY, 
      preserveInterword: true 
    },
  ];
  
  // Select configurations based on mode
  const selectedConfigs = configs.slice(0, config.configs);
  
  console.log(`⚙️ Testing ${selectedConfigs.length} configurations on ${samples.length} samples...`);
  
  // Update progress before starting training
  if (onProgress) {
    onProgress({
      stage: 'initializing training',
      currentConfig: 0,
      totalConfigs: selectedConfigs.length,
      currentSample: 0,
      totalSamples: samples.length,
      accuracy: 0,
      confidence: 0,
    });
  }
  
  // Small delay to ensure UI updates
  await new Promise(resolve => setTimeout(resolve, 100));
  
  console.log('🚀 Starting testConfigurations...');
  
  // Run training with progress updates
  const result = await testConfigurations(samples, selectedConfigs, onProgress);
  
  // Save best configuration to localStorage for reuse
  saveTrainedConfig(result.bestConfig, result.accuracy, result.avgConfidence);
  
  console.log('\n✅ Training completed!');
  console.log(`📈 Best accuracy: ${result.accuracy.toFixed(2)}%`);
  console.log(`📊 Best confidence: ${result.avgConfidence.toFixed(2)}%`);
  console.log(`⚙️ Best config: PSM=${result.bestConfig.psm}, OEM=${result.bestConfig.oem}`);
  console.log(`📦 Processed ${result.processedSamples} images from dataset`);
  
  return result;
}

/**
 * Save trained configuration to localStorage
 */
function saveTrainedConfig(config: TesseractConfig, accuracy: number, confidence: number): void {
  try {
    const trainedConfig = {
      config,
      accuracy,
      confidence,
      trainedAt: new Date().toISOString(),
      version: '1.0.0',
    };
    
    localStorage.setItem('tesseract_trained_config', JSON.stringify(trainedConfig));
    console.log('💾 Saved trained configuration to localStorage');
  } catch (error) {
    console.warn('Failed to save trained config:', error);
  }
}

/**
 * Load trained configuration from localStorage or return default
 */
export function getTrainedTesseractConfig(): TesseractConfig {
  try {
    const saved = localStorage.getItem('tesseract_trained_config');
    
    if (saved) {
      const { config, accuracy, confidence, trainedAt } = JSON.parse(saved);
      console.log(`📖 Loaded trained config: Accuracy=${accuracy.toFixed(2)}%, Confidence=${confidence.toFixed(2)}% (trained ${new Date(trainedAt).toLocaleDateString()})`);
      return config;
    }
  } catch (error) {
    console.warn('Failed to load trained config:', error);
  }
  
  // Return default optimized config
  console.log('📖 Using default optimized configuration');
  return {
    language: 'eng',
    psm: PSM.SPARSE_TEXT, // Best for screenshots with mixed content
    oem: OEM.LSTM_ONLY, // Neural network engine
    preserveInterword: true,
  };
}

/**
 * Evaluate OCR performance metrics
 */
export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  characterErrorRate: number;
  wordErrorRate: number;
}

/**
 * Calculate comprehensive OCR metrics (for future use)
 */
export function calculateMetrics(predicted: string, groundTruth: string): PerformanceMetrics {
  const predWords = predicted.trim().split(/\s+/);
  const truthWords = groundTruth.trim().split(/\s+/);
  
  // Character Error Rate (CER)
  const charDistance = levenshteinDistance(predicted, groundTruth);
  const cer = (charDistance / groundTruth.length) * 100;
  
  // Word Error Rate (WER)
  const wordDistance = levenshteinDistance(predWords.join(' '), truthWords.join(' '));
  const wer = (wordDistance / truthWords.length) * 100;
  
  // Accuracy
  const accuracy = calculateAccuracy(predicted, groundTruth);
  
  // Simple precision/recall/F1 (word-based)
  const correctWords = predWords.filter(w => truthWords.includes(w)).length;
  const precision = predWords.length > 0 ? (correctWords / predWords.length) * 100 : 0;
  const recall = truthWords.length > 0 ? (correctWords / truthWords.length) * 100 : 0;
  const f1Score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  
  return {
    accuracy,
    precision,
    recall,
    f1Score,
    characterErrorRate: cer,
    wordErrorRate: wer,
  };
}
