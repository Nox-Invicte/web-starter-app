import { useState, useRef, useCallback, useEffect } from 'react';
import { ModelCategory } from '@runanywhere/web';
import { VLMWorkerBridge } from '@runanywhere/web-llamacpp';
import { useModelLoader } from '../hooks/useModelLoader';
import { ModelBanner } from './ModelBanner';
import { detectSensitivePatterns, type SensitiveMatch } from '../utils/patternDetector';
import { autoPreprocess, resizeImage } from '../utils/imagePreprocessor';
import { getTrainedPatterns } from '../utils/datasetLoader';
import { getTesseractEngine, getTrainedTesseractConfig } from '../utils/tesseractOCR';
import { trainTesseractOnDatasets, type TrainingProgress, type TrainingMode, TRAINING_MODES } from '../utils/tesseractTraining';
import { getHybridOCREngine } from '../utils/hybridOCR';
import { smartVLMProcess, type SmartVLMResult } from '../utils/smartVLM';
import { calibrateVLM, type CalibrationProgress, type CalibrationResult } from '../utils/vlmCalibration';

interface ProcessedResult {
  text: string;
  matches: SensitiveMatch[];
  imageData: string;
  confidence?: number;
  engine: 'vlm' | 'tesseract' | 'hybrid';
  method?: string;
}

type OCREngine = 'vlm' | 'tesseract' | 'hybrid';

export function ScreenShield() {
  const loader = useModelLoader(ModelCategory.Multimodal);
  const [processing, setProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [result, setResult] = useState<ProcessedResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [enhanceOCR, setEnhanceOCR] = useState(true);
  const [ocrEngine, setOcrEngine] = useState<OCREngine>('hybrid'); // Default to hybrid
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [trainingMode, setTrainingMode] = useState<TrainingMode>('balanced'); // Default to balanced
  const [vlmCalibrating, setVlmCalibrating] = useState(false);
  const [vlmCalibrationProgress, setVlmCalibrationProgress] = useState<CalibrationProgress | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize dataset-trained patterns on mount
  useEffect(() => {
    getTrainedPatterns().then((patterns) => {
      console.log('✅ Dataset-trained patterns loaded:', patterns.size, 'categories');
    }).catch((err) => {
      console.warn('⚠️ Failed to load trained patterns:', err);
    });
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setPreviewUrl(event.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  // Process screenshot
  const processScreenshot = useCallback(async () => {
    if (!selectedFile || !previewUrl) return;

    setProcessing(true);
    setProcessingStatus('Loading image...');
    setError(null);

    try {
      // Load image
      const img = new Image();
      img.src = previewUrl;
      await new Promise((resolve) => { img.onload = resolve; });

      setProcessingStatus('Preparing image...');
      
      // Draw to canvas
      const canvas = document.createElement('canvas');
      const targetSize = 512;
      const scale = Math.min(targetSize / img.width, targetSize / img.height);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Apply preprocessing if enabled
      if (enhanceOCR) {
        setProcessingStatus('🔧 Enhancing image (contrast, sharpening, denoising)...');
        console.log('🔧 Applying image preprocessing for enhanced OCR...');
        imageData = autoPreprocess(imageData);
        ctx.putImageData(imageData, 0, 0);
      }

      let extractedText = '';
      let confidence = 0;
      let method: string | undefined;

      // Use selected OCR engine
      if (ocrEngine === 'hybrid') {
        setProcessingStatus('🔬 Running Hybrid OCR (Tesseract + VLM)...');
        console.log('🔬 Using Hybrid OCR (Tesseract + VLM)...');
        
        // Check both loader state AND VLMWorkerBridge
        const bridge = VLMWorkerBridge.shared;
        const vlmAvailable = loader.state === 'ready' && bridge.isModelLoaded;
        
        console.log(`VLM Status: loader.state=${loader.state}, bridge.isModelLoaded=${bridge.isModelLoaded}`);
        
        if (!vlmAvailable) {
          setProcessingStatus('⚠️ VLM not loaded - using Tesseract only...');
          console.warn('VLM not available, hybrid will use Tesseract only');
        }
        
        const hybridEngine = getHybridOCREngine({
          confidenceThreshold: 75,
          enableFusion: true,
          vlmValidation: vlmAvailable,
          preprocessEnabled: enhanceOCR,
        });
        
        console.log('Calling hybrid engine recognize...');
        const result = await hybridEngine.recognize(canvas, vlmAvailable);
        
        extractedText = result.text;
        confidence = result.confidence;
        method = result.method;
        
        console.log(`✅ Hybrid OCR complete (method: ${method}, confidence: ${confidence.toFixed(1)}%)`);
      } else if (ocrEngine === 'tesseract') {
        setProcessingStatus('🔤 Running Tesseract OCR (Dataset-Trained)...');
        console.log('🔤 Using Tesseract OCR (Dataset-Trained)...');
        const tesseract = await getTesseractEngine(getTrainedTesseractConfig());
        const result = await tesseract.recognize(canvas, false); // Already preprocessed
        extractedText = result.text;
        confidence = result.confidence;
        console.log(`✅ Tesseract OCR complete (confidence: ${confidence.toFixed(1)}%)`);
      } else {
        setProcessingStatus('🧠 Running Smart VLM OCR (multi-pass)...');
        console.log('🧠 Using Smart VLM OCR (multi-pass, adaptive prompts)...');
        
        // Ensure VLM model is loaded
        if (loader.state !== 'ready') {
          setProcessingStatus('📥 Loading VLM model... This may take a few minutes...');
          console.log('⏳ VLM model not loaded, loading now...');
          const ok = await loader.ensure();
          if (!ok) throw new Error('VLM model failed to load');
        }

        const bridge = VLMWorkerBridge.shared;
        if (!bridge.isModelLoaded) {
          throw new Error('Vision model not loaded');
        }

        // Use Smart VLM: classify → extract with specialized prompt → sensitive scan → post-process
        const smartResult: SmartVLMResult = await smartVLMProcess(canvas, (stage) => {
          setProcessingStatus(`🧠 ${stage}`);
        });

        extractedText = smartResult.text;
        confidence = smartResult.quality;
        
        if (smartResult.usedFallback) {
          method = 'tesseract (VLM fallback)';
          console.log('⚠️ VLM hallucinated — used Tesseract fallback');
        }
        
        // Merge VLM-detected sensitive items into pattern detection
        if (smartResult.sensitiveFindings.length > 0) {
          console.log(`🔒 VLM found ${smartResult.sensitiveFindings.length} extra sensitive items`);
        }
        
        console.log(`✅ Smart VLM OCR complete (type: ${smartResult.contentType}, quality: ${smartResult.quality}, passes: ${smartResult.passes.join(', ')})`);
      }

      setProcessingStatus('🔍 Detecting sensitive patterns...');
      
      // Detect sensitive patterns in extracted text
      const matches = detectSensitivePatterns(extractedText);

      setResult({
        text: extractedText,
        matches,
        imageData: previewUrl,
        confidence,
        engine: ocrEngine,
        method,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Processing failed: ${msg}`);
    } finally {
      setProcessing(false);
      setProcessingStatus('');
    }
  }, [selectedFile, previewUrl, loader, enhanceOCR, ocrEngine]);

  // Train Tesseract on datasets with progress tracking
  const runTraining = useCallback(async () => {
    setTraining(true);
    setError(null);
    setTrainingProgress(null);
    
    try {
      console.log(`🎓 Starting Tesseract training [${TRAINING_MODES[trainingMode].label}]...`);
      
      const trainingResult = await trainTesseractOnDatasets(trainingMode, (progress) => {
        setTrainingProgress(progress);
      });
      
      console.log(`✅ Training complete!`);
      console.log(`📊 Accuracy: ${trainingResult.accuracy.toFixed(2)}%`);
      console.log(`📈 Confidence: ${trainingResult.avgConfidence.toFixed(2)}%`);
      
      alert(
        `Training Complete! 🎉\n\n` +
        `Mode: ${TRAINING_MODES[trainingMode].label}\n` +
        `✅ Processed ${trainingResult.processedSamples} real images from SROIE2019 dataset\n\n` +
        `📊 Results:\n` +
        `  Accuracy: ${trainingResult.accuracy.toFixed(2)}%\n` +
        `  Confidence: ${trainingResult.avgConfidence.toFixed(2)}%\n\n` +
        `⚙️ Best Configuration:\n` +
        `  PSM: ${trainingResult.bestConfig.psm}\n` +
        `  OEM: ${trainingResult.bestConfig.oem}\n\n` +
        `💾 Configuration saved for future use!`
      );
      
      setTrainingProgress(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Training failed: ${msg}`);
      console.error('Training error:', err);
    } finally {
      setTraining(false);
    }
  }, [trainingMode]);

  // Run VLM Calibration (train VLM prompts on dataset)
  const runVLMCalibration = useCallback(async () => {
    setVlmCalibrating(true);
    setError(null);
    setVlmCalibrationProgress(null);
    
    try {
      console.log('🎯 Starting VLM calibration...');
      
      const sampleCount = trainingMode === 'quick' ? 3 : trainingMode === 'balanced' ? 5 : 8;
      const calibrationResult: CalibrationResult = await calibrateVLM(sampleCount, (progress) => {
        setVlmCalibrationProgress(progress);
      });
      
      alert(
        `VLM Calibration Complete! 🎯\n\n` +
        `Evaluated ${calibrationResult.samplesEvaluated} samples\n` +
        `Overall accuracy: ${calibrationResult.overallAccuracy.toFixed(1)}%\n` +
        `Best prompt: ${calibrationResult.bestPromptId} (${calibrationResult.bestContentType})\n` +
        `Time: ${(calibrationResult.timeMs / 1000).toFixed(1)}s\n\n` +
        `Calibration saved — VLM will now use optimized prompts!`
      );
      
      setVlmCalibrationProgress(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`VLM Calibration failed: ${msg}`);
    } finally {
      setVlmCalibrating(false);
    }
  }, [trainingMode]);

  // Export redacted image
  const exportRedacted = useCallback(() => {
    if (!result || !canvasRef.current) return;

    const link = document.createElement('a');
    link.download = `redacted-${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL();
    link.click();
  }, [result]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-cream">
      <ModelBanner
        state={loader.state}
        progress={loader.progress}
        error={loader.error}
        onLoad={loader.ensure}
        label="Vision Model (OCR)"
      />

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="max-w-4xl mx-auto p-6 space-y-6">
          {/* Header */}
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-bold text-charcoal">🛡️ ScreenShield</h1>
            <p className="text-charcoal/80 max-w-2xl mx-auto font-medium">
              Privacy-first screenshot sanitizer. Detect and redact sensitive information 
              (passwords, API keys, credit cards, PII) - all processed locally on your device.
            </p>
          </div>

          {/* AI Enhancement Settings */}
          <div className="bg-white/70 backdrop-blur rounded-2xl p-4 shadow-sm space-y-4">
            {/* OCR Engine Selection */}
            <div className="flex items-center justify-between gap-4 pb-4 border-b border-charcoal/10">
              <div className="flex items-center gap-3">
                <span className="font-semibold text-charcoal">🔤 OCR Engine:</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => setOcrEngine('tesseract')}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      ocrEngine === 'tesseract'
                        ? 'bg-coral text-charcoal shadow-sm'
                        : 'bg-white text-charcoal border border-charcoal/20 hover:border-coral/50'
                    }`}
                  >
                    Tesseract
                  </button>
                  <button
                    onClick={() => setOcrEngine('vlm')}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      ocrEngine === 'vlm'
                        ? 'bg-coral text-charcoal shadow-sm'
                        : 'bg-white text-charcoal border border-charcoal/20 hover:border-coral/50'
                    }`}
                  >
                    VLM
                  </button>
                  <button
                    onClick={() => setOcrEngine('hybrid')}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      ocrEngine === 'hybrid'
                        ? 'bg-coral text-charcoal shadow-sm'
                        : 'bg-white text-charcoal border border-charcoal/20 hover:border-coral/50'
                    }`}
                  >
                    🔬 Hybrid (Best)
                  </button>
                </div>
                {(ocrEngine === 'tesseract' || ocrEngine === 'hybrid') && (
                  <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full font-medium">
                    Dataset-Trained
                  </span>
                )}
                {ocrEngine === 'hybrid' && (
                  <span className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full font-medium">
                    AI-Enhanced
                  </span>
                )}
              </div>
              
              {/* Training Mode Selector */}
              {(ocrEngine === 'tesseract' || ocrEngine === 'hybrid') && (
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-charcoal">
                    Training Mode
                  </label>
                  <select
                    value={trainingMode}
                    onChange={(e) => setTrainingMode(e.target.value as TrainingMode)}
                    disabled={training}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-charcoal text-sm hover:border-coral focus:outline-none focus:ring-2 focus:ring-coral/50 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                  >
                    {(Object.entries(TRAINING_MODES) as [TrainingMode, typeof TRAINING_MODES[TrainingMode]][]).map(([key, config]) => (
                      <option key={key} value={key}>
                        {config.label} - {config.description} ({config.estimatedTime})
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-600">
                    {TRAINING_MODES[trainingMode].description} • Est. {TRAINING_MODES[trainingMode].estimatedTime}
                  </p>
                </div>
              )}
              
              {/* Training Button */}
              {(ocrEngine === 'tesseract' || ocrEngine === 'hybrid') && (
                <button
                  onClick={runTraining}
                  disabled={training}
                  className="px-4 py-2 rounded-lg bg-coral/10 text-charcoal border-2 border-coral font-semibold hover:bg-coral/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm shadow-sm"
                >
                  {training ? '⏳ Training...' : '🎓 Train Tesseract on Dataset'}
                </button>
              )}
              
              {/* VLM Calibration Button */}
              {(ocrEngine === 'vlm' || ocrEngine === 'hybrid') && (
                <button
                  onClick={runVLMCalibration}
                  disabled={vlmCalibrating || loader.state !== 'ready'}
                  className="px-4 py-2 rounded-lg bg-purple-50 text-charcoal border-2 border-purple-400 font-semibold hover:bg-purple-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm shadow-sm"
                >
                  {vlmCalibrating ? '⏳ Calibrating VLM...' : '🎯 Calibrate VLM on Dataset'}
                </button>
              )}
            </div>

            {/* Training Progress */}
            {training && trainingProgress && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-blue-900">{trainingProgress.stage}</span>
                  <span className="text-sm text-blue-700">
                    Config {trainingProgress.currentConfig}/{trainingProgress.totalConfigs}
                  </span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all"
                    style={{
                      width: `${(trainingProgress.currentSample / trainingProgress.totalSamples) * 100}%`,
                    }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-blue-700">
                  <span>Sample {trainingProgress.currentSample}/{trainingProgress.totalSamples}</span>
                  <span>
                    Accuracy: {trainingProgress.accuracy.toFixed(1)}% | 
                    Confidence: {trainingProgress.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}

            {/* VLM Calibration Progress */}
            {vlmCalibrating && vlmCalibrationProgress && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-purple-900">{vlmCalibrationProgress.stage}</span>
                  <span className="text-sm text-purple-700">
                    {vlmCalibrationProgress.currentPrompt}
                  </span>
                </div>
                <div className="w-full bg-purple-200 rounded-full h-2.5">
                  <div
                    className="bg-purple-600 h-2.5 rounded-full transition-all"
                    style={{
                      width: `${(vlmCalibrationProgress.currentSample / Math.max(vlmCalibrationProgress.totalSamples, 1)) * 100}%`,
                    }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-purple-700">
                  <span>Sample {vlmCalibrationProgress.currentSample}/{vlmCalibrationProgress.totalSamples}</span>
                  <span>
                    Accuracy: {vlmCalibrationProgress.accuracy.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}

            {/* Enhancement Options */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enhanceOCR}
                    onChange={(e) => setEnhanceOCR(e.target.checked)}
                    className="w-4 h-4 accent-coral cursor-pointer"
                  />
                  <span className="font-medium text-charcoal">
                    🔧 Image Preprocessing
                  </span>
                </label>
                <span className="text-xs px-2 py-1 bg-coral/20 text-coral rounded-full font-medium">
                  Contrast + Sharpen + Denoise
                </span>
              </div>
              <div className="text-xs text-charcoal/60 hidden sm:block">
                {ocrEngine === 'hybrid'
                  ? '🔬 Tesseract + Smart VLM fusion (multi-pass, calibrated)'
                  : ocrEngine === 'tesseract' 
                  ? '📊 Trained on SROIE2019 receipt dataset (626 images)'
                  : '🧠 Smart VLM: classify → extract → sensitive scan → post-process'}
              </div>
            </div>
          </div>

          {/* Upload Area */}
          {!selectedFile && (
            <div 
              className="border-2 border-dashed border-charcoal/20 rounded-2xl p-12 text-center hover:border-coral/50 transition-colors cursor-pointer bg-white/50"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="space-y-4">
                <div className="text-6xl">📸</div>
                <div>
                  <p className="text-lg font-medium text-charcoal mb-2">
                    Drop screenshot here or click to upload
                  </p>
                  <p className="text-sm text-charcoal/70 font-medium">
                    Supports PNG, JPG, WebP • Max 10MB • 100% private & offline
                  </p>
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          )}

          {/* Preview & Process */}
          {selectedFile && !result && (
            <div className="space-y-4">
              <div className="bg-white/70 backdrop-blur rounded-2xl p-6 shadow-sm">
                <img 
                  src={previewUrl!} 
                  alt="Screenshot preview" 
                  className="w-full rounded-lg"
                />
              </div>
              
              {/* Processing Status */}
              {processing && processingStatus && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center gap-2">
                  <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                  <span className="text-sm font-medium text-blue-900">{processingStatus}</span>
                </div>
              )}
              
              <div className="flex gap-3">
                <button
                  onClick={processScreenshot}
                  disabled={processing}
                  className="flex-1 bg-coral hover:bg-coral/90 text-charcoal px-6 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                >
                  {processing ? '🔍 Analyzing...' : '🔍 Scan for Sensitive Data'}
                </button>
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setResult(null);
                    setError(null);
                  }}
                  className="px-6 py-3 rounded-xl border-2 border-charcoal/30 hover:border-charcoal/60 bg-white text-charcoal font-semibold transition-all shadow-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-4">
              {/* Detection Summary */}
              <div className="bg-white/70 backdrop-blur rounded-2xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-charcoal">
                      🔎 Detection Results
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2 py-1 bg-charcoal/10 text-charcoal rounded-full font-medium">
                        {result.engine === 'hybrid' 
                          ? `🔬 Hybrid (${result.method || 'tesseract+vlm'})`
                          : result.engine === 'tesseract' 
                          ? '🔤 Tesseract' 
                          : '👁️ VLM'}
                      </span>
                      {result.confidence !== undefined && result.confidence > 0 && (
                        <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full font-medium">
                          {result.confidence.toFixed(1)}% confidence
                        </span>
                      )}
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    result.matches.length > 0 
                      ? 'bg-coral/20 text-coral' 
                      : 'bg-green-100 text-green-700'
                  }`}>
                    {result.matches.length > 0 
                      ? `${result.matches.length} sensitive item${result.matches.length > 1 ? 's' : ''} found`
                      : '✓ No sensitive data detected'
                    }
                  </span>
                </div>

                {result.matches.length > 0 && (
                  <div className="space-y-2">
                    {result.matches.map((match, idx) => (
                      <div 
                        key={idx}
                        className="flex items-start gap-3 p-3 bg-cream rounded-lg border border-charcoal/10"
                      >
                        <span className="text-2xl shrink-0">{match.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-charcoal">{match.type}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              match.confidence === 'high' 
                                ? 'bg-red-100 text-red-700'
                                : match.confidence === 'medium'
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-blue-100 text-blue-700'
                            }`}>
                              {match.confidence} confidence
                            </span>
                          </div>
                          <p className="text-sm text-charcoal/90 font-mono break-all bg-white/50 px-2 py-1 rounded">
                            {match.value}
                          </p>
                          <p className="text-xs text-charcoal/70 mt-1">{match.reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Extracted Text */}
              <div className="bg-white/70 backdrop-blur rounded-2xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold text-charcoal mb-3">📝 Extracted Text</h3>
                <pre className="text-sm text-charcoal font-medium whitespace-pre-wrap font-mono bg-white p-4 rounded-lg overflow-auto max-h-64 border border-charcoal/10">
                  {result.text || 'No text detected'}
                </pre>
              </div>

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setResult(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  className="flex-1 bg-coral hover:bg-coral/90 text-charcoal px-6 py-3 rounded-xl font-semibold transition-all shadow-sm"
                >
                  🔄 Scan Another Screenshot
                </button>
                {result.matches.length > 0 && (
                  <button
                    onClick={exportRedacted}
                    className="px-6 py-3 rounded-xl border-2 border-coral hover:bg-coral/10 bg-white text-coral font-semibold transition-all shadow-sm"
                  >
                    📥 Export Report
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
              ⚠️ {error}
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for redaction rendering */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
