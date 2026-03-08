import { useState, useRef, useEffect, useCallback } from 'react';
import { ModelCategory, VideoCapture } from '@runanywhere/web';
import { VLMWorkerBridge } from '@runanywhere/web-llamacpp';
import { useModelLoader } from '../hooks/useModelLoader';
import { ModelBanner } from './ModelBanner';

const LIVE_INTERVAL_MS = 2500;
const LIVE_MAX_TOKENS = 30;
const SINGLE_MAX_TOKENS = 80;
const CAPTURE_DIM = 256; // CLIP resizes internally; larger is wasted work

interface VisionResult {
  text: string;
  totalMs: number;
}

export function VisionTab() {
  const loader = useModelLoader(ModelCategory.Multimodal);
  const [cameraActive, setCameraActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [result, setResult] = useState<VisionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('Describe what you see briefly.');

  const videoMountRef = useRef<HTMLDivElement>(null);
  const captureRef = useRef<VideoCapture | null>(null);
  const processingRef = useRef(false);
  const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const liveModeRef = useRef(false);

  // Keep refs in sync with state so interval callbacks see latest values
  processingRef.current = processing;
  liveModeRef.current = liveMode;

  // ------------------------------------------------------------------
  // Camera
  // ------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    if (captureRef.current?.isCapturing) return;

    setError(null);

    try {
      const cam = new VideoCapture({ facingMode: 'environment' });
      await cam.start();
      captureRef.current = cam;

      const mount = videoMountRef.current;
      if (mount) {
        const el = cam.videoElement;
        el.style.width = '100%';
        el.style.borderRadius = '12px';
        mount.appendChild(el);
      }

      setCameraActive(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes('NotAllowed') || msg.includes('Permission')) {
        setError(
          'Camera permission denied. On macOS, check System Settings → Privacy & Security → Camera and ensure your browser is allowed.',
        );
      } else if (msg.includes('NotFound') || msg.includes('DevicesNotFound')) {
        setError('No camera found on this device.');
      } else if (msg.includes('NotReadable') || msg.includes('TrackStartError')) {
        setError('Camera is in use by another application.');
      } else {
        setError(`Camera error: ${msg}`);
      }
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
      const cam = captureRef.current;
      if (cam) {
        cam.stop();
        cam.videoElement.parentNode?.removeChild(cam.videoElement);
        captureRef.current = null;
      }
    };
  }, []);

  // ------------------------------------------------------------------
  // Core: capture + infer
  // ------------------------------------------------------------------
  const describeFrame = useCallback(async (maxTokens: number) => {
    if (processingRef.current) return;

    const cam = captureRef.current;
    if (!cam?.isCapturing) return;

    // Ensure model loaded
    if (loader.state !== 'ready') {
      const ok = await loader.ensure();
      if (!ok) return;
    }

    const frame = cam.captureFrame(CAPTURE_DIM);
    if (!frame) return;

    setProcessing(true);
    processingRef.current = true;
    setError(null);

    const t0 = performance.now();

    try {
      const bridge = VLMWorkerBridge.shared;
      if (!bridge.isModelLoaded) {
        throw new Error('VLM model not loaded in worker');
      }

      const res = await bridge.process(
        frame.rgbPixels,
        frame.width,
        frame.height,
        prompt,
        { maxTokens, temperature: 0.6 },
      );

      setResult({ text: res.text, totalMs: performance.now() - t0 });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isWasmCrash = msg.includes('memory access out of bounds')
        || msg.includes('RuntimeError');

      if (isWasmCrash) {
        setResult({ text: 'Recovering from memory error... next frame will retry.', totalMs: 0 });
      } else {
        setError(msg);
        if (liveModeRef.current) stopLive();
      }
    } finally {
      setProcessing(false);
      processingRef.current = false;
    }
  }, [loader, prompt]);

  // ------------------------------------------------------------------
  // Single-shot
  // ------------------------------------------------------------------
  const describeSingle = useCallback(async () => {
    if (!captureRef.current?.isCapturing) {
      await startCamera();
      return;
    }
    await describeFrame(SINGLE_MAX_TOKENS);
  }, [startCamera, describeFrame]);

  // ------------------------------------------------------------------
  // Live mode
  // ------------------------------------------------------------------
  const startLive = useCallback(async () => {
    if (!captureRef.current?.isCapturing) {
      await startCamera();
    }

    setLiveMode(true);
    liveModeRef.current = true;

    // Immediately describe first frame
    describeFrame(LIVE_MAX_TOKENS);

    // Then poll every 2.5s — skips ticks while inference is running
    liveIntervalRef.current = setInterval(() => {
      if (!processingRef.current && liveModeRef.current) {
        describeFrame(LIVE_MAX_TOKENS);
      }
    }, LIVE_INTERVAL_MS);
  }, [startCamera, describeFrame]);

  const stopLive = useCallback(() => {
    setLiveMode(false);
    liveModeRef.current = false;
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
  }, []);

  const toggleLive = useCallback(() => {
    if (liveMode) {
      stopLive();
    } else {
      startLive();
    }
  }, [liveMode, startLive, stopLive]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className="flex-1 flex flex-col overflow-hidden h-full">
      <ModelBanner
        state={loader.state}
        progress={loader.progress}
        error={loader.error}
        onLoad={loader.ensure}
        label="VLM"
      />

      <div className="relative flex-1 bg-charcoal/5 flex items-center justify-center overflow-hidden m-6 rounded-2xl border border-charcoal/10">
        {!cameraActive && (
          <div className="flex flex-col items-center justify-center text-center gap-3 p-6">
            <div className="text-5xl mb-2">📷</div>
            <h3 className="text-charcoal text-xl font-semibold">Camera Preview</h3>
            <p className="text-charcoal/60">Tap below to start the camera</p>
          </div>
        )}
        <div ref={videoMountRef} className="w-full h-full rounded-2xl overflow-hidden" />
      </div>

      <input
        className="mx-6 mb-4 px-4 py-3 border border-charcoal/20 rounded-xl bg-white text-charcoal text-sm outline-none focus:border-coral focus:ring-2 focus:ring-coral/20 transition-all placeholder:text-charcoal/40 disabled:opacity-50 disabled:bg-charcoal/5"
        type="text"
        placeholder="What do you want to know about the image?"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={liveMode}
      />

      <div className="flex gap-3 px-6 pb-6">
        {!cameraActive ? (
          <button 
            className="flex-1 px-5 py-3 rounded-xl bg-coral hover:bg-coral-dark text-white text-sm font-medium transition-all shadow-sm"
            onClick={startCamera}
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              className="flex-1 px-5 py-3 rounded-xl bg-coral hover:bg-coral-dark text-white text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
              onClick={describeSingle}
              disabled={processing || liveMode}
            >
              {processing && !liveMode ? 'Analyzing...' : 'Describe'}
            </button>
            <button
              className={`px-6 py-3 rounded-xl text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-sm ${
                liveMode 
                  ? 'bg-charcoal hover:bg-charcoal/90 text-white' 
                  : 'bg-white hover:bg-charcoal/5 text-charcoal border border-charcoal/20'
              }`}
              onClick={toggleLive}
              disabled={processing && !liveMode}
            >
              {liveMode ? '⏹ Stop Live' : '▶ Live'}
            </button>
          </>
        )}
      </div>

      {error && (
        <div className="mx-6 mb-6 p-4 bg-coral/10 border border-coral/20 rounded-xl">
          <span className="text-coral text-sm font-medium">Error: {error}</span>
        </div>
      )}

      {result && (
        <div className="mx-6 mb-6 p-5 bg-white border border-charcoal/10 rounded-xl shadow-sm">
          {liveMode && (
            <span className="inline-block mb-3 px-3 py-1 text-xs font-bold uppercase bg-coral text-white rounded-full">
              LIVE
            </span>
          )}
          <h4 className="text-sm font-semibold mb-2 text-charcoal/60 uppercase tracking-wide">Result</h4>
          <p className="text-sm text-charcoal leading-relaxed">{result.text}</p>
          {result.totalMs > 0 && (
            <div className="mt-3 text-xs text-charcoal/50">
              {(result.totalMs / 1000).toFixed(1)}s
            </div>
          )}
        </div>
      )}
    </div>
  );
}
