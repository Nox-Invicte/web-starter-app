import { useState, useEffect } from 'react';
import { initSDK, getAccelerationMode } from './runanywhere';
import { ScreenShield } from './components/ScreenShield';

export function App() {
  const [sdkReady, setSdkReady] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);

  useEffect(() => {
    initSDK()
      .then(() => setSdkReady(true))
      .catch((err) => setSdkError(err instanceof Error ? err.message : String(err)));
  }, []);

  if (sdkError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-5 text-center px-6 bg-cream">
        <div className="text-5xl">⚠️</div>
        <h2 className="text-2xl font-semibold text-charcoal">SDK Error</h2>
        <p className="text-coral max-w-md">{sdkError}</p>
      </div>
    );
  }

  if (!sdkReady) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-6 text-center px-6 bg-cream">
        <div className="spinner" />
        <h2 className="text-2xl font-semibold text-charcoal">Loading ScreenShield...</h2>
        <p className="text-charcoal/60">Initializing on-device AI engine</p>
      </div>
    );
  }

  const accel = getAccelerationMode();

  return (
    <div className="flex flex-col h-screen bg-cream">
      {/* Header */}
      <header className="bg-white/50 backdrop-blur-sm border-b border-charcoal/10 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🛡️</span>
            <div>
              <h1 className="text-xl font-bold text-charcoal">ScreenShield</h1>
              <p className="text-xs text-charcoal/50">Privacy-First Screenshot Sanitizer</p>
            </div>
          </div>
          {accel && (
            <span className="px-3 py-1 text-xs font-semibold uppercase bg-coral/10 text-coral rounded-full border border-coral/20">
              {accel === 'webgpu' ? '⚡ WebGPU' : '🖥️ CPU'}
            </span>
          )}
        </div>
      </header>

      {/* Main Content */}
      <ScreenShield />
    </div>
  );
}
