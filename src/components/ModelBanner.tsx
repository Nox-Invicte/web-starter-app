import type { LoaderState } from '../hooks/useModelLoader';

interface Props {
  state: LoaderState;
  progress: number;
  error: string | null;
  onLoad: () => void;
  label: string;
}

export function ModelBanner({ state, progress, error, onLoad, label }: Props) {
  if (state === 'ready') return null;

  return (
    <div className="flex items-center gap-3 px-6 py-3 bg-white/40 backdrop-blur-sm border-b border-charcoal/10 text-sm text-charcoal/70 flex-wrap">
      {state === 'idle' && (
        <>
          <span>No {label} model loaded.</span>
          <button 
            className="px-3 py-1.5 text-xs rounded-lg bg-coral text-charcoal font-semibold hover:bg-coral/90 transition-all whitespace-nowrap shadow-sm"
            onClick={onLoad}
          >
            Download & Load
          </button>
        </>
      )}
      {state === 'downloading' && (
        <>
          <span className="font-medium">Downloading {label} model... {(progress * 100).toFixed(0)}%</span>
          <div className="flex-1 h-2 bg-charcoal/10 rounded-full overflow-hidden min-w-[120px]">
            <div 
              className="h-full bg-coral rounded-full transition-all duration-300" 
              style={{ width: `${progress * 100}%` }} 
            />
          </div>
        </>
      )}
      {state === 'loading' && <span className="font-medium">Loading {label} model into engine...</span>}
      {state === 'error' && (
        <>
          <span className="text-coral font-medium">Error: {error}</span>
          <button 
            className="px-3 py-1.5 text-xs rounded-lg bg-coral text-charcoal font-semibold hover:bg-coral/90 transition-all whitespace-nowrap shadow-sm"
            onClick={onLoad}
          >
            Retry
          </button>
        </>
      )}
    </div>
  );
}
