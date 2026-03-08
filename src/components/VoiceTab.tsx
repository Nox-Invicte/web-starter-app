import { useState, useRef, useCallback, useEffect } from 'react';
import { VoicePipeline, ModelCategory, ModelManager, AudioCapture, AudioPlayback, SpeechActivity } from '@runanywhere/web';
import { VAD } from '@runanywhere/web-onnx';
import { useModelLoader } from '../hooks/useModelLoader';
import { ModelBanner } from './ModelBanner';

type VoiceState = 'idle' | 'loading-models' | 'listening' | 'processing' | 'speaking';

export function VoiceTab() {
  const llmLoader = useModelLoader(ModelCategory.Language, true);
  const sttLoader = useModelLoader(ModelCategory.SpeechRecognition, true);
  const ttsLoader = useModelLoader(ModelCategory.SpeechSynthesis, true);
  const vadLoader = useModelLoader(ModelCategory.Audio, true);

  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const micRef = useRef<AudioCapture | null>(null);
  const pipelineRef = useRef<VoicePipeline | null>(null);
  const vadUnsub = useRef<(() => void) | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      micRef.current?.stop();
      vadUnsub.current?.();
    };
  }, []);

  // Ensure all 4 models are loaded
  const ensureModels = useCallback(async (): Promise<boolean> => {
    setVoiceState('loading-models');
    setError(null);

    const results = await Promise.all([
      vadLoader.ensure(),
      sttLoader.ensure(),
      llmLoader.ensure(),
      ttsLoader.ensure(),
    ]);

    if (results.every(Boolean)) {
      setVoiceState('idle');
      return true;
    }

    setError('Failed to load one or more voice models');
    setVoiceState('idle');
    return false;
  }, [vadLoader, sttLoader, llmLoader, ttsLoader]);

  // Start listening
  const startListening = useCallback(async () => {
    setTranscript('');
    setResponse('');
    setError(null);

    // Load models if needed
    const anyMissing = !ModelManager.getLoadedModel(ModelCategory.Audio)
      || !ModelManager.getLoadedModel(ModelCategory.SpeechRecognition)
      || !ModelManager.getLoadedModel(ModelCategory.Language)
      || !ModelManager.getLoadedModel(ModelCategory.SpeechSynthesis);

    if (anyMissing) {
      const ok = await ensureModels();
      if (!ok) return;
    }

    setVoiceState('listening');

    const mic = new AudioCapture({ sampleRate: 16000 });
    micRef.current = mic;

    if (!pipelineRef.current) {
      pipelineRef.current = new VoicePipeline();
    }

    // Start VAD + mic
    VAD.reset();

    vadUnsub.current = VAD.onSpeechActivity((activity) => {
      if (activity === SpeechActivity.Ended) {
        const segment = VAD.popSpeechSegment();
        if (segment && segment.samples.length > 1600) {
          processSpeech(segment.samples);
        }
      }
    });

    await mic.start(
      (chunk) => { VAD.processSamples(chunk); },
      (level) => { setAudioLevel(level); },
    );
  }, [ensureModels]);

  // Process a speech segment through the full pipeline
  const processSpeech = useCallback(async (audioData: Float32Array) => {
    const pipeline = pipelineRef.current;
    if (!pipeline) return;

    // Stop mic during processing
    micRef.current?.stop();
    vadUnsub.current?.();
    setVoiceState('processing');

    try {
      const result = await pipeline.processTurn(audioData, {
        maxTokens: 60,
        temperature: 0.7,
        systemPrompt: 'You are a helpful voice assistant. Keep responses concise — 1-2 sentences max.',
      }, {
        onTranscription: (text) => {
          setTranscript(text);
        },
        onResponseToken: (_token, accumulated) => {
          setResponse(accumulated);
        },
        onResponseComplete: (text) => {
          setResponse(text);
        },
        onSynthesisComplete: async (audio, sampleRate) => {
          setVoiceState('speaking');
          const player = new AudioPlayback({ sampleRate });
          await player.play(audio, sampleRate);
          player.dispose();
        },
        onStateChange: (s) => {
          if (s === 'processingSTT') setVoiceState('processing');
          if (s === 'generatingResponse') setVoiceState('processing');
          if (s === 'playingTTS') setVoiceState('speaking');
        },
      });

      if (result) {
        setTranscript(result.transcription);
        setResponse(result.response);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }

    setVoiceState('idle');
    setAudioLevel(0);
  }, []);

  const stopListening = useCallback(() => {
    micRef.current?.stop();
    vadUnsub.current?.();
    setVoiceState('idle');
    setAudioLevel(0);
  }, []);

  // Which loaders are still loading?
  const pendingLoaders = [
    { label: 'VAD', loader: vadLoader },
    { label: 'STT', loader: sttLoader },
    { label: 'LLM', loader: llmLoader },
    { label: 'TTS', loader: ttsLoader },
  ].filter((l) => l.loader.state !== 'ready');

  return (
    <div className="flex-1 flex flex-col overflow-hidden h-full">
      {pendingLoaders.length > 0 && voiceState === 'idle' && (
        <ModelBanner
          state={pendingLoaders[0].loader.state}
          progress={pendingLoaders[0].loader.progress}
          error={pendingLoaders[0].loader.error}
          onLoad={ensureModels}
          label={`Voice (${pendingLoaders.map((l) => l.label).join(', ')})`}
        />
      )}

      {error && (
        <div className="mx-6 mt-6 p-4 bg-coral/10 border border-coral/20 rounded-xl">
          <span className="text-coral text-sm font-medium">{error}</span>
        </div>
      )}

      <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6">
        <div 
          className={`relative w-36 h-36 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl ${
            voiceState === 'listening' 
              ? 'bg-coral shadow-coral/30' 
              : voiceState === 'processing' || voiceState === 'speaking'
              ? 'bg-charcoal shadow-charcoal/30'
              : 'bg-white border-2 border-charcoal/10'
          }`}
          style={{ 
            transform: voiceState === 'listening' ? `scale(${1 + audioLevel * 0.2})` : 'scale(1)',
            transition: 'transform 0.1s ease-out, background-color 0.3s ease, box-shadow 0.3s ease'
          }}
        >
          <div className={`w-28 h-28 rounded-full flex items-center justify-center ${
            voiceState === 'listening' || voiceState === 'processing' || voiceState === 'speaking'
              ? 'bg-cream'
              : 'bg-charcoal/5'
          }`}>
            <span className="text-5xl">
              {voiceState === 'listening' ? '🎤' : voiceState === 'processing' ? '⚙️' : voiceState === 'speaking' ? '🔊' : '🎙️'}
            </span>
          </div>
        </div>

        <p className="text-charcoal/70 text-center font-medium text-lg">
          {voiceState === 'idle' && 'Tap to start listening'}
          {voiceState === 'loading-models' && 'Loading models...'}
          {voiceState === 'listening' && 'Listening... speak now'}
          {voiceState === 'processing' && 'Processing...'}
          {voiceState === 'speaking' && 'Speaking...'}
        </p>

        {voiceState === 'idle' || voiceState === 'loading-models' ? (
          <button
            className="px-8 py-4 rounded-xl bg-coral hover:bg-coral-dark text-white font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg"
            onClick={startListening}
            disabled={voiceState === 'loading-models'}
          >
            Start Listening
          </button>
        ) : voiceState === 'listening' ? (
          <button 
            className="px-8 py-4 rounded-xl bg-white hover:bg-charcoal/5 text-charcoal font-medium border-2 border-charcoal/20 transition-all"
            onClick={stopListening}
          >
            Stop
          </button>
        ) : null}
      </div>

      {transcript && (
        <div className="mx-6 mb-4 p-5 bg-white border border-charcoal/10 rounded-xl shadow-sm">
          <h4 className="text-xs font-semibold mb-2 text-charcoal/60 uppercase tracking-wide">You said:</h4>
          <p className="text-sm text-charcoal leading-relaxed">{transcript}</p>
        </div>
      )}

      {response && (
        <div className="mx-6 mb-6 p-5 bg-coral/5 border border-coral/20 rounded-xl">
          <h4 className="text-xs font-semibold mb-2 text-coral uppercase tracking-wide">AI response:</h4>
          <p className="text-sm text-charcoal leading-relaxed">{response}</p>
        </div>
      )}
    </div>
  );
}
