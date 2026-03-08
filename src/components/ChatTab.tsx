import { useState, useRef, useEffect, useCallback } from 'react';
import { ModelCategory } from '@runanywhere/web';
import { TextGeneration } from '@runanywhere/web-llamacpp';
import { useModelLoader } from '../hooks/useModelLoader';
import { ModelBanner } from './ModelBanner';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  stats?: { tokens: number; tokPerSec: number; latencyMs: number };
}

export function ChatTab() {
  const loader = useModelLoader(ModelCategory.Language);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || generating) return;

    // Ensure model is loaded
    if (loader.state !== 'ready') {
      const ok = await loader.ensure();
      if (!ok) return;
    }

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', text }]);
    setGenerating(true);

    // Add empty assistant message for streaming
    const assistantIdx = messages.length + 1;
    setMessages((prev) => [...prev, { role: 'assistant', text: '' }]);

    try {
      const { stream, result: resultPromise, cancel } = await TextGeneration.generateStream(text, {
        maxTokens: 512,
        temperature: 0.7,
      });
      cancelRef.current = cancel;

      let accumulated = '';
      for await (const token of stream) {
        accumulated += token;
        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantIdx] = { role: 'assistant', text: accumulated };
          return updated;
        });
      }

      const result = await resultPromise;
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantIdx] = {
          role: 'assistant',
          text: result.text || accumulated,
          stats: {
            tokens: result.tokensUsed,
            tokPerSec: result.tokensPerSecond,
            latencyMs: result.latencyMs,
          },
        };
        return updated;
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantIdx] = { role: 'assistant', text: `Error: ${msg}` };
        return updated;
      });
    } finally {
      cancelRef.current = null;
      setGenerating(false);
    }
  }, [input, generating, messages.length, loader]);

  const handleCancel = () => {
    cancelRef.current?.();
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden relative h-full">
      <ModelBanner
        state={loader.state}
        progress={loader.progress}
        error={loader.error}
        onLoad={loader.ensure}
        label="LLM"
      />

      <div className="flex-1 overflow-y-auto px-6 py-8 flex flex-col gap-4 custom-scrollbar" ref={listRef}>
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center flex-1 text-center gap-3">
            <div className="text-5xl mb-2">💬</div>
            <h3 className="text-charcoal text-xl font-semibold">Start a conversation</h3>
            <p className="text-charcoal/60 max-w-sm">Type a message below to chat with on-device AI</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed break-words whitespace-pre-wrap shadow-sm ${
              msg.role === 'user' 
                ? 'bg-coral text-white rounded-br-md' 
                : 'bg-white text-charcoal border border-charcoal/10 rounded-bl-md'
            }`}>
              <p>{msg.text || '...'}</p>
              {msg.stats && (
                <div className={`mt-2 text-[11px] opacity-60 ${
                  msg.role === 'user' ? 'text-white' : 'text-charcoal'
                }`}>
                  {msg.stats.tokens} tokens · {msg.stats.tokPerSec.toFixed(1)} tok/s · {msg.stats.latencyMs.toFixed(0)}ms
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <form
        className="flex gap-2 p-4 px-6 border-t border-charcoal/10 bg-white/50 backdrop-blur-sm"
        onSubmit={(e) => { e.preventDefault(); send(); }}
      >
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={generating}
          className="flex-1 px-4 py-3 border border-charcoal/20 rounded-xl bg-white text-charcoal text-sm outline-none focus:border-coral focus:ring-2 focus:ring-coral/20 transition-all placeholder:text-charcoal/40"
        />
        {generating ? (
          <button 
            type="button" 
            className="px-5 py-3 border border-charcoal/20 rounded-xl bg-white text-charcoal text-sm font-medium hover:bg-charcoal/5 transition-all whitespace-nowrap"
            onClick={handleCancel}
          >
            Stop
          </button>
        ) : (
          <button 
            type="submit" 
            className="px-5 py-3 rounded-xl bg-coral text-white text-sm font-medium hover:bg-coral-dark transition-all whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed shadow-sm" 
            disabled={!input.trim()}
          >
            Send
          </button>
        )}
      </form>
    </div>
  );
}
