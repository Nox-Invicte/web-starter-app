import { ModelCategory } from '@runanywhere/web';
import {
  ToolCalling,
  ToolCallFormat,
  toToolValue,
  getStringArg,
  getNumberArg,
  type ToolDefinition,
  type ToolCall,
  type ToolResult,
  type ToolCallingResult,
  type ToolValue,
} from '@runanywhere/web-llamacpp';
import { useState, useRef, useEffect, useCallback } from 'react';

import { useModelLoader } from '../hooks/useModelLoader';
import { ModelBanner } from './ModelBanner';

// ---------------------------------------------------------------------------
// Built-in demo tools
// ---------------------------------------------------------------------------

const DEMO_TOOLS: { def: ToolDefinition; executor: Parameters<typeof ToolCalling.registerTool>[1] }[] = [
  {
    def: {
      name: 'get_weather',
      description: 'Gets the current weather for a city. Returns temperature in Fahrenheit and a short condition.',
      parameters: [
        { name: 'location', type: 'string', description: 'City name (e.g. "San Francisco")', required: true },
      ],
      category: 'Utility',
    },
    executor: async (args) => {
      const city = getStringArg(args, 'location') ?? 'Unknown';
      const conditions = ['Sunny', 'Partly Cloudy', 'Overcast', 'Rainy', 'Windy', 'Foggy'];
      const temp = Math.round(45 + Math.random() * 50);
      const condition = conditions[Math.floor(Math.random() * conditions.length)];
      return {
        location: toToolValue(city),
        temperature_f: toToolValue(temp),
        condition: toToolValue(condition),
        humidity_pct: toToolValue(Math.round(30 + Math.random() * 60)),
      };
    },
  },
  {
    def: {
      name: 'calculate',
      description: 'Evaluates a mathematical expression and returns the numeric result.',
      parameters: [
        { name: 'expression', type: 'string', description: 'Math expression (e.g. "2 + 3 * 4")', required: true },
      ],
      category: 'Math',
    },
    executor: async (args): Promise<Record<string, ToolValue>> => {
      const expr = getStringArg(args, 'expression') ?? '0';
      try {
        const sanitized = expr.replace(/[^0-9+\-*/().%\s^]/g, '');
        const val = Function(`"use strict"; return (${sanitized})`)();
        return { result: toToolValue(Number(val)), expression: toToolValue(expr) };
      } catch {
        return { error: toToolValue(`Invalid expression: ${expr}`) };
      }
    },
  },
  {
    def: {
      name: 'get_time',
      description: 'Returns the current date and time, optionally for a specific timezone.',
      parameters: [
        { name: 'timezone', type: 'string', description: 'IANA timezone (e.g. "America/New_York"). Defaults to UTC.', required: false },
      ],
      category: 'Utility',
    },
    executor: async (args): Promise<Record<string, ToolValue>> => {
      const tz = getStringArg(args, 'timezone') ?? 'UTC';
      try {
        const now = new Date();
        const formatted = now.toLocaleString('en-US', { timeZone: tz, dateStyle: 'full', timeStyle: 'long' });
        return { datetime: toToolValue(formatted), timezone: toToolValue(tz) };
      } catch {
        return { datetime: toToolValue(new Date().toISOString()), timezone: toToolValue('UTC'), note: toToolValue('Fell back to UTC — invalid timezone') };
      }
    },
  },
  {
    def: {
      name: 'random_number',
      description: 'Generates a random integer between min and max (inclusive).',
      parameters: [
        { name: 'min', type: 'number', description: 'Minimum value', required: true },
        { name: 'max', type: 'number', description: 'Maximum value', required: true },
      ],
      category: 'Math',
    },
    executor: async (args) => {
      const min = getNumberArg(args, 'min') ?? 1;
      const max = getNumberArg(args, 'max') ?? 100;
      const value = Math.floor(Math.random() * (max - min + 1)) + min;
      return { value: toToolValue(value), min: toToolValue(min), max: toToolValue(max) };
    },
  },
];

// ---------------------------------------------------------------------------
// Types for the execution trace
// ---------------------------------------------------------------------------

interface TraceStep {
  type: 'user' | 'tool_call' | 'tool_result' | 'response';
  content: string;
  detail?: ToolCall | ToolResult;
}

// ---------------------------------------------------------------------------
// Custom tool form state
// ---------------------------------------------------------------------------

interface ParamDraft {
  name: string;
  type: 'string' | 'number' | 'boolean';
  description: string;
  required: boolean;
}

const EMPTY_PARAM: ParamDraft = { name: '', type: 'string', description: '', required: true };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ToolsTab() {
  const loader = useModelLoader(ModelCategory.Language);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [autoExecute, setAutoExecute] = useState(true);
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [registeredTools, setRegisteredTools] = useState<ToolDefinition[]>([]);
  const [showToolForm, setShowToolForm] = useState(false);
  const [showRegistry, setShowRegistry] = useState(false);
  const traceRef = useRef<HTMLDivElement>(null);

  // Custom tool form state
  const [toolName, setToolName] = useState('');
  const [toolDesc, setToolDesc] = useState('');
  const [toolParams, setToolParams] = useState<ParamDraft[]>([{ ...EMPTY_PARAM }]);

  // Register demo tools on mount
  useEffect(() => {
    ToolCalling.clearTools();
    for (const { def, executor } of DEMO_TOOLS) {
      ToolCalling.registerTool(def, executor);
    }
    setRegisteredTools(ToolCalling.getRegisteredTools());
    return () => { ToolCalling.clearTools(); };
  }, []);

  // Auto-scroll trace
  useEffect(() => {
    traceRef.current?.scrollTo({ top: traceRef.current.scrollHeight, behavior: 'smooth' });
  }, [trace]);

  const refreshRegistry = useCallback(() => {
    setRegisteredTools(ToolCalling.getRegisteredTools());
  }, []);

  // -------------------------------------------------------------------------
  // Generate with tools
  // -------------------------------------------------------------------------

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || generating) return;

    if (loader.state !== 'ready') {
      const ok = await loader.ensure();
      if (!ok) return;
    }

    setInput('');
    setGenerating(true);
    setTrace([{ type: 'user', content: text }]);

    try {
      const result: ToolCallingResult = await ToolCalling.generateWithTools(text, {
        autoExecute,
        maxToolCalls: 5,
        temperature: 0.3,
        maxTokens: 512,
        format: ToolCallFormat.Default,
      });

      // Build trace from result
      const steps: TraceStep[] = [{ type: 'user', content: text }];

      for (let i = 0; i < result.toolCalls.length; i++) {
        const call = result.toolCalls[i];
        const argSummary = Object.entries(call.arguments)
          .map(([k, v]) => `${k}=${JSON.stringify('value' in v ? v.value : v)}`)
          .join(', ');
        steps.push({
          type: 'tool_call',
          content: `${call.toolName}(${argSummary})`,
          detail: call,
        });

        if (result.toolResults[i]) {
          const res = result.toolResults[i];
          const resultStr = res.success && res.result
            ? JSON.stringify(Object.fromEntries(Object.entries(res.result).map(([k, v]) => [k, 'value' in v ? v.value : v])), null, 2)
            : res.error ?? 'Unknown error';
          steps.push({
            type: 'tool_result',
            content: res.success ? resultStr : `Error: ${resultStr}`,
            detail: res,
          });
        }
      }

      if (result.text) {
        steps.push({ type: 'response', content: result.text });
      }

      setTrace(steps);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setTrace((prev) => [...prev, { type: 'response', content: `Error: ${msg}` }]);
    } finally {
      setGenerating(false);
    }
  }, [input, generating, autoExecute, loader]);

  // -------------------------------------------------------------------------
  // Register custom tool
  // -------------------------------------------------------------------------

  const addParam = () => setToolParams((p) => [...p, { ...EMPTY_PARAM }]);

  const updateParam = (idx: number, field: keyof ParamDraft, value: string | boolean) => {
    setToolParams((prev) => prev.map((p, i) => (i === idx ? { ...p, [field]: value } : p)));
  };

  const removeParam = (idx: number) => {
    setToolParams((prev) => prev.filter((_, i) => i !== idx));
  };

  const registerCustomTool = () => {
    const name = toolName.trim().replace(/\s+/g, '_').toLowerCase();
    const desc = toolDesc.trim();
    if (!name || !desc) return;

    const params = toolParams
      .filter((p) => p.name.trim())
      .map((p) => ({
        name: p.name.trim(),
        type: p.type as 'string' | 'number' | 'boolean',
        description: p.description.trim() || p.name.trim(),
        required: p.required,
      }));

    const def: ToolDefinition = { name, description: desc, parameters: params, category: 'Custom' };

    // Mock executor that returns the args back as acknowledgement
    const executor = async (args: Record<string, ToolValue>): Promise<Record<string, ToolValue>> => {
      const result: Record<string, ToolValue> = {
        status: toToolValue('executed'),
        tool: toToolValue(name),
      };
      for (const [k, v] of Object.entries(args)) {
        result[`input_${k}`] = v;
      }
      return result;
    };

    ToolCalling.registerTool(def, executor);
    refreshRegistry();
    setToolName('');
    setToolDesc('');
    setToolParams([{ ...EMPTY_PARAM }]);
    setShowToolForm(false);
  };

  const unregisterTool = (name: string) => {
    ToolCalling.unregisterTool(name);
    refreshRegistry();
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex-1 flex flex-col overflow-hidden h-full">
      <ModelBanner
        state={loader.state}
        progress={loader.progress}
        error={loader.error}
        onLoad={loader.ensure}
        label="LLM"
      />

      {/* Toolbar */}
      <div className="flex items-center gap-2 px-6 py-3 bg-white/40 backdrop-blur-sm border-b border-charcoal/10 flex-wrap">
        <button
          className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all whitespace-nowrap shadow-sm ${
            showRegistry 
              ? 'bg-coral text-white' 
              : 'bg-white border border-charcoal/20 text-charcoal hover:bg-charcoal/5'
          }`}
          onClick={() => { setShowRegistry(!showRegistry); setShowToolForm(false); }}
        >
          🔧 Tools ({registeredTools.length})
        </button>
        <button
          className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all whitespace-nowrap shadow-sm ${
            showToolForm 
              ? 'bg-coral text-white' 
              : 'bg-white border border-charcoal/20 text-charcoal hover:bg-charcoal/5'
          }`}
          onClick={() => { setShowToolForm(!showToolForm); setShowRegistry(false); }}
        >
          + Add Tool
        </button>
        <label className="flex items-center gap-2 text-xs text-charcoal/70 cursor-pointer font-medium">
          <input 
            type="checkbox" 
            checked={autoExecute} 
            onChange={(e) => setAutoExecute(e.target.checked)}
            className="w-4 h-4 accent-coral"
          />
          Auto-execute
        </label>
      </div>

      {/* Tool registry panel */}
      {showRegistry && (
        <div className="p-4 bg-slate-800 border-b border-slate-700 max-h-64 overflow-y-auto custom-scrollbar">
          <h4 className="text-sm font-semibold mb-3 text-slate-50">Registered Tools</h4>
          {registeredTools.length === 0 && <p className="text-sm text-slate-400">No tools registered</p>}
          <div className="flex flex-col gap-2">
            {registeredTools.map((t) => (
              <div key={t.name} className="p-3 bg-slate-700 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <strong className="text-sm text-slate-50">{t.name}</strong>
                    {t.category && (
                      <span className="px-1.5 py-0.5 text-[10px] font-semibold uppercase bg-slate-600 text-slate-300 rounded">
                        {t.category}
                      </span>
                    )}
                  </div>
                  <button 
                    className="px-2 py-0.5 text-sm font-bold text-slate-400 hover:text-slate-50 transition-colors"
                    onClick={() => unregisterTool(t.name)}
                  >
                    ×
                  </button>
                </div>
                <p className="text-xs text-slate-300 mb-2">{t.description}</p>
                {t.parameters.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {t.parameters.map((p) => (
                      <span key={p.name} className="px-2 py-0.5 text-[10px] bg-slate-800 text-slate-400 rounded font-mono">
                        {p.name}: {p.type}{p.required ? ' *' : ''}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Custom tool form */}
      {showToolForm && (
        <div className="p-4 bg-slate-800 border-b border-slate-700 max-h-96 overflow-y-auto custom-scrollbar">
          <h4 className="text-sm font-semibold mb-3 text-slate-50">Register Custom Tool</h4>
          <input
            className="w-full px-3 py-2 mb-2 border border-slate-700 rounded-lg bg-slate-700 text-slate-50 text-sm outline-none focus:border-primary transition-colors"
            placeholder="Tool name (e.g. search_web)"
            value={toolName}
            onChange={(e) => setToolName(e.target.value)}
          />
          <input
            className="w-full px-3 py-2 mb-3 border border-slate-700 rounded-lg bg-slate-700 text-slate-50 text-sm outline-none focus:border-primary transition-colors"
            placeholder="Description (e.g. Searches the web for a query)"
            value={toolDesc}
            onChange={(e) => setToolDesc(e.target.value)}
          />
          <div className="mb-3">
            <span className="block text-xs font-medium text-slate-400 mb-2">Parameters</span>
            {toolParams.map((p, i) => (
              <div key={i} className="flex gap-1.5 mb-1.5">
                <input
                  className="flex-1 px-2 py-1 border border-slate-700 rounded bg-slate-700 text-slate-50 text-xs outline-none focus:border-primary"
                  placeholder="name"
                  value={p.name}
                  onChange={(e) => updateParam(i, 'name', e.target.value)}
                />
                <select
                  className="px-2 py-1 border border-slate-700 rounded bg-slate-700 text-slate-50 text-xs outline-none focus:border-primary"
                  value={p.type}
                  onChange={(e) => updateParam(i, 'type', e.target.value)}
                >
                  <option value="string">string</option>
                  <option value="number">number</option>
                  <option value="boolean">boolean</option>
                </select>
                <input
                  className="flex-1 px-2 py-1 border border-slate-700 rounded bg-slate-700 text-slate-50 text-xs outline-none focus:border-primary"
                  placeholder="description"
                  value={p.description}
                  onChange={(e) => updateParam(i, 'description', e.target.value)}
                />
                <label className="flex items-center gap-1 px-2 text-xs text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={p.required} 
                    onChange={(e) => updateParam(i, 'required', e.target.checked)}
                    className="w-3 h-3"
                  />
                  req
                </label>
                {toolParams.length > 1 && (
                  <button 
                    className="px-2 py-1 text-xs rounded bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600"
                    onClick={() => removeParam(i)}
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            <button 
              className="px-2.5 py-1 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 font-medium hover:bg-slate-600 transition-all"
              onClick={addParam}
            >
              + Param
            </button>
          </div>
          <div className="flex gap-2 mb-2">
            <button 
              className="px-3 py-1.5 text-xs rounded-lg bg-primary hover:bg-primary-hover text-white font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              onClick={registerCustomTool} 
              disabled={!toolName.trim() || !toolDesc.trim()}
            >
              Register Tool
            </button>
            <button 
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600 transition-all"
              onClick={() => setShowToolForm(false)}
            >
              Cancel
            </button>
          </div>
          <p className="text-xs text-slate-400 italic">
            Custom tools use a mock executor that echoes back the arguments. Replace with real logic in code.
          </p>
        </div>
      )}

      {/* Execution trace */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar" ref={traceRef}>
        {trace.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-slate-400">
            <h3 className="text-slate-50 text-lg font-medium">Tool Calling</h3>
            <p className="text-sm max-w-md">{'Ask a question that requires tools — e.g. "What\'s the weather in Tokyo?" or "What is 42 * 17?"'}</p>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              <button 
                className="px-2.5 py-1 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600 transition-all"
                onClick={() => setInput('What is the weather in San Francisco?')}
              >
                🌤️ Weather
              </button>
              <button 
                className="px-2.5 py-1 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600 transition-all"
                onClick={() => setInput('What is 123 * 456 + 789?')}
              >
                🧮 Calculate
              </button>
              <button 
                className="px-2.5 py-1 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600 transition-all"
                onClick={() => setInput('What time is it in Tokyo?')}
              >
                🕐 Time
              </button>
              <button 
                className="px-2.5 py-1 text-xs rounded-lg bg-slate-700 border border-slate-600 text-slate-50 hover:bg-slate-600 transition-all"
                onClick={() => setInput('Give me a random number between 1 and 1000')}
              >
                🎲 Random
              </button>
            </div>
          </div>
        )}
        <div className="flex flex-col gap-2">
          {trace.map((step, i) => (
            <div key={i} className="p-3 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs font-semibold mb-1.5 text-slate-400">
                {step.type === 'user' && '👤 User'}
                {step.type === 'tool_call' && '🔧 Tool Call'}
                {step.type === 'tool_result' && '📦 Result'}
                {step.type === 'response' && '🤖 Response'}
              </div>
              <div className="text-xs text-slate-300 font-mono">
                <pre className="whitespace-pre-wrap break-words">{step.content}</pre>
              </div>
            </div>
          ))}
          {generating && (
            <div className="p-3 bg-slate-800 rounded-lg border border-slate-700">
              <div className="text-xs font-semibold text-slate-400">⏳ Generating...</div>
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <form 
        className="flex gap-2 p-3 px-4 border-t border-slate-700 bg-slate-900" 
        onSubmit={(e) => { e.preventDefault(); send(); }}
      >
        <input
          type="text"
          placeholder="Ask something that needs tools..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={generating}
          className="flex-1 px-3.5 py-2.5 border border-slate-700 rounded-lg bg-slate-700 text-slate-50 text-sm outline-none focus:border-primary transition-colors"
        />
        <button 
          type="submit" 
          className="px-4 py-2 rounded-lg bg-primary hover:bg-primary-hover text-white text-sm font-medium transition-all whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed" 
          disabled={!input.trim() || generating}
        >
          {generating ? 'Running...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
