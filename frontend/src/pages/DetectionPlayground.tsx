import React, { useState } from 'react';
import { FlaskConical, Play, Sliders, ShieldCheck, AlertTriangle, RefreshCw, CheckCircle2 } from 'lucide-react';
import { runDetectionApi } from '../api';
import { DetectResponse } from '../types';

const PRESET_CASES = [
  {
    name: 'EndoDiag: Gastric Neoplasm Diagnostic Risk',
    expId: 'endo_qwen_pilot20',
    prompt: 'Patient with recurrent epigastric pain undergoing upper endoscopy. Antral biopsy reveals mucosal ulceration with atypical signet ring cells. What is the pathological diagnosis and staging guideline?',
    generatedText: 'The endoscopic examination of the gastric antrum reveals deep mucosal ulceration. Pathological assessment of signet ring cell morphology confirms advanced diffuse-type adenocarcinoma of the stomach with early infiltration into the muscularis propria requiring surgical staging.',
  },
  {
    name: 'MedHallu: Contraindication Hallucination',
    expId: 'medhallu_biomistral_pilot20',
    prompt: 'Explain the safety considerations and renal dosing guidelines for Metformin in patients presenting with Type 2 Diabetes and an estimated GFR of 35 mL/min/1.73m2.',
    generatedText: 'Metformin can be safely continued without any dose modification in patients with an eGFR of 35 mL/min/1.73m2, and there is no documented risk of lactic acidosis or renal impairment under current endocrinology society guidelines.',
  },
  {
    name: 'MedQA: Pharmacological Mechanism Query',
    expId: 'medqa_mistral_pilot20',
    prompt: 'A 45-year-old female presents with resistant hypertension. Which second-line antihypertensive agent acts as a direct renin inhibitor?',
    generatedText: 'Aliskiren acts as a direct competitive inhibitor of renin, preventing the initial conversion of angiotensinogen to angiotensin I in the renal systemic circulation.',
  },
  {
    name: 'PubMedQA: Biomedical Evidence-Based Reasoning',
    expId: 'pubmedqa_llama_pilot20',
    prompt: 'Does intensive glycemic control reduce microvascular complications in patients with newly diagnosed type 2 diabetes mellitus according to UKPDS longitudinal trial outcomes?',
    generatedText: 'Based on the UK Prospective Diabetes Study (UKPDS), intensive blood-glucose control significantly decreases the risk of microvascular complications such as retinopathy and nephropathy over 10-year follow-up.',
  },
];

export const DetectionPlayground: React.FC = () => {
  const [selectedExpId, setSelectedExpId] = useState('endo_qwen_pilot20');
  const [prompt, setPrompt] = useState(PRESET_CASES[0].prompt);
  const [generatedText, setGeneratedText] = useState(PRESET_CASES[0].generatedText);
  const [threshold, setThreshold] = useState(0.85);
  const [minTokens, setMinTokens] = useState(10);
  const [maWindow, setMaWindow] = useState(5);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectResponse | null>(null);

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await runDetectionApi({
        experiment_id: selectedExpId,
        prompt,
        generated_text: generatedText,
        threshold,
        min_tokens: minTokens,
        ma_window: maWindow,
      });
      setResult(res);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  const handlePresetChange = (idx: number) => {
    const p = PRESET_CASES[idx];
    setSelectedExpId(p.expId);
    setPrompt(p.prompt);
    setGeneratedText(p.generatedText);
    setResult(null);
  };

  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <FlaskConical className="h-6 w-6 text-emerald-400" />
          <span>Hallucination Detection & Pre-Emptive Intervention Playground</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Test candidate clinical responses against trained probe classifiers. Adjust the decision threshold $\theta$, observe real-time risk accumulation, and verify automated early truncation.
        </p>
      </div>

      {/* Preset Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-mono text-slate-400 mr-2">Load Preset Clinical Case:</span>
        {PRESET_CASES.map((preset, idx) => (
          <button
            key={preset.name}
            onClick={() => handlePresetChange(idx)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition"
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Main Grid: Controls + Input vs Outputs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Left Col: Inputs & Parameters */}
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <div>
            <label className="text-xs font-mono text-slate-300 uppercase block mb-1">
              Select Probe Model & Benchmark Calibration
            </label>
            <select
              value={selectedExpId}
              onChange={(e) => setSelectedExpId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-lg px-3 py-2.5 font-mono focus:outline-none focus:border-blue-500"
            >
              <option value="endo_qwen_pilot20">Qwen3-8B Probe (Calibrated on EndoDiag)</option>
              <option value="medhallu_biomistral_pilot20">BioMistral-7B Probe (Calibrated on MedHallu)</option>
              <option value="medqa_mistral_pilot20">Mistral-7B-Instruct-v0.3 Probe (Calibrated on MedQA)</option>
              <option value="pubmedqa_llama_pilot20">Llama-3.1-8B-Instruct Probe (Calibrated on PubMedQA)</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-mono text-slate-300 uppercase block mb-1">
              Clinical Medical Prompt / Vignette
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-3 font-sans focus:outline-none focus:border-blue-500"
            />
          </div>

          <div>
            <label className="text-xs font-mono text-slate-300 uppercase block mb-1">
              Candidate LLM Generated Text (to be evaluated)
            </label>
            <textarea
              value={generatedText}
              onChange={(e) => setGeneratedText(e.target.value)}
              rows={4}
              className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-3 font-sans focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Hyperparameters Slider Grid */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-mono text-slate-400">Intervention Risk Threshold ($\theta$):</span>
              <span className="font-mono font-bold text-amber-400">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-blue-500"
            />

            <div className="grid grid-cols-2 gap-3 pt-2">
              <div>
                <span className="font-mono text-slate-400 block mb-1">Min Safe Tokens:</span>
                <input
                  type="number"
                  value={minTokens}
                  onChange={(e) => setMinTokens(parseInt(e.target.value) || 5)}
                  className="w-full bg-slate-900 border border-slate-700 text-white p-1.5 rounded font-mono text-xs"
                />
              </div>
              <div>
                <span className="font-mono text-slate-400 block mb-1">MA Smoothing Window ($w$):</span>
                <input
                  type="number"
                  value={maWindow}
                  onChange={(e) => setMaWindow(parseInt(e.target.value) || 3)}
                  className="w-full bg-slate-900 border border-slate-700 text-white p-1.5 rounded font-mono text-xs"
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs uppercase tracking-wider font-mono flex items-center justify-center space-x-2 transition shadow-lg shadow-blue-600/30"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Running Probe Inference...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Evaluate Trajectory & Run Intervention</span>
              </>
            )}
          </button>
        </div>

        {/* Right Col: Output & Diagnostic Panel */}
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-5">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            <span>Detection Diagnostics & Intervention Outcome</span>
          </h2>

          {!result ? (
            <div className="p-12 text-center text-slate-400 font-mono text-xs border border-dashed border-slate-800 rounded-xl">
              Click &ldquo;Evaluate Trajectory&rdquo; above to run the probe and inspect real-time intervention behavior.
            </div>
          ) : (
            <div className="space-y-4">
              {/* Verdict Card */}
              <div
                className={`p-4 rounded-xl border flex items-start space-x-3 ${
                  result.triggered
                    ? 'bg-amber-950/40 border-amber-800/60 text-amber-200'
                    : 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200'
                }`}
              >
                {result.triggered ? (
                  <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
                )}
                <div>
                  <span className="font-bold block text-sm">{result.summary_verdict}</span>
                  <p className="text-xs mt-1 text-slate-300">
                    {result.triggered
                      ? `Probe detected high-risk latent uncertainty exceeding θ = ${result.threshold} at token position ${result.t_star}. Truncation triggered pre-emptively.`
                      : 'All token activations remained below the risk threshold. Response passed without intervention.'}
                  </p>
                </div>
              </div>

              {/* Timing & Stats Strip */}
              <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Critical Trigger ($t^*$)</span>
                  <span className="font-bold text-white text-sm">
                    {result.t_star !== null ? `Token #${result.t_star}` : 'None'}
                  </span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Inference Latency</span>
                  <span className="font-bold text-emerald-400 text-sm">{result.latency_ms} ms</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Tokens Evaluated</span>
                  <span className="font-bold text-blue-400 text-sm">{result.risk_scores.length}</span>
                </div>
              </div>

              {/* Truncated vs Raw Text Comparison */}
              <div className="space-y-3 pt-2">
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
                    Delivered Output (After Intervention)
                  </span>
                  <p className="text-xs text-white leading-relaxed font-sans">{result.truncated_text}</p>
                </div>

                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
                    Unmonitored Raw Output (With Hallucination)
                  </span>
                  <p className="text-xs text-slate-400 leading-relaxed font-sans">{result.full_text}</p>
                </div>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
