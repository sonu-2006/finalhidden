import React from 'react';
import { Network, Cpu, Shield, Zap, GitBranch, ArrowRight } from 'lucide-react';

export const Methodology: React.FC = () => {
  return (
    <div className="space-y-8 p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Network className="h-6 w-6 text-blue-400" />
          <span>System Architecture & Mathematical Formulation</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Technical specification of the Real-Time Hallucination Detection (RTHD) pipeline: token hidden-state caching, 3-state supervision, calibrated MLP probes, and simulated early intervention.
        </p>
      </div>

      {/* End-to-End Pipeline Schematic */}
      <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
        <h2 className="text-sm font-bold text-white">End-to-End Architectural Pipeline Flow</h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono text-blue-400 uppercase font-bold block">01. Decoding & Caching</span>
            <h4 className="text-sm font-bold text-white">Transformer Activation Tap</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Autoregressive generation via 4-bit NF4 backbone LLM. Internal hidden activations are captured at layers 4, 20, and 32 ($h_t \in \mathbb{'{R}'}^{'{4096}'}$).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono text-amber-400 uppercase font-bold block">02. Label Alignment</span>
            <h4 className="text-sm font-bold text-white">Three-State Annotation</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Token sequences aligned with ground truth via ROUGE-L / keyword matching: <code className="text-emerald-400">Safe</code>, <code className="text-amber-400">At-Risk</code> (pre-error window $w=10$), <code className="text-rose-400">Hallucinating</code>.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono text-purple-400 uppercase font-bold block">03. Probe Classification</span>
            <h4 className="text-sm font-bold text-white">Calibrated 2-Layer MLP</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Trained on hidden vectors with inverse class-frequency weighting: $\text{'{Linear}'}(4096, 64) \rightarrow \text{'{ReLU}'} \rightarrow \text{'{Linear}'}(64, 3)$.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold block">04. Pre-Emptive Stop</span>
            <h4 className="text-sm font-bold text-white">Critical Point Trigger</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Moving average risk $\bar{'{r}'}(t) \ge \theta$ triggers early truncation at $t^*$, suppressing the hallucination with average lead times of 14-32 tokens.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Formulations Card */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white">Mathematical Decision Mechanics</h2>

          <div className="space-y-4 text-xs font-mono text-slate-300">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-blue-400 font-bold block mb-1">1. Instantaneous Token Risk Score:</span>
              <p className="text-slate-200">
                {'r(t) = 1.0 - P(y_t = Safe | h_t) = P(AtRisk) + P(Hallucinating)'}
              </p>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-emerald-400 font-bold block mb-1">2. Moving Average Filtering (w=5):</span>
              <p className="text-slate-200">
                {'\bar{r}(t) = (1/w) \sum_{i=0}^{w-1} r(t - i)'}
              </p>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-amber-400 font-bold block mb-1">3. Critical Stopping Condition (t*):</span>
              <p className="text-slate-200">
                {'t* = min { t >= t_min | \bar{r}(t) >= theta }'}
              </p>
            </div>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-purple-400 font-bold block mb-1">4. Lead-Time Window (Δt):</span>
              <p className="text-slate-200">
                {'\Delta t = t_error - t*   (Validated early detection when \Delta t > 0)'}
              </p>
            </div>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white">MLP Probe Specification</h2>
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-xs space-y-2 text-slate-300">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Input Dimension:</span>
              <span className="text-white">4096 (LLM hidden activation)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Hidden Layer Dimension:</span>
              <span className="text-blue-400 font-bold">64 neurons (ReLU activation)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Output Classes:</span>
              <span className="text-white">3 (Safe, At-Risk, Hallucinating)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Loss Function:</span>
              <span className="text-white">Weighted Cross-Entropy ($w_c = 1/N_c$)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span className="text-slate-400">Optimization:</span>
              <span className="text-white">AdamW (lr=1e-3, batch=256, epochs=5)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Inference Latency:</span>
              <span className="text-emerald-400 font-bold">&lt;1.2 milliseconds per token</span>
            </div>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            By keeping the probe footprint minimal (&lt;1.05MB checkpoint size), probe inference can run asynchronously on CPU or GPU concurrent with autoregressive token sampling without stalling generation speed.
          </p>
        </div>
      </div>
    </div>
  );
};
