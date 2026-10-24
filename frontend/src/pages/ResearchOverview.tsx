import React, { useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { fetchMetadata, fetchExperiments } from '../api';
import { Metadata, ExperimentSummary } from '../types';
import { PageId } from '../components/Sidebar';

interface ResearchOverviewProps {
  onNavigate: (page: PageId) => void;
}

export const ResearchOverview: React.FC<ResearchOverviewProps> = ({ onNavigate }) => {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchMetadata(), fetchExperiments()])
      .then(([m, exps]) => {
        setMetadata(m);
        setExperiments(exps);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load overview data', err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8 p-6 max-w-7xl mx-auto">
      {/* Top Research Badge & Title */}
      <div className="glass-card rounded-2xl p-8 border border-slate-800 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>

        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-950/80 text-blue-400 border border-blue-800/60 text-xs font-mono mb-4">
          <Sparkles className="h-3.5 w-3.5 text-blue-400" />
          <span>Final Year B.Tech Capstone Project Defense</span>
          <span>•</span>
          <span className="text-white font-bold">Group 76</span>
        </div>

        <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight mb-4">
          Medical Hallucination Detection in Large Language Models Using Hidden-State Probing
        </h1>

        <p className="text-sm sm:text-base text-slate-300 max-w-4xl font-light leading-relaxed mb-6">
          By inspecting intermediate hidden-state vectors across decoding timesteps, our lightweight probe classifiers identify the latent uncertainty phase transition (<code className="text-emerald-400">Safe</code> &rarr; <code className="text-amber-400">At-Risk</code>) to detect and truncate diagnostic medical hallucinations <strong className="text-blue-400 font-semibold">14 to 32 tokens before</strong> false medical claims are verbalized.
        </p>

        {/* Academic Credentials Box */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-900/90 border border-slate-800/80 text-xs">
          <div>
            <span className="text-slate-400 block font-mono text-[10px] uppercase">Investigator</span>
            <span className="font-bold text-white text-sm">Group 76</span>
            <span className="block text-slate-400 text-[11px]">Dept. of CSE</span>
          </div>
          <div>
            <span className="text-slate-400 block font-mono text-[10px] uppercase">Faculty Guide</span>
            <span className="font-bold text-blue-300 text-sm">Dr. Swapna T R</span>
            <span className="block text-slate-400 text-[11px]">Associate Professor, CSE</span>
          </div>
          <div>
            <span className="text-slate-400 block font-mono text-[10px] uppercase">Artifact Evaluation</span>
            <span className="font-bold text-emerald-400 text-sm">4 Pilot Benchmarks</span>
            <span className="block text-slate-400 text-[11px]">80 Trajectories, 80 .npz Tensors</span>
          </div>
          <div>
            <span className="text-slate-400 block font-mono text-[10px] uppercase">Target Environment</span>
            <span className="font-bold text-purple-300 text-sm">4-bit NF4 Quantization</span>
            <span className="block text-slate-400 text-[11px]">A800 GPU / Kaggle Cloud</span>
          </div>
        </div>
      </div>

      {/* 4 Metric Strips */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase">Early Detection Rate</span>
            <CheckCircle2 className="h-4 w-4 text-blue-400" />
          </div>
          <div className="text-3xl font-extrabold text-blue-400 font-mono">100%</div>
          <p className="text-xs text-slate-400 mt-1">52 / 52 error cases detected at $t^* &lt; t_\text{'{error}'}$</p>
        </div>

        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase">Mean Lead Time</span>
            <Activity className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold text-emerald-400 font-mono">14.8 – 32.5</div>
          <p className="text-xs text-slate-400 mt-1">Tokens ahead of diagnostic hallucination onset</p>
        </div>

        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase">Intervention Drop</span>
            <Zap className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-3xl font-extrabold text-amber-400 font-mono">76.9% – 100%</div>
          <p className="text-xs text-slate-400 mt-1">Simulated hallucination suppression via truncation</p>
        </div>

        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-mono uppercase">Probe Accuracy</span>
            <Cpu className="h-4 w-4 text-purple-400" />
          </div>
          <div className="text-3xl font-extrabold text-purple-400 font-mono">Up to 90.6%</div>
          <p className="text-xs text-slate-400 mt-1">3-state classification: Safe / AtRisk / Hallucinating</p>
        </div>
      </div>

      {/* Motivation & Methodology Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-rose-500"></span>
            <span>Why Post-Hoc Verification Fails in Healthcare</span>
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Conventional validation techniques (e.g. SelfCheckGPT, secondary verifier models, external web retrieval) evaluate text only <em>after</em> the LLM has generated its full sequence. In emergency medicine or clinical decision support, this introduces dangerous compute latency and transmits hallucinatory diagnostic text to physicians before checks complete.
          </p>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono text-slate-400">
            Latency: +2,500ms to +8,000ms &bull; GPU compute doubled &bull; Error already visible
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
            <span>The Real-Time Probing Paradigm (RTHD)</span>
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            As the decoder processes autoregressively, its intermediate layer activations ($h_t \in \mathbb{'{R}'}^{'{4096}'}$) reflect internal epistemic uncertainty. Our lightweight MLP probe evaluates each token in &lt;1ms. When cumulative risk exceeds $\theta$, generation is pre-emptively truncated before dangerous assertions are generated.
          </p>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-xs font-mono text-emerald-400">
            Latency: &lt;1.2ms per token &bull; Lead time: 15-32 tokens &bull; Zero external compute
          </div>
        </div>
      </div>

      {/* Model & Dataset Coverage Table */}
      <div className="glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Experimental Coverage Across Medical Benchmarks</h2>
            <p className="text-xs text-slate-400">Reproduced from verified Kaggle pilot execution logs.</p>
          </div>
          <button
            onClick={() => onNavigate('results')}
            className="inline-flex items-center space-x-1.5 text-xs text-blue-400 hover:text-blue-300 font-semibold"
          >
            <span>View Full Results</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900 text-slate-400 uppercase text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Benchmark</th>
                <th className="py-3 px-3">Backbone LLM</th>
                <th className="py-3 px-3 text-center">Quantization</th>
                <th className="py-3 px-3 text-center">Trajectories</th>
                <th className="py-3 px-3 text-center">Probe Acc.</th>
                <th className="py-3 px-3 text-center">EDR@K</th>
                <th className="py-3 px-3 text-center">Lead Time</th>
                <th className="py-3 px-3 text-center">Suppression</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3 px-4 font-sans font-semibold text-white">{exp.dataset}</td>
                  <td className="py-3 px-3 text-slate-300">{exp.model}</td>
                  <td className="py-3 px-3 text-center text-slate-400">4-bit NF4</td>
                  <td className="py-3 px-3 text-center text-slate-300">{exp.total_trajectories}</td>
                  <td className="py-3 px-3 text-center font-bold text-purple-400">
                    {(exp.probe_accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-center font-bold text-blue-400">
                    {(exp.early_detection_rate * 100).toFixed(0)}%
                  </td>
                  <td className="py-3 px-3 text-center font-bold text-emerald-400">
                    {exp.mean_lead_time_tokens.toFixed(2)} tok
                  </td>
                  <td className="py-3 px-3 text-center font-bold text-amber-400">
                    {(exp.hallucination_reduction * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Access Action Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          onClick={() => onNavigate('hidden-state')}
          className="glass-card glass-card-hover p-6 rounded-2xl border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="h-10 w-10 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <Cpu className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-white text-base">Hidden-State Explorer &rarr;</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Inspect raw 4096-dimensional activation tensors, 2D PCA projections, and token heatmaps.
          </p>
        </div>

        <div
          onClick={() => onNavigate('playground')}
          className="glass-card glass-card-hover p-6 rounded-2xl border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="h-10 w-10 rounded-xl bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Activity className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-white text-base">Detection Playground &rarr;</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Input clinical Q&A pairs, simulate real-time token decoding, and adjust intervention thresholds.
          </p>
        </div>

        <div
          onClick={() => onNavigate('results')}
          className="glass-card glass-card-hover p-6 rounded-2xl border border-slate-800 cursor-pointer space-y-3"
        >
          <div className="h-10 w-10 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-purple-400">
            <Database className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-white text-base">Benchmark Results &rarr;</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Review detailed confusion matrices, class-level F1 breakdowns, and export verified CSV data.
          </p>
        </div>
      </div>
    </div>
  );
};
