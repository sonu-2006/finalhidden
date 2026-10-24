import React, { useEffect, useState } from 'react';
import { BarChart3, Download, FileText, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import { fetchExperiments } from '../api';
import { ExperimentSummary } from '../types';

export const ExperimentResults: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments().then((data) => {
      setExperiments(data);
      setLoading(false);
    });
  }, []);

  const handleExport = (expId: string, format: 'csv' | 'json') => {
    window.open(`/api/export/${expId}/${format}`, '_blank');
  };

  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <BarChart3 className="h-6 w-6 text-purple-400" />
            <span>Verified Experiment Results Dashboard</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Empirical results across 4 pilot benchmarks (80 trajectories, 4-bit NF4 quantized LLMs). All metrics verified against saved PyTorch probe checkpoints and evaluation logs.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => handleExport(experiments[0]?.id || 'endo_qwen_pilot20', 'csv')}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono border border-slate-700 flex items-center space-x-1.5 transition"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={() => handleExport(experiments[0]?.id || 'endo_qwen_pilot20', 'json')}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-mono border border-slate-700 flex items-center space-x-1.5 transition"
          >
            <FileText className="h-3.5 w-3.5" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* Main Benchmark Table Card */}
      <div className="glass-card rounded-2xl border border-slate-800 p-6 space-y-4">
        <h2 className="text-sm font-bold text-white">Full Cross-Benchmark Quantitative Comparison</h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900 text-slate-400 uppercase text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Dataset & Model</th>
                <th className="py-3 px-3 text-center">Trajectories</th>
                <th className="py-3 px-3 text-center">Probe Acc.</th>
                <th className="py-3 px-3 text-center">Safe F1</th>
                <th className="py-3 px-3 text-center">AtRisk F1</th>
                <th className="py-3 px-3 text-center">Hallu F1</th>
                <th className="py-3 px-3 text-center text-blue-400">EDR@K</th>
                <th className="py-3 px-3 text-center text-emerald-400">Lead Time</th>
                <th className="py-3 px-3 text-center">FPR</th>
                <th className="py-3 px-3 text-center text-amber-400">Hallu. Drop</th>
                <th className="py-3 px-3 text-center">Completeness</th>
                <th className="py-3 px-3 text-center">Export</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3.5 px-4 font-sans font-semibold text-white">
                    {exp.dataset}
                    <span className="block text-[11px] font-mono text-slate-400 font-normal">{exp.model}</span>
                  </td>
                  <td className="py-3.5 px-3 text-center text-slate-300">{exp.total_trajectories}</td>
                  <td className="py-3.5 px-3 text-center font-bold text-white">
                    {(exp.probe_accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-3 text-center text-slate-300">{exp.f1_safe.toFixed(3)}</td>
                  <td className="py-3.5 px-3 text-center text-slate-300">{exp.f1_atrisk.toFixed(3)}</td>
                  <td className="py-3.5 px-3 text-center text-slate-300">{exp.f1_hallucinating.toFixed(3)}</td>
                  <td className="py-3.5 px-3 text-center font-bold text-blue-400">
                    {(exp.early_detection_rate * 100).toFixed(0)}%
                  </td>
                  <td className="py-3.5 px-3 text-center font-bold text-emerald-400">
                    {exp.mean_lead_time_tokens.toFixed(2)}t
                  </td>
                  <td className="py-3.5 px-3 text-center text-slate-400">
                    {exp.false_positive_rate != null ? `${(exp.false_positive_rate * 100).toFixed(1)}%` : 'N/A*'}
                  </td>
                  <td className="py-3.5 px-3 text-center font-bold text-amber-400">
                    {(exp.hallucination_reduction * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-3 text-center text-slate-300">
                    {(exp.completeness * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-3 text-center">
                    <button
                      onClick={() => handleExport(exp.id, 'csv')}
                      className="text-blue-400 hover:text-blue-300 text-[11px] underline"
                    >
                      CSV
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix / Class Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <Layers className="h-4 w-4 text-blue-400" />
            <span>3-State Token Classification Dynamics</span>
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            The probe classifies tokens into 3 classes: <code className="text-emerald-400">Safe</code> (0), <code className="text-amber-400">At-Risk</code> (1), and <code className="text-rose-400">Hallucinating</code> (2). Safe token F1 scores reach up to <strong>0.952</strong>, showing high specificity against false alarms in biomedical prose.
          </p>

          <div className="grid grid-cols-3 gap-2 font-mono text-center text-xs pt-2">
            <div className="p-3 bg-emerald-950/40 rounded-lg border border-emerald-800/60">
              <span className="text-[10px] text-emerald-300 uppercase block">Safe F1</span>
              <span className="text-lg font-bold text-white">0.855 - 0.952</span>
            </div>
            <div className="p-3 bg-amber-950/40 rounded-lg border border-amber-800/60">
              <span className="text-[10px] text-amber-300 uppercase block">At-Risk F1</span>
              <span className="text-lg font-bold text-white">0.296 - 0.618</span>
            </div>
            <div className="p-3 bg-rose-950/40 rounded-lg border border-rose-800/60">
              <span className="text-[10px] text-rose-300 uppercase block">Hallu F1</span>
              <span className="text-lg font-bold text-white">0.148 - 0.526</span>
            </div>
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white flex items-center space-x-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            <span>Evaluation Metric Definitions</span>
          </h2>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li>
              <strong className="text-white">EDR@K (Early Detection Rate):</strong> Proportion of error cases where the critical trigger $t^*$ preceded error onset $t_\text{'{error}'}$ within tolerance $K$.
            </li>
            <li>
              <strong className="text-white">Mean Lead Time ($\Delta t$):</strong> Average number of generated tokens between probe alarm and first erroneous medical claim ($t_\text{'{error}'} - t^*$).
            </li>
            <li>
              <strong className="text-white">Output Completeness:</strong> Ratio of preserved tokens after simulated intervention compared to baseline unmonitored response length.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
