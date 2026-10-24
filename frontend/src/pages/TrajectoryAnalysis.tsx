import React, { useEffect, useState } from 'react';
import { Activity, Clock, ShieldCheck, AlertCircle, RefreshCw } from 'lucide-react';
import { fetchExperiments, fetchExperimentSamples, fetchTrajectory } from '../api';
import { ExperimentSummary, SampleSummary, TrajectoryResponse } from '../types';

export const TrajectoryAnalysis: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExpId, setSelectedExpId] = useState('endo_qwen_pilot20');

  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState('');

  const [trajectory, setTrajectory] = useState<TrajectoryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments().then((exps) => {
      setExperiments(exps);
      if (exps.length > 0) setSelectedExpId(exps[0].id);
    });
  }, []);

  useEffect(() => {
    if (!selectedExpId) return;
    fetchExperimentSamples(selectedExpId).then((sampleList) => {
      setSamples(sampleList);
      if (sampleList.length > 0) setSelectedSampleId(sampleList[0].id);
    });
  }, [selectedExpId]);

  useEffect(() => {
    if (!selectedExpId || !selectedSampleId) return;
    setLoading(true);
    fetchTrajectory(selectedExpId, selectedSampleId)
      .then((data) => {
        setTrajectory(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [selectedExpId, selectedSampleId]);

  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Header & Controls */}
      <div className="border-b border-slate-800 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Activity className="h-6 w-6 text-blue-400" />
            <span>Generation Trajectory & Lead-Time Analysis</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Examine how token-level risk signals accumulate across decoding steps. Quantify the exact early-warning lead time $\Delta t = t_\text{'{error}'} - t^*$ achieved before diagnostic errors manifest.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <select
            value={selectedExpId}
            onChange={(e) => setSelectedExpId(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 font-mono"
          >
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </select>

          <select
            value={selectedSampleId}
            onChange={(e) => setSelectedSampleId(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 font-mono max-w-xs"
          >
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                #{s.index} - {s.id.slice(0, 18)} (Lead: {s.lead_time ?? 0}t)
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading || !trajectory ? (
        <div className="glass-card rounded-2xl p-16 text-center border border-slate-800">
          <RefreshCw className="h-8 w-8 text-blue-400 animate-spin mx-auto mb-3" />
          <p className="text-xs font-mono text-slate-400">Loading risk trajectory curve...</p>
        </div>
      ) : (
        <>
          {/* Key Trajectory Metric Strip */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Critical Trigger ($t^*$)</span>
              <div className="text-2xl font-bold text-amber-400 font-mono mt-1">
                {trajectory.t_star !== null ? `Token #${trajectory.t_star}` : 'Not Triggered'}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">First point exceeding $\theta = {trajectory.threshold}$</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Error Onset ($t_\text{'{error}'}$)</span>
              <div className="text-2xl font-bold text-rose-400 font-mono mt-1">
                {trajectory.t_error !== null ? `Token #${trajectory.t_error}` : 'No Error Label'}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Ground-truth heuristic deviation point</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Early Detection Lead Time</span>
              <div className="text-2xl font-bold text-emerald-400 font-mono mt-1">
                {trajectory.lead_time !== null ? `+${trajectory.lead_time} Tokens` : 'N/A'}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Pre-emptive window prior to hallucination</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Output Completeness</span>
              <div className="text-2xl font-bold text-blue-400 font-mono mt-1">
                {(trajectory.completeness_ratio * 100).toFixed(1)}%
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Fraction of clinical prompt context preserved</p>
            </div>
          </div>

          {/* Trajectory Curve Visualization */}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-white">Risk Score Dynamics Across Response Decoding</h2>
                <p className="text-xs text-slate-400">
                  Instantaneous token risk $r(t)$ and moving average $\bar{'{r}'}(t)$ against threshold line $\theta = {trajectory.threshold}$.
                </p>
              </div>

              <div className="flex items-center space-x-3 text-xs font-mono">
                <span className="flex items-center space-x-1">
                  <span className="h-0.5 w-4 bg-blue-500"></span>
                  <span className="text-slate-300">Raw Risk</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="h-0.5 w-4 bg-emerald-400"></span>
                  <span className="text-slate-300">MA Window (w=5)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="h-0.5 w-4 bg-rose-500 border-b border-dashed"></span>
                  <span className="text-slate-300">Threshold ($\theta$)</span>
                </span>
              </div>
            </div>

            {/* SVG Trajectory Chart */}
            <div className="h-72 p-4 bg-slate-950 rounded-xl border border-slate-800/80 relative">
              <svg className="w-full h-full" viewBox={`0 0 ${Math.max(64, trajectory.total_tokens) * 10} 100`} preserveAspectRatio="none">
                {/* Horizontal Threshold line at 100 - (theta * 100) */}
                <line
                  x1="0"
                  y1={100 - trajectory.threshold * 100}
                  x2={trajectory.total_tokens * 10}
                  y2={100 - trajectory.threshold * 100}
                  stroke="#ef4444"
                  strokeWidth="1"
                  strokeDasharray="4 2"
                />

                {/* Vertical Line at t_star */}
                {trajectory.t_star != null && (
                  <line
                    x1={trajectory.t_star * 10}
                    y1="0"
                    x2={trajectory.t_star * 10}
                    y2="100"
                    stroke="#f59e0b"
                    strokeWidth="1.5"
                    strokeDasharray="2 2"
                  />
                )}

                {/* Vertical Line at t_error */}
                {trajectory.t_error != null && (
                  <line
                    x1={trajectory.t_error * 10}
                    y1="0"
                    x2={trajectory.t_error * 10}
                    y2="100"
                    stroke="#ef4444"
                    strokeWidth="1.5"
                  />
                )}

                {/* Raw Risk Line */}
                <polyline
                  fill="none"
                  stroke="rgba(59, 130, 246, 0.4)"
                  strokeWidth="1"
                  points={trajectory.points.map((p) => `${p.token_index * 10},${100 - p.risk * 100}`).join(' ')}
                />

                {/* Smoothed Risk Line */}
                <polyline
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2"
                  points={trajectory.points.map((p) => `${p.token_index * 10},${100 - p.smoothed_risk * 100}`).join(' ')}
                />
              </svg>

              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 mt-2">
                <span>Token t = 0 (Start)</span>
                {trajectory.t_star !== null && (
                  <span className="text-amber-400 font-bold">t* = Token #{trajectory.t_star} (Alert Trigger)</span>
                )}
                {trajectory.t_error !== null && (
                  <span className="text-rose-400 font-bold">t_error = Token #{trajectory.t_error} (Error Onset)</span>
                )}
                <span>Token t = {trajectory.total_tokens - 1} (End)</span>
              </div>
            </div>
          </div>

          {/* Intervention Text Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
              <span className="font-mono text-[10px] uppercase text-emerald-400 font-bold block">
                Pre-Emptively Truncated Generation (With Probe Intervention)
              </span>
              <p className="text-xs text-white leading-relaxed font-sans p-3 bg-slate-950 rounded-xl border border-slate-800">
                {trajectory.truncated_text}
              </p>
              <div className="text-[11px] text-slate-400 font-mono">
                Preserved context: {(trajectory.completeness_ratio * 100).toFixed(1)}% &bull; Diagnostic error suppressed.
              </div>
            </div>

            <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
              <span className="font-mono text-[10px] uppercase text-rose-400 font-bold block">
                Unmonitored Raw Output (Baseline)
              </span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans p-3 bg-slate-950 rounded-xl border border-slate-800">
                {trajectory.full_text}
              </p>
              <div className="text-[11px] text-slate-400 font-mono">
                Total tokens: {trajectory.total_tokens} &bull; Emitted false diagnostic assertion.
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
