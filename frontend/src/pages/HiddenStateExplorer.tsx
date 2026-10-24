import React, { useEffect, useState, useMemo } from 'react';
import {
  Cpu,
  Layers,
  Sliders,
  Sparkles,
  Info,
  Activity,
  AlertTriangle,
  CheckCircle2,
  GitCompare,
  Eye,
  Maximize2,
  RefreshCw,
} from 'lucide-react';
import {
  fetchExperiments,
  fetchExperimentSamples,
  fetchHiddenStateDetail,
  fetchTokenVector,
} from '../api';
import {
  ExperimentSummary,
  SampleSummary,
  HiddenStateDetail,
  TokenVectorDetail,
  TokenState,
} from '../types';

export const HiddenStateExplorer: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>('endo_qwen_pilot20');

  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');

  const [selectedLayer, setSelectedLayer] = useState<string>('mean');
  const [detail, setDetail] = useState<HiddenStateDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Selected Token for deep inspection
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0);
  const [vectorDetail, setVectorDetail] = useState<TokenVectorDetail | null>(null);
  const [vectorLoading, setVectorLoading] = useState<boolean>(false);

  // Comparison Token
  const [compareTokenIndex, setCompareTokenIndex] = useState<number | null>(null);
  const [compareVector, setCompareVector] = useState<TokenVectorDetail | null>(null);
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);

  // 1. Load Experiments
  useEffect(() => {
    fetchExperiments().then((exps) => {
      setExperiments(exps);
      if (exps.length > 0) {
        setSelectedExpId(exps[0].id);
      }
    });
  }, []);

  // 2. Load Samples for selected experiment
  useEffect(() => {
    if (!selectedExpId) return;
    setLoading(true);
    fetchExperimentSamples(selectedExpId).then((sampleList) => {
      setSamples(sampleList);
      if (sampleList.length > 0) {
        setSelectedSampleId(sampleList[0].id);
      }
    });
  }, [selectedExpId]);

  // 3. Load Hidden State Details for selected sample
  useEffect(() => {
    if (!selectedExpId || !selectedSampleId) return;
    setLoading(true);
    fetchHiddenStateDetail(selectedExpId, selectedSampleId)
      .then((data) => {
        setDetail(data);
        const initialToken = data.t_star !== null && data.t_star !== undefined ? data.t_star : 0;
        setSelectedTokenIndex(initialToken);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [selectedExpId, selectedSampleId]);

  // 4. Load Vector Details when selectedTokenIndex changes
  useEffect(() => {
    if (!selectedExpId || !selectedSampleId) return;
    setVectorLoading(true);
    fetchTokenVector(selectedExpId, selectedSampleId, selectedTokenIndex)
      .then((vec) => {
        setVectorDetail(vec);
        setVectorLoading(false);
      })
      .catch(() => setVectorLoading(false));
  }, [selectedExpId, selectedSampleId, selectedTokenIndex]);

  // 5. Load Comparison Vector if compareTokenIndex changes
  useEffect(() => {
    if (!selectedExpId || !selectedSampleId || compareTokenIndex === null) {
      setCompareVector(null);
      return;
    }
    fetchTokenVector(selectedExpId, selectedSampleId, compareTokenIndex).then((vec) => {
      setCompareVector(vec);
    });
  }, [selectedExpId, selectedSampleId, compareTokenIndex]);

  const selectedTokenObj: TokenState | undefined = useMemo(() => {
    if (!detail) return undefined;
    return detail.tokens.find((t) => t.index === selectedTokenIndex);
  }, [detail, selectedTokenIndex]);

  return (
    <div className="space-y-6 p-6 max-w-7xl mx-auto">
      {/* Page Title & Explanation */}
      <div className="border-b border-slate-800 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="h-2.5 w-2.5 rounded-full bg-blue-500 animate-pulse"></span>
            <span className="text-xs font-mono uppercase text-blue-400 font-bold tracking-wider">
              FLAGSHIP RESEARCH MODULE
            </span>
          </div>
          <h1 className="text-2xl font-bold text-white mt-1">Hidden-State Representation Explorer</h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Directly inspect the 4,096-dimensional activation vectors extracted from intermediate transformer layers, examine 2D PCA geometric transitions, and verify probe risk scores.
          </p>
        </div>

        {/* Global Controls: Experiment & Sample Selector */}
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Experiment Benchmark</label>
            <select
              value={selectedExpId}
              onChange={(e) => setSelectedExpId(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 font-mono focus:outline-none focus:border-blue-500"
            >
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Sample Trajectory</label>
            <select
              value={selectedSampleId}
              onChange={(e) => setSelectedSampleId(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 font-mono focus:outline-none focus:border-blue-500 max-w-xs"
            >
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  #{s.index} - {s.id.slice(0, 18)} ({s.verdict})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[10px] font-mono uppercase text-slate-400 block mb-1">Layer Projection</label>
            <select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 font-mono focus:outline-none focus:border-blue-500"
            >
              <option value="mean">Mean-Pooled (Default)</option>
              <option value="layer4">Layer 4 (Early Semantic)</option>
              <option value="layer20">Layer 20 (Intermediate)</option>
              <option value="layer32">Layer 32 (Late Output)</option>
            </select>
          </div>
        </div>
      </div>

      {loading || !detail ? (
        <div className="glass-card rounded-2xl p-16 text-center border border-slate-800">
          <RefreshCw className="h-8 w-8 text-blue-400 animate-spin mx-auto mb-3" />
          <p className="text-sm font-mono text-slate-300">Loading authentic hidden-state tensors from disk (.npz)...</p>
        </div>
      ) : (
        <>
          {/* SECTION B: Tensor Shape & Tensor Summary Card */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Tensor Shape</span>
              <div className="text-lg font-bold text-blue-400 font-mono mt-1">
                [{detail.tensor_shape.join(', ')}]
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                [batch=1, seq_len={detail.sequence_length}, hidden_dim={detail.hidden_dimension}]
              </p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Extracted Layers</span>
              <div className="text-lg font-bold text-purple-400 font-mono mt-1">
                Layers {detail.extracted_layers.join(', ')}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">Transformer intermediate taps</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Mean L2 Vector Norm</span>
              <div className="text-lg font-bold text-emerald-400 font-mono mt-1">
                {detail.summary_stats.l2_norm_mean.toFixed(2)}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Min: {detail.summary_stats.min.toFixed(2)} &bull; Max: {detail.summary_stats.max.toFixed(2)}
              </p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono text-slate-400 uppercase block">Critical Detection</span>
              <div className="text-lg font-bold text-amber-400 font-mono mt-1">
                {detail.t_star !== null ? `Triggered at t* = ${detail.t_star}` : 'No Error Trigger'}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                {detail.lead_time !== null ? `Lead Time: +${detail.lead_time} tokens` : 'Safe Trajectory'}
              </p>
            </div>
          </div>

          {/* Prompt & Generated Context Card */}
          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-mono text-slate-400 uppercase text-[10px] font-bold">Medical Query & Ground Truth</span>
              {detail.reference_answer && (
                <span className="font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/60 text-[10px]">
                  Ref: {detail.reference_answer.slice(0, 45)}
                </span>
              )}
            </div>
            <p className="text-slate-200 font-sans italic">&ldquo;{detail.prompt}&rdquo;</p>
          </div>

          {/* SECTION C: Synchronized Token-by-Token Sequence Timeline */}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-blue-400" />
                  <span>Token Sequence & Representation State Timeline</span>
                </h3>
                <p className="text-[11px] text-slate-400">
                  Click any token to inspect its authentic 4096-d vector, layer projections, and probe risk score.
                </p>
              </div>

              {/* State Legend */}
              <div className="flex items-center space-x-3 text-[11px] font-mono">
                <span className="flex items-center space-x-1">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-500"></span>
                  <span className="text-slate-300">Safe (r &lt; 0.4)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="h-2.5 w-2.5 rounded-full bg-amber-500"></span>
                  <span className="text-slate-300">At-Risk (0.4 &le; r &lt; 0.8)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="h-2.5 w-2.5 rounded-full bg-rose-500"></span>
                  <span className="text-slate-300">Hallucinating (r &ge; 0.8)</span>
                </span>
              </div>
            </div>

            {/* Token Pills Container */}
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800/90 max-h-56 overflow-y-auto flex flex-wrap gap-1.5 leading-relaxed font-mono text-xs">
              {detail.tokens.map((tok) => {
                const isSelected = tok.index === selectedTokenIndex;
                const isCompared = tok.index === compareTokenIndex;

                let stateColor = 'bg-emerald-950/40 text-emerald-300 border-emerald-800/50 hover:bg-emerald-900/50';
                if (tok.state === 'AtRisk') {
                  stateColor = 'bg-amber-950/40 text-amber-300 border-amber-800/50 hover:bg-amber-900/50';
                } else if (tok.state === 'Hallucinating') {
                  stateColor = 'bg-rose-950/40 text-rose-300 border-rose-800/50 hover:bg-rose-900/50';
                }

                return (
                  <button
                    key={tok.index}
                    onClick={() => {
                      if (isCompareMode) {
                        setCompareTokenIndex(tok.index);
                      } else {
                        setSelectedTokenIndex(tok.index);
                      }
                    }}
                    className={`px-2 py-1 rounded text-[11px] border transition relative flex items-center space-x-1 ${stateColor} ${
                      isSelected
                        ? 'ring-2 ring-blue-500 font-bold shadow-lg scale-105 z-10'
                        : isCompared
                        ? 'ring-2 ring-purple-500 font-bold scale-105 z-10'
                        : ''
                    }`}
                  >
                    <span>{tok.token_text}</span>
                    <span className="text-[9px] opacity-75 font-sans">({tok.risk_score.toFixed(2)})</span>
                    {tok.is_critical_trigger && (
                      <span className="h-2 w-2 rounded-full bg-amber-400 absolute -top-1 -right-1 animate-ping"></span>
                    )}
                    {tok.is_error_onset && (
                      <span className="h-2 w-2 rounded-full bg-rose-500 absolute -top-1 -right-1"></span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Quick Context Strip */}
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 pt-1">
              <span>
                Selected: <strong className="text-white">Token #{selectedTokenIndex}</strong> (&ldquo;{selectedTokenObj?.token_text}&rdquo;) &bull; State: <span className="text-blue-400 font-bold">{selectedTokenObj?.state}</span> &bull; Risk $r(t)$: <span className="text-white font-bold">{selectedTokenObj?.risk_score.toFixed(4)}</span>
              </span>
              <button
                onClick={() => setIsCompareMode(!isCompareMode)}
                className={`px-2.5 py-1 rounded text-[11px] border flex items-center space-x-1 transition ${
                  isCompareMode
                    ? 'bg-purple-600 text-white border-purple-500'
                    : 'bg-slate-900 text-slate-300 border-slate-700 hover:text-white'
                }`}
              >
                <GitCompare className="h-3 w-3" />
                <span>{isCompareMode ? 'Comparison Mode Active (Click 2nd Token)' : 'Compare Tokens'}</span>
              </button>
            </div>
          </div>

          {/* SECTION D & F: Heatmap Matrix + 2D PCA Dimensionality Reduction */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Heatmap Card */}
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                    <Layers className="h-4 w-4 text-emerald-400" />
                    <span>Hidden-State Matrix Heatmap</span>
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    X-axis: Tokens (0 to {detail.sequence_length - 1}) &bull; Y-axis: Sampled Dimensions (32 of 4096)
                  </p>
                </div>
                <div className="flex items-center space-x-2 text-[10px] font-mono text-slate-400">
                  <span className="h-2 w-6 bg-blue-900 rounded"></span>
                  <span>-Act</span>
                  <span className="h-2 w-6 bg-slate-800 rounded"></span>
                  <span>0</span>
                  <span className="h-2 w-6 bg-cyan-400 rounded"></span>
                  <span>+Act</span>
                </div>
              </div>

              {/* Heatmap Canvas / SVG Visualizer */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 overflow-x-auto">
                <div
                  className="grid gap-[1px]"
                  style={{
                    gridTemplateColumns: `repeat(${detail.sequence_length}, minmax(8px, 1fr))`,
                    gridTemplateRows: 'repeat(32, 6px)',
                  }}
                >
                  {detail.heatmap_feature_indices.map((featIdx, rowIdx) =>
                    detail.heatmap_matrix.map((row, colIdx) => {
                      const val = row[rowIdx] || 0;
                      // Color mapping
                      let bg = '#0f172a';
                      if (val > 10) bg = '#22d3ee';
                      else if (val > 2) bg = '#0284c7';
                      else if (val > 0) bg = '#0369a1';
                      else if (val < -10) bg = '#6366f1';
                      else if (val < -2) bg = '#4338ca';

                      const isSelectedCol = colIdx === selectedTokenIndex;

                      return (
                        <div
                          key={`${rowIdx}-${colIdx}`}
                          onClick={() => setSelectedTokenIndex(colIdx)}
                          title={`Token ${colIdx} (${detail.tokens[colIdx]?.token_text}), Dim ${featIdx}: ${val.toFixed(2)}`}
                          className={`cursor-pointer transition-opacity ${
                            isSelectedCol ? 'ring-1 ring-yellow-400 z-10' : 'hover:opacity-80'
                          }`}
                          style={{ backgroundColor: bg }}
                        />
                      );
                    })
                  )}
                </div>
              </div>
              <p className="text-[11px] text-slate-500 font-mono italic">
                *Clicking any column in the activation matrix synchronizes token selection across all views.
              </p>
            </div>

            {/* PCA 2D Dimensionality Reduction Scatter Plot */}
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                    <Cpu className="h-4 w-4 text-purple-400" />
                    <span>2D PCA Projection (Geometric Phase Transition)</span>
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    sklearn.decomposition.PCA: $\mathbb{'{R}'}^{'{4096}'} \rightarrow \mathbb{'{R}'}^2$. Demonstrating epistemic cluster shift.
                  </p>
                </div>
              </div>

              {/* 2D Scatter Canvas */}
              <div className="h-64 p-4 bg-slate-950 rounded-xl border border-slate-800/80 relative flex items-center justify-center overflow-hidden">
                <svg className="w-full h-full" viewBox="-50 -50 100 100">
                  {/* Grid crosshairs */}
                  <line x1="-50" y1="0" x2="50" y2="0" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" />
                  <line x1="0" y1="-50" x2="0" y2="50" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" />

                  {/* Connecting Trajectory Line */}
                  <polyline
                    fill="none"
                    stroke="rgba(59, 130, 246, 0.25)"
                    strokeWidth="1"
                    points={detail.pca_projection
                      .map((p) => {
                        // scale to fit -40 to 40
                        const sx = Math.max(-45, Math.min(45, p.x * 0.4));
                        const sy = Math.max(-45, Math.min(45, p.y * 0.4));
                        return `${sx},${sy}`;
                      })
                      .join(' ')}
                  />

                  {/* Points */}
                  {detail.pca_projection.map((pt) => {
                    const sx = Math.max(-45, Math.min(45, pt.x * 0.4));
                    const sy = Math.max(-45, Math.min(45, pt.y * 0.4));
                    const isSelected = pt.token_index === selectedTokenIndex;
                    let fill = '#10b981';
                    if (pt.state === 'AtRisk') fill = '#f59e0b';
                    if (pt.state === 'Hallucinating') fill = '#ef4444';

                    return (
                      <g key={pt.token_index} className="cursor-pointer" onClick={() => setSelectedTokenIndex(pt.token_index)}>
                        <circle
                          cx={sx}
                          cy={sy}
                          r={isSelected ? 3.5 : 2}
                          fill={fill}
                          stroke={isSelected ? '#ffffff' : 'none'}
                          strokeWidth={isSelected ? 1 : 0}
                        />
                      </g>
                    );
                  })}
                </svg>

                <div className="absolute bottom-2 right-2 text-[10px] font-mono text-slate-500">
                  PC1 vs PC2 Coordinates
                </div>
              </div>
              <p className="text-[11px] text-slate-400 leading-snug">
                Notice how safe tokens cluster tightly, while tokens at risk disperse across principal directions prior to hallucinating error emission.
              </p>
            </div>
          </div>

          {/* SECTION E: Token Vector Inspector (Detailed Numerical Deep-Dive) */}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Sliders className="h-4 w-4 text-amber-400" />
                  <span>Token Vector Numerical Inspector: Token #{selectedTokenIndex} (&ldquo;{vectorDetail?.token_text}&rdquo;)</span>
                </h3>
                <p className="text-[11px] text-slate-400">
                  Full dimensional breakdown of the $h_{'{' + selectedTokenIndex + '}'} \in \mathbb{'{R}'}^{'{4096}'}$ representation vector.
                </p>
              </div>

              {vectorDetail && (
                <div className="flex items-center space-x-2 font-mono text-xs">
                  <span className="px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                    L2 Norm: {vectorDetail.vector_summary.l2_norm.toFixed(2)}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-800">
                    Mean: {vectorDetail.vector_summary.mean.toFixed(4)}
                  </span>
                </div>
              )}
            </div>

            {vectorLoading || !vectorDetail ? (
              <div className="p-8 text-center text-xs font-mono text-slate-400">Loading vector channels...</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
                {/* Col 1: Top Positive Activation Channels */}
                <div className="space-y-2">
                  <span className="font-mono text-slate-400 uppercase text-[10px] font-bold block">
                    Top Positive Dimensions (Excitation)
                  </span>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono space-y-1.5 max-h-48 overflow-y-auto">
                    {vectorDetail.top_positive_features.map((feat) => (
                      <div key={feat.dimension} className="flex items-center justify-between">
                        <span className="text-slate-400">Channel #{feat.dimension}</span>
                        <span className="text-emerald-400 font-bold">+{feat.activation.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Col 2: Top Negative Activation Channels */}
                <div className="space-y-2">
                  <span className="font-mono text-slate-400 uppercase text-[10px] font-bold block">
                    Top Negative Dimensions (Inhibition)
                  </span>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono space-y-1.5 max-h-48 overflow-y-auto">
                    {vectorDetail.top_negative_features.map((feat) => (
                      <div key={feat.dimension} className="flex items-center justify-between">
                        <span className="text-slate-400">Channel #{feat.dimension}</span>
                        <span className="text-rose-400 font-bold">{feat.activation.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Col 3: Layer-by-Layer Activation Norms */}
                <div className="space-y-2">
                  <span className="font-mono text-slate-400 uppercase text-[10px] font-bold block">
                    Layer-by-Layer Norm Distribution
                  </span>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono space-y-3">
                    {Object.entries(vectorDetail.layer_activation_norms).map(([layer, norm]) => (
                      <div key={layer}>
                        <div className="flex justify-between mb-1">
                          <span className="text-slate-300">{layer}</span>
                          <span className="text-purple-400 font-bold">{norm}</span>
                        </div>
                        <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-purple-500 h-1.5 rounded-full"
                            style={{ width: `${Math.min(100, (norm / 80) * 100)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Component Inspection Preview Table */}
            {vectorDetail && (
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <span className="font-mono text-slate-400 uppercase text-[10px] font-bold block">
                  First 64 Vector Floating-Point Components ($h_{'{' + selectedTokenIndex + ', 0:63}'}$)
                </span>
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-[10px] font-mono text-slate-300 grid grid-cols-8 gap-2">
                  {vectorDetail.components_preview.map((comp, idx) => (
                    <div key={idx} className="p-1 rounded bg-slate-900/60 text-center">
                      <span className="text-slate-500 block text-[9px]">d_{idx}</span>
                      <span className={comp >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{comp.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
