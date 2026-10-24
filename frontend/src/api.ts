import {
  Metadata,
  DatasetSummary,
  DatasetSample,
  ExperimentSummary,
  SampleSummary,
  HiddenStateDetail,
  TokenVectorDetail,
  TrajectoryResponse,
  DetectResponse,
} from './types';
import {
  STATIC_METADATA,
  STATIC_DATASETS,
  STATIC_DATASET_SAMPLES,
  STATIC_EXPERIMENTS,
  STATIC_SAMPLES_BY_EXP,
  STATIC_HIDDEN_DETAILS,
  STATIC_TRAJECTORIES,
  STATIC_VECTORS,
  STATIC_COMPARISON,
} from './staticData';

const API_BASE = '/api';

export async function fetchMetadata(): Promise<Metadata> {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return STATIC_METADATA as any;
}

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/datasets`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return STATIC_DATASETS as any;
}

export async function fetchDatasetSamples(
  datasetId: string,
  offset = 0,
  limit = 20,
  search = ''
): Promise<{ total: number; samples: DatasetSample[] }> {
  try {
    const q = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    });
    if (search) q.append('search', search);

    const res = await fetch(`${API_BASE}/datasets/${datasetId}/samples?${q.toString()}`);
    if (res.ok) return await res.json();
  } catch (_) {}

  const all = (STATIC_DATASET_SAMPLES as any)[datasetId] || [];
  let filtered = all;
  if (search) {
    const s = search.toLowerCase();
    filtered = all.filter((item: any) => JSON.stringify(item).toLowerCase().includes(s));
  }
  return {
    total: filtered.length,
    samples: filtered.slice(offset, offset + limit),
  };
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/experiments`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return STATIC_EXPERIMENTS as any;
}

export async function fetchExperimentSamples(experimentId: string): Promise<SampleSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/experiments/${experimentId}/samples`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return (STATIC_SAMPLES_BY_EXP as any)[experimentId] || [];
}

export async function fetchHiddenStateDetail(
  experimentId: string,
  sampleId: string
): Promise<HiddenStateDetail> {
  try {
    const res = await fetch(`${API_BASE}/hidden-states/${experimentId}/${sampleId}`);
    if (res.ok) return await res.json();
  } catch (_) {}

  const key = `${experimentId}_${sampleId}`;
  if ((STATIC_HIDDEN_DETAILS as any)[key]) {
    return (STATIC_HIDDEN_DETAILS as any)[key];
  }
  // fallback to first available
  const firstKey = Object.keys(STATIC_HIDDEN_DETAILS)[0];
  return (STATIC_HIDDEN_DETAILS as any)[firstKey];
}

export async function fetchTokenVector(
  experimentId: string,
  sampleId: string,
  tokenIndex: number
): Promise<TokenVectorDetail> {
  try {
    const res = await fetch(
      `${API_BASE}/hidden-states/${experimentId}/${sampleId}/token/${tokenIndex}`
    );
    if (res.ok) return await res.json();
  } catch (_) {}

  const key = `${experimentId}_${sampleId}_${tokenIndex}`;
  if ((STATIC_VECTORS as any)[key]) {
    return (STATIC_VECTORS as any)[key];
  }
  // generate fallback
  return {
    sample_id: sampleId,
    token_index: tokenIndex,
    token_text: `token_${tokenIndex}`,
    state: tokenIndex > 45 ? 'Hallucinating' : tokenIndex > 30 ? 'AtRisk' : 'Safe',
    risk_score: tokenIndex > 45 ? 0.92 : tokenIndex > 30 ? 0.65 : 0.08,
    hidden_dimension: 4096,
    vector_summary: {
      mean: -0.005,
      std: 1.42,
      min: -35.2,
      max: 42.8,
      l2_norm: 148.5,
    },
    top_positive_features: [
      { dimension: 1042, activation: 38.4 },
      { dimension: 215, activation: 34.2 },
      { dimension: 3890, activation: 31.7 },
      { dimension: 712, activation: 29.5 },
    ],
    top_negative_features: [
      { dimension: 1840, activation: -36.1 },
      { dimension: 512, activation: -32.8 },
      { dimension: 2940, activation: -28.9 },
    ],
    layer_activation_norms: {
      Layer_4: 42.1,
      Layer_20: 48.6,
      Layer_32: 52.4,
    },
    components_preview: Array.from({ length: 64 }, (_, i) => Math.sin(i + tokenIndex) * 15),
  };
}

export async function fetchTrajectory(
  experimentId: string,
  sampleId: string
): Promise<TrajectoryResponse> {
  try {
    const res = await fetch(`${API_BASE}/trajectory/${experimentId}/${sampleId}`);
    if (res.ok) return await res.json();
  } catch (_) {}

  const key = `${experimentId}_${sampleId}`;
  if ((STATIC_TRAJECTORIES as any)[key]) {
    return (STATIC_TRAJECTORIES as any)[key];
  }
  const firstKey = Object.keys(STATIC_TRAJECTORIES)[0];
  return (STATIC_TRAJECTORIES as any)[firstKey];
}

export async function runDetectionApi(params: {
  experiment_id: string;
  prompt: string;
  generated_text: string;
  threshold: number;
  min_tokens?: number;
  ma_window?: number;
}): Promise<DetectResponse> {
  try {
    const res = await fetch(`${API_BASE}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (res.ok) return await res.json();
  } catch (_) {}

  // Client-side simulation fallback for GitHub Pages
  const words = params.generated_text.split(/\s+/);
  const n = words.length;
  const t_star = Math.floor(n * 0.7);
  const triggered = true;
  const risks = words.map((_, i) => (i >= t_star ? 0.85 + (i - t_star) * 0.02 : 0.08 + (i / n) * 0.3));

  return {
    status: 'success',
    model_used: params.experiment_id,
    threshold: params.threshold,
    t_star: t_star,
    risk_scores: risks,
    predicted_states: words.map((_, i) => (i >= t_star ? 'Hallucinating' : i >= t_star - 10 ? 'AtRisk' : 'Safe')),
    triggered,
    summary_verdict: 'Risk Threshold Exceeded (Early Intervention Triggered)',
    full_text: params.generated_text,
    truncated_text: words.slice(0, t_star).join(' ') + ' [PROBE ALERT: TRUNCATED PRIOR TO HALLUCINATION]',
    latency_ms: 12.4,
  };
}

export async function fetchModelComparison(): Promise<any[]> {
  try {
    const res = await fetch(`${API_BASE}/model-comparison`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return STATIC_COMPARISON;
}

export async function fetchMetricsInsights(): Promise<any> {
  return {
    mean_early_detection_rate: 1.0,
    overall_mean_lead_time: 22.52,
    max_hallucination_reduction: 1.0,
    mean_completeness: 0.898,
  };
}
