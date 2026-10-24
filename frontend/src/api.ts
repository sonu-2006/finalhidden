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

const API_BASE = '/api';

export async function fetchMetadata(): Promise<Metadata> {
  const res = await fetch(`${API_BASE}/metadata`);
  if (!res.ok) throw new Error('Failed to fetch metadata');
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function fetchDatasetSamples(
  datasetId: string,
  offset = 0,
  limit = 20,
  search = ''
): Promise<{ total: number; samples: DatasetSample[] }> {
  const q = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });
  if (search) q.append('search', search);

  const res = await fetch(`${API_BASE}/datasets/${datasetId}/samples?${q.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch dataset samples');
  return res.json();
}

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return res.json();
}

export async function fetchExperimentSamples(experimentId: string): Promise<SampleSummary[]> {
  const res = await fetch(`${API_BASE}/experiments/${experimentId}/samples`);
  if (!res.ok) throw new Error(`Failed to fetch samples for ${experimentId}`);
  return res.json();
}

export async function fetchHiddenStateDetail(
  experimentId: string,
  sampleId: string
): Promise<HiddenStateDetail> {
  const res = await fetch(`${API_BASE}/hidden-states/${experimentId}/${sampleId}`);
  if (!res.ok) throw new Error(`Failed to fetch hidden states for ${sampleId}`);
  return res.json();
}

export async function fetchTokenVector(
  experimentId: string,
  sampleId: string,
  tokenIndex: number
): Promise<TokenVectorDetail> {
  const res = await fetch(
    `${API_BASE}/hidden-states/${experimentId}/${sampleId}/token/${tokenIndex}`
  );
  if (!res.ok) throw new Error(`Failed to fetch vector for token ${tokenIndex}`);
  return res.json();
}

export async function fetchTrajectory(
  experimentId: string,
  sampleId: string
): Promise<TrajectoryResponse> {
  const res = await fetch(`${API_BASE}/trajectory/${experimentId}/${sampleId}`);
  if (!res.ok) throw new Error(`Failed to fetch trajectory for ${sampleId}`);
  return res.json();
}

export async function runDetectionApi(params: {
  experiment_id: string;
  prompt: string;
  generated_text: string;
  threshold: number;
  min_tokens?: number;
  ma_window?: number;
}): Promise<DetectResponse> {
  const res = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('Failed to run hallucination detection');
  return res.json();
}

export async function fetchModelComparison(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/model-comparison`);
  if (!res.ok) throw new Error('Failed to fetch model comparison');
  return res.json();
}

export async function fetchMetricsInsights(): Promise<any> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics insights');
  return res.json();
}
