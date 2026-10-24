export interface Metadata {
  project_name: string;
  group_id: string;
  student_name: string;
  guide_name: string;
  department: string;
  description: string;
  total_experiments: number;
  total_datasets: number;
  models_evaluated: string[];
  datasets_evaluated: string[];
  probe_architecture: string;
  hardware_specification: string;
  github_url: string;
}

export interface DatasetSummary {
  id: string;
  name: string;
  domain: string;
  total_samples: number;
  file_size_bytes: number;
  description: string;
  sample_format: string;
}

export interface DatasetSample {
  id: string | number;
  question?: string;
  prompt?: string;
  answer?: string;
  options?: Record<string, string>;
  context?: string;
  source?: string;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  model: string;
  dataset: string;
  quantization: string;
  total_trajectories: number;
  probe_accuracy: number;
  f1_safe: number;
  f1_atrisk: number;
  f1_hallucinating: number;
  early_detection_rate: number;
  mean_lead_time_tokens: number;
  false_positive_rate?: number | null;
  hallucination_reduction: number;
  completeness: number;
  stop_threshold: number;
  min_tokens: number;
}

export interface SampleSummary {
  id: string;
  index: number;
  length: number;
  verdict: string;
  t_error?: number | null;
  t_star?: number | null;
  lead_time?: number | null;
  triggered: boolean;
  false_positive: boolean;
  max_risk: number;
  mean_risk: number;
  risk_at_error?: number | null;
  risk_at_t_star?: number | null;
}

export interface TokenState {
  index: number;
  token_id: number;
  token_text: string;
  state: 'Safe' | 'AtRisk' | 'Hallucinating';
  risk_score: number;
  max_prob: number;
  is_critical_trigger: boolean;
  is_error_onset: boolean;
}

export interface PCAPoint {
  x: number;
  y: number;
  token_index: number;
  token_text: string;
  state: string;
  risk: number;
}

export interface HiddenStateDetail {
  sample_id: string;
  experiment_id: string;
  model_name: string;
  dataset_name: string;
  tensor_shape: number[];
  extracted_layers: number[];
  hidden_dimension: number;
  sequence_length: number;
  summary_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
    l2_norm_mean: number;
  };
  tokens: TokenState[];
  heatmap_matrix: number[][];
  heatmap_feature_indices: number[];
  pca_projection: PCAPoint[];
  prompt: string;
  generated_text: string;
  reference_answer?: string | null;
  t_star?: number | null;
  t_error?: number | null;
  lead_time?: number | null;
}

export interface TokenVectorDetail {
  sample_id: string;
  token_index: number;
  token_text: string;
  state: string;
  risk_score: number;
  hidden_dimension: number;
  vector_summary: {
    mean: number;
    std: number;
    min: number;
    max: number;
    l2_norm: number;
  };
  top_positive_features: Array<{ dimension: number; activation: number }>;
  top_negative_features: Array<{ dimension: number; activation: number }>;
  layer_activation_norms: Record<string, number>;
  components_preview: number[];
}

export interface TrajectoryPoint {
  token_index: number;
  token_text: string;
  risk: number;
  smoothed_risk: number;
  state: string;
  is_threshold_exceeded: boolean;
}

export interface TrajectoryResponse {
  sample_id: string;
  experiment_id: string;
  model_name: string;
  threshold: number;
  t_star?: number | null;
  t_error?: number | null;
  lead_time?: number | null;
  total_tokens: number;
  points: TrajectoryPoint[];
  full_text: string;
  truncated_text: string;
  completeness_ratio: number;
  intervention_effective: boolean;
}

export interface DetectResponse {
  status: string;
  model_used: string;
  threshold: number;
  t_star?: number | null;
  risk_scores: number[];
  predicted_states: string[];
  triggered: boolean;
  summary_verdict: string;
  full_text: string;
  truncated_text: string;
  latency_ms: number;
}
