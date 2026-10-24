from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MetadataResponse(BaseModel):
    project_name: str
    group_id: str
    student_name: str
    guide_name: str
    department: str
    description: str
    total_experiments: int
    total_datasets: int
    models_evaluated: List[str]
    datasets_evaluated: List[str]
    probe_architecture: str
    hardware_specification: str
    github_url: str


class DatasetSummary(BaseModel):
    id: str
    name: str
    domain: str
    total_samples: int
    file_size_bytes: int
    description: str
    sample_format: str


class DatasetSample(BaseModel):
    id: Any
    question: Optional[str] = None
    prompt: Optional[str] = None
    answer: Optional[str] = None
    options: Optional[Dict[str, str]] = None
    context: Optional[str] = None
    source: Optional[str] = None


class ExperimentSummary(BaseModel):
    id: str
    name: str
    model: str
    dataset: str
    quantization: str
    total_trajectories: int
    probe_accuracy: float
    f1_safe: float
    f1_atrisk: float
    f1_hallucinating: float
    early_detection_rate: float
    mean_lead_time_tokens: float
    false_positive_rate: Optional[float] = None
    hallucination_reduction: float
    completeness: float
    stop_threshold: float
    min_tokens: int


class SampleSummary(BaseModel):
    id: str
    index: int
    length: int
    verdict: str
    t_error: Optional[int] = None
    t_star: Optional[int] = None
    lead_time: Optional[int] = None
    triggered: bool
    false_positive: bool
    max_risk: float
    mean_risk: float
    risk_at_error: Optional[float] = None
    risk_at_t_star: Optional[float] = None


class TokenState(BaseModel):
    index: int
    token_id: int
    token_text: str
    state: str  # Safe, AtRisk, Hallucinating
    risk_score: float
    max_prob: float
    is_critical_trigger: bool = False
    is_error_onset: bool = False


class PCAPoint(BaseModel):
    x: float
    y: float
    token_index: int
    token_text: str
    state: str
    risk: float


class HiddenStateDetail(BaseModel):
    sample_id: str
    experiment_id: str
    model_name: str
    dataset_name: str
    tensor_shape: List[int]
    extracted_layers: List[int]
    hidden_dimension: int
    sequence_length: int
    summary_stats: Dict[str, float]
    tokens: List[TokenState]
    heatmap_matrix: List[List[float]]
    heatmap_feature_indices: List[int]
    pca_projection: List[PCAPoint]
    prompt: str
    generated_text: str
    reference_answer: Optional[str] = None
    t_star: Optional[int] = None
    t_error: Optional[int] = None
    lead_time: Optional[int] = None


class TokenVectorDetail(BaseModel):
    sample_id: str
    token_index: int
    token_text: str
    state: str
    risk_score: float
    hidden_dimension: int
    vector_summary: Dict[str, float]
    top_positive_features: List[Dict[str, Any]]
    top_negative_features: List[Dict[str, Any]]
    layer_activation_norms: Dict[str, float]
    components_preview: List[float]


class TrajectoryPoint(BaseModel):
    token_index: int
    token_text: str
    risk: float
    smoothed_risk: float
    state: str
    is_threshold_exceeded: bool


class TrajectoryResponse(BaseModel):
    sample_id: str
    experiment_id: str
    model_name: str
    threshold: float
    t_star: Optional[int]
    t_error: Optional[int]
    lead_time: Optional[int]
    total_tokens: int
    points: List[TrajectoryPoint]
    full_text: str
    truncated_text: str
    completeness_ratio: float
    intervention_effective: bool


class DetectRequest(BaseModel):
    experiment_id: str = "endo_qwen_pilot20"
    prompt: str
    generated_text: str
    threshold: float = 0.85
    min_tokens: int = 10
    ma_window: int = 5


class DetectResponse(BaseModel):
    status: str
    model_used: str
    threshold: float
    t_star: Optional[int]
    risk_scores: List[float]
    predicted_states: List[str]
    triggered: bool
    summary_verdict: str
    full_text: str
    truncated_text: str
    latency_ms: float
