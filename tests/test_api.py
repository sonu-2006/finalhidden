import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_metadata():
    res = client.get("/api/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "Medical Hallucination Detection" in data["project_name"]
    assert data["group_id"] == "G076"
    assert data["student_name"] == "Gnyana Samhitha"
    assert data["guide_name"] == "Dr. Swapna T R"


def test_datasets():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) == 4
    dataset_ids = [d["id"] for d in datasets]
    assert "endo_diag" in dataset_ids
    assert "medhallu" in dataset_ids
    assert "medqa" in dataset_ids
    assert "pubmedqa" in dataset_ids


def test_experiments():
    res = client.get("/api/experiments")
    assert res.status_code == 200
    exps = res.json()
    assert len(exps) == 4
    exp_ids = [e["id"] for e in exps]
    assert "endo_qwen_pilot20" in exp_ids


def test_experiment_samples():
    res = client.get("/api/experiments/endo_qwen_pilot20/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) == 20
    first = samples[0]
    assert "id" in first
    assert "verdict" in first


def test_hidden_states():
    # Fetch sample ID
    samples_res = client.get("/api/experiments/endo_qwen_pilot20/samples")
    sample_id = samples_res.json()[0]["id"]

    res = client.get(f"/api/hidden-states/endo_qwen_pilot20/{sample_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["tensor_shape"] == [1, 64, 4096]
    assert len(data["tokens"]) == 64
    assert len(data["heatmap_matrix"]) == 64
    assert len(data["pca_projection"]) == 64


def test_token_vector():
    samples_res = client.get("/api/experiments/endo_qwen_pilot20/samples")
    sample_id = samples_res.json()[0]["id"]

    res = client.get(f"/api/hidden-states/endo_qwen_pilot20/{sample_id}/token/52")
    assert res.status_code == 200
    data = res.json()
    assert data["hidden_dimension"] == 4096
    assert "l2_norm" in data["vector_summary"]
    assert len(data["top_positive_features"]) > 0
    assert len(data["components_preview"]) == 64


def test_trajectory():
    samples_res = client.get("/api/experiments/endo_qwen_pilot20/samples")
    sample_id = samples_res.json()[0]["id"]

    res = client.get(f"/api/trajectory/endo_qwen_pilot20/{sample_id}")
    assert res.status_code == 200
    data = res.json()
    assert "points" in data
    assert len(data["points"]) == 64
    assert "truncated_text" in data


def test_detect():
    payload = {
        "experiment_id": "endo_qwen_pilot20",
        "prompt": "Test medical prompt",
        "generated_text": "The patient shows signet ring cell adenocarcinoma with high risk of metastasis.",
        "threshold": 0.85,
    }
    res = client.post("/api/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "risk_scores" in data


def test_model_comparison():
    res = client.get("/api/model-comparison")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4
