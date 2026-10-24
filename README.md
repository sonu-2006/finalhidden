# Real-Time Hallucination Detection and Intervention in Medical LLMs via Calibrated Hidden-State Probes

> **Note on Project Scope & Attribution:**
> This repository is an experimental evaluation and study based on the Real-Time Hallucination Detection (RTHD) framework. The core architecture and method were developed by the original RTHD authors. This repository documents our independent execution, verification, and analysis of four pilot experimental runs across medical LLMs and benchmark datasets using the implementation.

---

## Overview

This repository provides the implementation for real-time hallucination detection and intervention in medical LLMs using calibrated hidden-state probes. The framework monitors internal hidden representations across decoding timesteps using lightweight probe classifiers to identify when a model transitions from safe generation to an "at-risk" state, enabling early intervention before hallucinated claims are emitted.

```
                    ┌─────────────────────────┐
                    │      Medical Prompt     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Medical LLM Engine  │
                     └───────────┬───────────┘
                                 │ (token-by-token hidden states)
                                 ▼
               ┌───────────────────────────────────┐
               │    Calibrated Probe Classifier    │
               │   (Safe / AtRisk / Hallucinating) │
               └─────────────────┬─────────────────┘
                                 │
                     Risk Score Threshold (θ)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Risk <= θ: Continue]           [Risk > θ: Trigger Alert]
                                                 │
                                                 ▼
                                     [Simulated Intervention:
                                      Early Stop / Truncate]
```

---

## Experimental Evaluation — Four Pilot Runs

As part of this project evaluation, we executed and analyzed four pilot experiments on Kaggle (utilizing 4-bit quantized backbones) across distinct medical QA benchmarks and model architectures. Each pilot was evaluated with 20 trajectories to examine token-level probe accuracy, early detection capability (EDR@K), detection lead times, false positive rates, and simulated intervention effects.

### 1. Benchmark Datasets and Model Pairings

| Dataset | Medical Domain | Backbone Model | Quantization | Trajectories |
|---|---|---|---|---|
| **EndoDiag** | Endoscopy Diagnostic QA | Qwen3-8B | 4-bit NF4 | 20 |
| **MedHallu** | Medical Hallucination Benchmark | BioMistral-7B | 4-bit NF4 | 20 |
| **MedQA** | USMLE Medical Board Exams | Mistral-7B-Instruct-v0.3 | 4-bit NF4 | 20 |
| **PubMedQA** | Biomedical Research QA | Llama-3.1-8B-Instruct | 4-bit NF4 | 20 |

### 2. Verified Experimental Results

All reported metrics below have been verified directly from the saved pilot checkpoints, hidden states, and evaluation logs:

| Metric | EndoDiag<br>(Qwen3-8B) | MedHallu<br>(BioMistral-7B) | MedQA<br>(Mistral-7B) | PubMedQA<br>(Llama-3.1-8B) |
|---|---|---|---|---|
| **Total Trajectories** | 20 | 20 | 20 | 20 |
| **Labelled Error Cases** | 20 | 13 | 15 | 4 |
| **Non-Error Cases** | 0 | 7 | 5 | 16 |
| **Probe Accuracy** | **75.8%** (0.7578) | **76.4%** (0.7642) | **84.3%** (0.8431) | **90.6%** (0.9062) |
| **F1 Score (Safe)** | 0.882 | 0.855 | 0.914 | 0.952 |
| **F1 Score (At-Risk)** | 0.462 | 0.618 | 0.548 | 0.296 |
| **F1 Score (Hallucinating)** | 0.148 | 0.526 | 0.211 | 0.250 |
| **Early Detection Rate (EDR@K)** | **100%** (20/20) | **100%** (13/13) | **100%** (15/15) | **100%** (4/4) |
| **Mean Lead Time (tokens)** | **15.65** | **14.85** | **27.07** | **32.50** |
| **False Positive Rate (FPR)** | N/A* | **0.0%** (0/7) | **40.0%** (2/5) | **6.25%** (1/16) |
| **Simulated Hallucination Reduction** | **90.0%** | **76.9%** | **86.7%** | **100.0%** |
| **Simulated Completeness** | **87.4%** | **84.8%** | **91.4%** | **95.6%** |

*\*Note: In the EndoDiag pilot run, all 20 trajectories were classified with errors by the heuristic verifier, leaving no verified error-free cases to calculate a false positive rate.*

### 3. Experiment Artifacts Structure

The experiment outputs are organized under `results/`:

```
results/
├── endo_qwen_pilot20/               # EndoDiag + Qwen3-8B pilot run
│   ├── trajectories.jsonl           # 20 generated trajectories with prompts
│   ├── states.jsonl                 # Heuristically annotated token states
│   ├── probe.pt                     # Trained 2-layer MLP probe checkpoint
│   ├── critical_cases.jsonl         # Critical point detection per trajectory
│   ├── critical_cases_summary.csv   # Tabular summary of detection & lead times
│   ├── pilot_summary.json           # Aggregated metrics and configuration
│   └── hidden/                      # Extracted hidden states (.npz files)
├── medhallu_biomistral_pilot20/     # MedHallu + BioMistral-7B pilot run
│   ├── trajectories.jsonl
│   ├── states.jsonl
│   ├── probe.pt
│   ├── critical_cases.jsonl
│   ├── critical_cases_summary.csv
│   ├── pilot_summary.json
│   └── hidden/
├── medqa_mistral_pilot20/           # MedQA + Mistral-7B-Instruct-v0.3 pilot run
│   ├── trajectories.jsonl
│   ├── states.jsonl
│   ├── probe.pt
│   ├── critical_cases.jsonl
│   ├── critical_cases_summary.csv
│   ├── pilot_summary.json
│   └── hidden/
└── pubmedqa_llama_pilot20/          # PubMedQA + Llama-3.1-8B-Instruct pilot run
    ├── trajectories.jsonl
    ├── states.jsonl
    ├── probe.pt
    ├── critical_cases.jsonl
    ├── critical_cases_summary.csv
    ├── pilot_summary.json
    └── hidden/
```

### 4. How to Inspect & Reproduce the Pilot Evaluations

To evaluate the saved probes and intervention metrics on any of the pilot runs:

```bash
# 1. Evaluate Critical-Point Detection (Lead time, EDR, FPR)
python -m src.critical_detector \
    --labels results/endo_qwen_pilot20/states.jsonl \
    --model results/endo_qwen_pilot20/probe.pt \
    --theta 0.80 \
    --k 5

# 2. Evaluate Simulated Intervention (Hallucination reduction & completeness)
python -m src.intervention \
    --labels results/endo_qwen_pilot20/states.jsonl \
    --model results/endo_qwen_pilot20/probe.pt \
    --stop_threshold 0.85 \
    --min_tokens 10 \
    --ma_window 5
```

Replace `endo_qwen_pilot20` with `medhallu_biomistral_pilot20`, `medqa_mistral_pilot20`, or `pubmedqa_llama_pilot20` to inspect other benchmarks.

---

## Limitations & Disclosures

1. **Small Sample Size (Pilot Study):** Each pilot run evaluated exactly 20 trajectories. These results serve as a proof-of-concept for internal probe dynamics rather than a large-scale statistical validation.
2. **Heuristic State Labels:** Token-level states (`Safe`, `AtRisk`, `Hallucinating`) are generated using automated ROUGE-L / keyword verifiers with sliding window heuristics. They have not been independently validated by medical clinicians.
3. **Simulated Interventions:** The reported hallucination reduction and output completeness are calculated by simulating probe-triggered trajectory truncation (early stopping), not real-time guided decoding.
4. **Evaluation Dependency:** Detection sensitivity and lead times depend directly on the chosen risk threshold $\theta$, window parameters, and the heuristic label boundary.
5. **Not a Clinical Device:** These pilot runs are strictly for academic and experimental evaluation. They do **not** demonstrate real-world clinical safety, clinical utility, or readiness for deployment in healthcare settings.

---

## Repository Structure

```
.
├── src/                         # Core library modules
│   ├── model_loader.py          # LLM loading (Qwen, Llama, BioMistral)
│   ├── hidden_utils.py          # Hidden state I/O and history pooling
│   ├── text_utils.py            # ROUGE-L, Jaccard, sentence splitting
│   ├── trajectory_collector.py  # Trajectory generation + hidden state extraction
│   ├── claim_verifier.py        # Answer verification (ROUGE-L, Jaccard)
│   ├── state_annotator.py       # Safe/AtRisk/Hallucinating label assignment
│   ├── state_classifier.py      # Probe MLP training
│   ├── critical_detector.py     # Rule-gated detection + operating point selection
│   └── intervention.py          # Intervention policy evaluation
├── scripts/                     # End-to-end experiment scripts
│   ├── run_pipeline.py          # Full pipeline (PubMedQA, MedHallu)
│   ├── run_medqa_pipeline.py    # MedQA-USMLE pipeline
│   ├── run_ablations.py         # Binary/ternary + history pooling ablations
│   ├── run_selfcheck_nli.py     # SelfCheckGPT-NLI baseline
│   ├── run_semantic_entropy.py  # Semantic Entropy baseline
│   ├── run_cusum.py             # CUSUM trigger baseline
│   ├── run_bootstrap.py         # Bootstrap confidence intervals
│   └── run_tau_sweep.py         # ROUGE-L threshold sensitivity
├── data/                        # Benchmark Datasets (JSONL format)
│   ├── endo_diag.jsonl          # Endoscopy diagnostic QA (163 prompts)
│   ├── pubmedqa.jsonl           # PubMedQA (1,000 prompts)
│   ├── medhallu.jsonl           # MedHallu (500 prompts)
│   └── medqa.jsonl              # MedQA-USMLE (1,273 prompts)
├── data_processing/             # Data conversion utilities
│   ├── convert_pubmedqa.py
│   ├── convert_medhallu.py
│   └── prepare_dataset.py
├── results/                     # Experimental outputs from the 4 pilot runs
│   ├── endo_qwen_pilot20/
│   ├── medhallu_biomistral_pilot20/
│   ├── medqa_mistral_pilot20/
│   └── pubmedqa_llama_pilot20/
├── visualization/               # Figure generation scripts
│   ├── plot_trajectories.py     # Risk trajectory visualizations
│   └── plot_figures.py          # Paper figures (FPR-EDR curves, etc.)
├── requirements.txt             # Project dependencies
├── LICENSE                      # MIT License
└── README.md                    # Documentation & evaluation report
```

---

## Requirements & Setup

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.40
- scikit-learn
- numpy
- tqdm
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Citation & Original Attribution

```bibtex
@article{anonymous2026hallucination,
  title={Real-Time Hallucination Detection and Intervention in Medical LLMs via Calibrated Hidden-State Probes},
  author={Anonymous},
  year={2026}
}
```
