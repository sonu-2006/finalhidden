import os
import json
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DATASET_METADATA = {
    "endo_diag": {
        "name": "EndoDiag (Endoscopy Diagnostic QA)",
        "domain": "Gastroenterology & Endoscopic Diagnosis",
        "description": "Clinical endoscopy diagnostic questions paired with ground truth findings and expert diagnostic classifications.",
        "sample_format": "JSONL records with image/patient IDs, diagnostic prompt, and ground truth reference.",
    },
    "medhallu": {
        "name": "MedHallu (Medical Hallucination Benchmark)",
        "domain": "General Clinical & Biomedical Assertions",
        "description": "Standardized medical benchmark designed specifically for factual consistency and hallucination detection in clinical reasoning.",
        "sample_format": "JSONL records with medical premise, clinical query, and ground truth claim.",
    },
    "medqa": {
        "name": "MedQA (USMLE Medical Board Exam)",
        "domain": "Multi-Specialty Clinical Medicine",
        "description": "Four-option multiple-choice clinical board examination questions evaluating high-level medical diagnostic and pharmacological knowledge.",
        "sample_format": "JSONL records with clinical vignette, multiple-choice options (A-D), and correct answer letter.",
    },
    "pubmedqa": {
        "name": "PubMedQA (Biomedical Research QA)",
        "domain": "Biomedical Literature & Evidence-Based Medicine",
        "description": "Biomedical research questions derived from PubMed abstracts with expert yes/no/maybe conclusions and reasoning contexts.",
        "sample_format": "JSONL records with abstract context, research query, and gold-standard conclusion.",
    },
}

EXPERIMENT_METADATA = {
    "endo_qwen_pilot20": {
        "name": "EndoDiag + Qwen3-8B",
        "model": "Qwen3-8B (Alibaba Cloud)",
        "dataset": "EndoDiag",
        "quantization": "4-bit NF4 (bitsandbytes)",
    },
    "medhallu_biomistral_pilot20": {
        "name": "MedHallu + BioMistral-7B",
        "model": "BioMistral-7B (BioMedical Domain-Adapted)",
        "dataset": "MedHallu",
        "quantization": "4-bit NF4 (bitsandbytes)",
    },
    "medqa_mistral_pilot20": {
        "name": "MedQA + Mistral-7B-Instruct-v0.3",
        "model": "Mistral-7B-Instruct-v0.3",
        "dataset": "MedQA-USMLE",
        "quantization": "4-bit NF4 (bitsandbytes)",
    },
    "pubmedqa_llama_pilot20": {
        "name": "PubMedQA + Llama-3.1-8B-Instruct",
        "model": "Llama-3.1-8B-Instruct (Meta AI)",
        "dataset": "PubMedQA",
        "quantization": "4-bit NF4 (bitsandbytes)",
    },
}


class DataService:
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "project_name": "Medical Hallucination Detection in Large Language Models Using Hidden-State Probing",
            "group_id": "G076",
            "student_name": "Gnyana Samhitha",
            "guide_name": "Dr. Swapna T R",
            "department": "Department of Computer Science & Engineering",
            "description": "Real-time token-level monitoring of internal LLM activations (layers 4, 20, 32) via calibrated probes to detect and pre-emptively truncate clinical hallucinations 14-32 tokens before emission.",
            "total_experiments": len(EXPERIMENT_METADATA),
            "total_datasets": len(DATASET_METADATA),
            "models_evaluated": [
                "Qwen3-8B (4-bit)",
                "BioMistral-7B (4-bit)",
                "Mistral-7B-Instruct-v0.3 (4-bit)",
                "Llama-3.1-8B-Instruct (4-bit)",
            ],
            "datasets_evaluated": ["EndoDiag", "MedHallu", "MedQA-USMLE", "PubMedQA"],
            "probe_architecture": "2-Layer Multilayer Perceptron (MLP): Linear(4096, 64) -> ReLU -> Linear(64, 3)",
            "hardware_specification": "Evaluated on Kaggle GPU Environments (NVIDIA A800 / T4, 4-bit NF4 Quantization)",
            "github_url": "https://github.com/sonu-2006/finalhidden",
        }

    @staticmethod
    def get_dataset_summaries() -> List[Dict[str, Any]]:
        results = []
        for dataset_id, meta in DATASET_METADATA.items():
            path = os.path.join(DATA_DIR, f"{dataset_id}.jsonl")
            total_samples = 0
            size_bytes = 0
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    total_samples = sum(1 for _ in f)

            results.append({
                "id": dataset_id,
                "name": meta["name"],
                "domain": meta["domain"],
                "total_samples": total_samples,
                "file_size_bytes": size_bytes,
                "description": meta["description"],
                "sample_format": meta["sample_format"],
            })
        return results

    @staticmethod
    def get_dataset_samples(dataset_id: str, offset: int = 0, limit: int = 20, search: Optional[str] = None) -> Dict[str, Any]:
        path = os.path.join(DATA_DIR, f"{dataset_id}.jsonl")
        if not os.path.exists(path):
            return {"total": 0, "samples": []}

        all_matching = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text_blob = json.dumps(record).lower()
                    if search and search.lower() not in text_blob:
                        continue

                    # normalize fields
                    sample = {
                        "id": record.get("id", record.get("question_id", idx)),
                        "question": record.get("question", record.get("prompt", "")),
                        "prompt": record.get("prompt", record.get("question", "")),
                        "answer": record.get("answer", record.get("ideal_answer", record.get("final_decision", ""))),
                        "options": record.get("options", None),
                        "context": record.get("context", record.get("abstract", None)),
                        "source": record.get("source", dataset_id),
                    }
                    all_matching.append(sample)
                except Exception:
                    continue

        total = len(all_matching)
        sliced = all_matching[offset : offset + limit]
        return {"total": total, "samples": sliced}

    @staticmethod
    def get_experiment_summaries() -> List[Dict[str, Any]]:
        results = []
        for exp_id, meta in EXPERIMENT_METADATA.items():
            exp_dir = os.path.join(RESULTS_DIR, exp_id)
            sum_path = os.path.join(exp_dir, "pilot_summary.json")

            if os.path.exists(sum_path):
                with open(sum_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                probe_m = data.get("probe_metrics", {})
                results.append({
                    "id": exp_id,
                    "name": meta["name"],
                    "model": meta["model"],
                    "dataset": meta["dataset"],
                    "quantization": meta["quantization"],
                    "total_trajectories": data.get("number_of_trajectories", 20),
                    "probe_accuracy": float(probe_m.get("accuracy", 0.0)),
                    "f1_safe": float(probe_m.get("f1_safe", 0.0)),
                    "f1_atrisk": float(probe_m.get("f1_atrisk", 0.0)),
                    "f1_hallucinating": float(probe_m.get("f1_hallucinating", 0.0)),
                    "early_detection_rate": float(data.get("early_detection_proportion", 1.0)),
                    "mean_lead_time_tokens": float(data.get("mean_lead_time_tokens", 0.0)),
                    "false_positive_rate": data.get("false_positive_rate", None),
                    "hallucination_reduction": float(data.get("intervention_hallucination_reduction", 0.0)),
                    "completeness": float(data.get("intervention_completeness", 0.0)),
                    "stop_threshold": float(data.get("intervention_stop_threshold", 0.85)),
                    "min_tokens": int(data.get("intervention_min_tokens", 10)),
                })
        return results

    @staticmethod
    def get_experiment_samples(experiment_id: str) -> List[Dict[str, Any]]:
        exp_dir = os.path.join(RESULTS_DIR, experiment_id)
        cc_path = os.path.join(exp_dir, "critical_cases.jsonl")
        if not os.path.exists(cc_path):
            return []

        samples = []
        with open(cc_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                samples.append({
                    "id": str(c.get("id")),
                    "index": idx,
                    "length": int(c.get("length", 64)),
                    "verdict": str(c.get("verdict", "unknown")),
                    "t_error": c.get("t_error"),
                    "t_star": c.get("t_star"),
                    "lead_time": c.get("lead_time"),
                    "triggered": bool(c.get("triggered", False)),
                    "false_positive": bool(c.get("false_positive", False)),
                    "max_risk": float(c.get("max_risk", 0.0)),
                    "mean_risk": float(c.get("mean_risk", 0.0)),
                    "risk_at_error": c.get("risk_at_error"),
                    "risk_at_t_star": c.get("risk_at_t_star"),
                })
        return samples

    @staticmethod
    def get_sample_records(experiment_id: str, sample_id: str) -> Optional[Dict[str, Any]]:
        exp_dir = os.path.join(RESULTS_DIR, experiment_id)
        
        # 1. Trajectory record
        traj_path = os.path.join(exp_dir, "trajectories.jsonl")
        target_traj = None
        if os.path.exists(traj_path):
            with open(traj_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if str(item.get("id")) == str(sample_id):
                        target_traj = item
                        break

        # 2. State record
        states_path = os.path.join(exp_dir, "states.jsonl")
        target_state = None
        if os.path.exists(states_path):
            with open(states_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if str(item.get("id")) == str(sample_id):
                        target_state = item
                        break

        # 3. Critical case record
        cc_path = os.path.join(exp_dir, "critical_cases.jsonl")
        target_cc = None
        if os.path.exists(cc_path):
            with open(cc_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if str(item.get("id")) == str(sample_id):
                        target_cc = item
                        break

        if not target_traj and not target_state and not target_cc:
            return None

        return {
            "trajectory": target_traj or {},
            "state": target_state or {},
            "critical_case": target_cc or {},
        }
