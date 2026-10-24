import React from 'react';
import { BookOpen, ShieldAlert, Award, FileText, CheckCircle2 } from 'lucide-react';

export const Documentation: React.FC = () => {
  return (
    <div className="space-y-8 p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <BookOpen className="h-6 w-6 text-purple-400" />
          <span>Academic Project Defense Documentation</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Comprehensive project report for final year capstone defense (B.Tech Computer Science & Engineering, Group 76).
        </p>
      </div>

      {/* Document Sections */}
      <div className="glass-card p-8 rounded-2xl border border-slate-800 space-y-8 text-slate-300 text-xs sm:text-sm leading-relaxed">
        
        {/* Section 1: Abstract */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white uppercase tracking-wider font-mono text-blue-400">
            1. Executive Abstract
          </h2>
          <p>
            Large Language Models (LLMs) deployed in clinical diagnostic question answering frequently generate confident, plausible, yet medically hazardous factual hallucinations. While existing safeguard pipelines rely on post-hoc self-consistency sampling (e.g. SelfCheckGPT) or external knowledge retrieval critics, these retrospective techniques introduce unacceptable multi-second token latencies and high GPU compute costs in time-critical emergency healthcare. This project investigates Real-Time Hallucination Detection (RTHD) by directly monitoring intermediate transformer hidden states ($h_t \in \mathbb{'{R}'}^{'{4096}'}$) using lightweight 2-layer MLP probe classifiers. We evaluate four 4-bit NF4 quantized foundation models (Qwen3-8B, BioMistral-7B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct) across four medical QA benchmarks (EndoDiag, MedHallu, MedQA-USMLE, PubMedQA). Empirical results demonstrate a 100% early-detection rate across 52 error trajectories, providing a mean warning lead time of 14.85 to 32.50 tokens ahead of diagnostic hallucination emission and achieving 76.9% to 100% simulated error suppression while maintaining 84.8% to 95.6% response completeness.
          </p>
        </section>

        {/* Section 2: Motivation & Problem Statement */}
        <section className="space-y-3 pt-4 border-t border-slate-800">
          <h2 className="text-base font-bold text-white uppercase tracking-wider font-mono text-blue-400">
            2. Problem Statement & Motivation
          </h2>
          <p>
            Medical hallucination differs fundamentally from conversational generation errors due to the asymmetry of clinical risk: incorrect contraindication advice, erroneous staging guidelines, or fabricated pharmacology can result in irreversible patient morbidity. Current evaluation methods suffer from two acute vulnerabilities:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-300">
            <li>
              <strong>Post-Hoc Verification Latency:</strong> Methods that sample multiple generation paths (e.g. Semantic Entropy) or perform claim verification require waiting for complete sequence termination, delaying clinical workflows.
            </li>
            <li>
              <strong>Black-Box Token Insensitivity:</strong> Generation probabilities (softmax confidence) are frequently uncalibrated; models routinely hallucinate erroneous clinical claims with high token confidence.
            </li>
          </ul>
        </section>

        {/* Section 3: Architecture & 3-State Formulation */}
        <section className="space-y-3 pt-4 border-t border-slate-800">
          <h2 className="text-base font-bold text-white uppercase tracking-wider font-mono text-blue-400">
            3. Three-State Probing Formulation
          </h2>
          <p>
            Rather than a naive binary classification, the state sequence is parameterized into three distinct operational states:
          </p>
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 font-mono text-xs space-y-2">
            <div><strong className="text-emerald-400">Safe ($S_0$):</strong> Factually aligned tokens adhering to reference clinical evidence.</div>
            <div><strong className="text-amber-400">At-Risk ($S_1$):</strong> A sliding buffer window ($w=10$ tokens) immediately preceding a diagnostic failure, representing the latent epistemic uncertainty phase transition.</div>
            <div><strong className="text-rose-400">Hallucinating ($S_2$):</strong> The explicit manifestation of inaccurate medical assertions or contraindicated statements.</div>
          </div>
        </section>

        {/* Section 4: Clinical Disclosures & Governance */}
        <section className="space-y-3 pt-4 border-t border-slate-800">
          <h2 className="text-base font-bold text-white uppercase tracking-wider font-mono text-amber-400 flex items-center space-x-2">
            <ShieldAlert className="h-4 w-4" />
            <span>4. Clinical Limitations & Governance Disclosures</span>
          </h2>
          <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-900/40 text-slate-300 text-xs space-y-2">
            <p>
              1. <strong>Academic Pilot Scope:</strong> Experiments were executed on Kaggle GPU instances with $N=20$ trajectories per dataset. While demonstrating clear internal representation dynamics, broad generalization requires clinical-scale trials.
            </p>
            <p>
              2. <strong>Heuristic Error Labeling:</strong> Ground-truth token states are produced via automated ROUGE-L and keyword alignment heuristics without independent double-blind physician review.
            </p>
            <p>
              3. <strong>Simulated Intervention:</strong> Hallucination suppression metrics are measured via simulated trajectory truncation at critical trigger $t^*$, not active forward-pass logit steering.
            </p>
            <p>
              4. <strong>Non-Medical Device Disclaimer:</strong> This software is strictly for academic research and ML interpretability. It is not licensed for clinical diagnosis or treatment guidance.
            </p>
          </div>
        </section>

      </div>
    </div>
  );
};
