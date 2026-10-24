import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.common.hidden_utils import load_hidden_from_npz
from src.direction_b.state_classifier import MLP, STATE2ID


def moving_average(x, w):
    if w <= 1:
        return x
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    csum = np.cumsum(x)
    for i in range(len(x)):
        j = i - w
        if j >= 0:
            out[i] = (csum[i] - csum[j]) / w
        else:
            out[i] = csum[i] / (i + 1)
    return out


def pick_t_star(risk, theta, persistence=1, min_tokens=0):
    if len(risk) == 0:
        return None
    cnt = 0
    start_t = max(0, min_tokens - 1)
    for t in range(start_t, len(risk)):
        if risk[t] > theta:
            cnt += 1
            if cnt >= persistence:
                return t - persistence + 1
        else:
            cnt = 0
    return None


def compute_metrics(cases, k=5):
    with_error = 0
    detected = 0
    lead_times = []
    fp = 0
    no_error = 0
    for c in cases:
        t_error = c['t_error']
        t_star = c['t_star']
        if t_error is None:
            no_error += 1
            if t_star is not None:
                fp += 1
        else:
            with_error += 1
            if t_star is not None and t_star < t_error + k:
                detected += 1
                lead_times.append(t_error - t_star)
    return {
        'edr_at_k': detected / with_error if with_error else 0.0,
        'lead_time': float(np.mean(lead_times)) if lead_times else 0.0,
        'fpr': fp / no_error if no_error else 0.0,
        'with_error': with_error,
        'no_error': no_error,
    }


def bootstrap_ci(cases, k=5, B=1000, seed=42):
    random.seed(seed)
    edr, lead, fpr = [], [], []
    n = len(cases)
    for _ in range(B):
        sample = [random.choice(cases) for _ in range(n)]
        m = compute_metrics(sample, k=k)
        edr.append(m['edr_at_k'])
        lead.append(m['lead_time'])
        fpr.append(m['fpr'])

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        'B': B,
        'edr_at_k_ci95': ci(edr),
        'lead_time_ci95': ci(lead),
        'fpr_ci95': ci(fpr),
    }


def length_bins(cases, k=5, q=4):
    lengths = [c['length'] for c in cases]
    qs = np.quantile(lengths, np.linspace(0.0, 1.0, q + 1)).tolist()
    bins = []
    for i in range(q):
        lo = int(qs[i])
        hi = int(qs[i + 1])
        subset = [c for c in cases if lo <= c['length'] <= hi]
        m = compute_metrics(subset, k=k)
        bins.append({'lo': lo, 'hi': hi, 'n': len(subset), **m})
    return {'quantiles': qs, 'bins': bins}


def export_cases(base, cfg, risks_pack):
    theta = cfg['theta']
    persistence = cfg['persistence']
    min_tokens = cfg['min_tokens']
    ma_window = cfg['ma_window']
    tag = cfg['tag']

    cases = []
    for item in risks_pack:
        risk = moving_average(item['risk'], ma_window)
        t_star = pick_t_star(risk, theta=theta, persistence=persistence, min_tokens=min_tokens)
        t_error = item['t_error']
        lead_time = None
        if t_error is not None and t_star is not None:
            lead_time = int(t_error - t_star)
        cases.append({
            'id': item['id'],
            'length': item['length'],
            'verdict': item['verdict'],
            't_error': t_error,
            't_star': t_star,
            'lead_time': lead_time,
            'triggered': bool(t_error is not None and t_star is not None and t_star < t_error + 5),
            'false_positive': bool(t_error is None and t_star is not None),
            'max_risk': float(np.max(risk)) if len(risk) else None,
            'mean_risk': float(np.mean(risk)) if len(risk) else None,
            'risk_at_error': float(risk[t_error]) if t_error is not None and t_error < len(risk) else None,
            'risk_at_t_star': float(risk[t_star]) if t_star is not None and t_star < len(risk) else None,
            'theta': theta,
            'persistence': persistence,
            'ma_window': ma_window,
            'min_tokens': min_tokens,
            'k': 5,
        })

    out_jsonl = base / f'cases_{tag}.jsonl'
    with out_jsonl.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    out_csv = base / f'cases_{tag}.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader()
        w.writerows(cases)

    m = compute_metrics(cases, k=5)
    b = bootstrap_ci(cases, k=5, B=1000)
    lb = length_bins(cases, k=5, q=4)

    out_meta = base / f'cases_{tag}_stats.json'
    out_meta.write_text(json.dumps({
        'config': cfg,
        'point': m,
        'bootstrap': b,
        'length_bins': lb,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print('exported', tag, m)
    return out_jsonl, out_csv, out_meta


def main():
    base = Path('/home/yzq/paper/Probe_HalluDynamics_B/results/b_suite_v6_1kgen_qwen_rouge012_fast')
    labels_path = base / 'states_window10.jsonl'
    model_path = base / 'clf_window10_layer2_h8.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(hidden_dim=4096).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    risks_pack = []
    with labels_path.open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            hidden = load_hidden_from_npz(item['hidden_path'], layer_idx=2, layer_pool='mean', history=8)
            if len(hidden) == 0:
                continue
            X = torch.tensor(hidden, dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = torch.softmax(model(X), dim=-1).cpu().numpy()
            risk = 1.0 - probs[:, STATE2ID['Safe']]

            t_error = None
            for t, s in enumerate(item['states']):
                if s == 'Hallucinating':
                    t_error = t
                    break

            risks_pack.append({
                'id': item.get('id'),
                'length': len(item['states']),
                'verdict': item.get('verdict'),
                't_error': t_error,
                'risk': risk.astype(np.float32),
            })

    configs = [
        {'tag': 'rule_cons_fpr005_theta099_p5_ma5', 'theta': 0.99, 'persistence': 5, 'min_tokens': 0, 'ma_window': 5},
        {'tag': 'rule_mid_fpr02_theta096_p5_ma5', 'theta': 0.96, 'persistence': 5, 'min_tokens': 0, 'ma_window': 5},
        {'tag': 'rule_highrecall_theta091_p5_ma5', 'theta': 0.91, 'persistence': 5, 'min_tokens': 0, 'ma_window': 5},
    ]

    summary = []
    for cfg in configs:
        j, c, s = export_cases(base, cfg, risks_pack)
        summary.append({'config': cfg, 'jsonl': str(j), 'csv': str(c), 'stats': str(s)})

    out = base / 'cases_rule_summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
