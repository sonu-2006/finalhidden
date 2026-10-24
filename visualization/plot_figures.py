import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("/home/yzq/paper/Probe_HalluDynamics_B/results/b_run")

# EDR curve
ks = []
edr = []
lead = []
with (base / "fig_edr_curve.csv").open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        ks.append(int(row["K"]))
        edr.append(float(row["EDR@K"]))
        lead.append(float(row["LeadTime"]))

plt.figure(figsize=(4,3))
plt.plot(ks, edr, marker="o")
plt.xlabel("K")
plt.ylabel("EDR@K")
plt.title("EDR@K (ours)")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base / "fig_edr_curve.png", dpi=200)
plt.close()

# Trade-off curve
ths = []
red = []
comp = []
with (base / "fig_tradeoff.csv").open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        ths.append(float(row["stop_threshold"]))
        red.append(float(row["hallucination_reduction"]))
        comp.append(float(row["completeness"]))

plt.figure(figsize=(4,3))
plt.plot(comp, red, marker="o")
for i, th in enumerate(ths):
    plt.annotate(f"{th:.2f}", (comp[i], red[i]), fontsize=7, xytext=(3,3), textcoords="offset points")
plt.xlabel("Completeness")
plt.ylabel("Hallucination Reduction")
plt.title("Intervention Trade-off")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base / "fig_tradeoff.png", dpi=200)
plt.close()

# Theta sweep
ths = []
edr = []
lead = []
with (base / "fig_theta_sweep.csv").open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        ths.append(float(row["theta"]))
        edr.append(float(row["EDR@20"]))
        lead.append(float(row["LeadTime"]))

plt.figure(figsize=(4,3))
plt.plot(ths, edr, marker="o")
plt.xlabel("theta")
plt.ylabel("EDR@20")
plt.title("Theta Sensitivity")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base / "fig_theta_sweep.png", dpi=200)
plt.close()

print("wrote figures to", base)
