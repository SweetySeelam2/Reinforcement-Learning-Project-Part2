"""Summarize real evaluation files and export plots and slide data."""

import json, csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("outputs")
results = json.loads((root / "results.json").read_text())
analysis = root / "analysis"
analysis.mkdir(exist_ok=True)
summary = {}
curves = {}
for lam in [0.95, 0.0]:
    runs = [r for r in results if r["lam"] == lam]
    key = f"GAE {lam}"
    summary[key] = {
        metric: dict(
            mean=float(np.mean([r[metric] for r in runs])),
            sd=float(np.std([r[metric] for r in runs], ddof=1)),
        )
        for metric in ["return_", "distance", "effort", "success", "training_seconds"]
    }
    logs = [
        json.loads(
            (root / f'lambda_{lam}_seed_{r["seed"]}' / "validation.json").read_text()
        )
        for r in runs
    ]
    curves[key] = dict(
        steps=[r["steps"] for r in logs[0]],
        returns=[[r["return_"] for r in log] for log in logs],
        successes=[[r["success"] for r in log] for log in logs],
    )
    for r, log in zip(runs, logs):
        r["steps_to_70_percent"] = next(
            (x["steps"] for x in log if x["success"] >= 0.7), None
        )
for name in ["random", "zero"]:
    summary[name] = json.loads((root / f"{name}_test.json").read_text())["summary"]
with (analysis / "run_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
(analysis / "summary.json").write_text(json.dumps(summary, indent=2))
(analysis / "slide_data.json").write_text(
    json.dumps(dict(summary=summary, curves=curves, runs=results), indent=2)
)
plt.rcParams.update(
    {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
)
for metric, field, label in [
    ("reward", "returns", "Validation episode return"),
    ("success", "successes", "Validation success rate"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color in [("GAE 0.95", "#087e8b"), ("GAE 0.0", "#b86b20")]:
        vals = np.array(curves[key][field])
        x = np.array(curves[key]["steps"]) / 1000
        for row in vals:
            ax.plot(x, row, color=color, alpha=0.22, lw=1)
        ax.plot(x, vals.mean(0), color=color, lw=2.5, label=key + " mean, 3 seeds")
    if metric == "reward":
        for name, style in [("random", ":"), ("zero", "--")]:
            ax.axhline(
                summary[name]["return_"],
                color="#666666",
                ls=style,
                label=name + " test reference",
            )
    ax.set(xlabel="Training transitions (thousands)", ylabel=label)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(analysis / f"{metric}_curve.png", dpi=160)
    plt.close(fig)
print(json.dumps(summary, indent=2))
