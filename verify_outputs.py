import json
from pathlib import Path
import numpy as np
from src.model import ActorCritic, RunningNorm
from src.evaluate import evaluate
import torch

root = Path("outputs")
results = json.loads((root / "results.json").read_text())
assert len(results) == 6
for r in results:
    folder = root / f'lambda_{r["lam"]}_seed_{r["seed"]}'
    rows = json.loads((folder / "test.json").read_text())["episodes"]
    assert len(rows) == 100 and r["steps"] == 200704
    for k in ["return_", "distance", "effort", "success"]:
        assert np.isclose(r[k], np.mean([e[k] for e in rows]))
    logs = json.loads((folder / "validation.json").read_text())
    assert len(logs) == 11 and logs[0]["steps"] == 0 and logs[-1]["steps"] == 200704
# Confirm checkpoint reload reproduces deterministic held-out episode metrics.
folder = root / "lambda_0.95_seed_0"
c = torch.load(folder / "final.pt", weights_only=False, map_location="cpu")
m = ActorCritic()
m.load_state_dict(c["model"])
n = RunningNorm(10)
n.load(c["norm"])
s, rows = evaluate(m, n, [20000], records=True)
saved = json.loads((folder / "test.json").read_text())["episodes"][0]
assert all(
    np.isclose(rows[0][k], saved[k])
    for k in ["return_", "distance", "effort", "success"]
)
print(
    "Six runs verified; 600 held-out episodes; complete checkpoints and logs; reloaded episode reproduced."
)
