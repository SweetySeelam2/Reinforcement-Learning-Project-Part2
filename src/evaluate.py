"""Fixed-seed evaluation with frozen observation normalization."""

import numpy as np
import torch
import gymnasium as gym


def evaluate(model, norm, seeds, policy="ppo", records=False):
    env = gym.make("Reacher-v5")
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 500000)
        total = effort = 0.0
        distances = []
        for _ in range(50):
            if policy == "random":
                action = rng.uniform(-1, 1, 2).astype(np.float32)
            elif policy == "zero":
                action = np.zeros(2, dtype=np.float32)
            else:
                with torch.no_grad():
                    action = model.action(
                        torch.from_numpy(norm.apply(obs)).unsqueeze(0), True
                    )[0][0].numpy()
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            effort += float(np.square(action).sum())
            distances.append(float(np.linalg.norm(obs[-2:])))
            if term or trunc:
                break
        rows.append(
            dict(
                episode_seed=int(seed),
                return_=float(total),
                distance=float(np.mean(distances[-10:])),
                effort=effort,
                success=int(all(d <= 0.03 for d in distances[-10:])),
            )
        )
    env.close()
    summary = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ["return_", "distance", "effort", "success"]
    }
    return (summary, rows) if records else summary
