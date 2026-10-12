"""Small correctness checks for numerical and episode-boundary behavior."""

import numpy as np
import torch
from src.model import ActorCritic
from src.ppo import advantages

# Time-limit bootstrap is retained, but advantage recursion ends at reset.
a = advantages(
    np.array([1.0, 2.0]),
    np.zeros(2),
    np.array([5.0, 8.0]),
    np.array([False, True]),
    np.array([True, True]),
    0.9,
    0.95,
)
assert np.allclose(a, [5.5, 2.0])
model = ActorCritic()
obs = torch.randn(8, 10)
with torch.no_grad():
    action, u, old, _ = model.action(obs)
    assert torch.all(abs(action) <= 1)
    assert torch.allclose((model.log_prob(obs, u) - old).exp(), torch.ones(8))
    dist = torch.distributions.TransformedDistribution(
        model.distribution(obs), [torch.distributions.TanhTransform(cache_size=1)]
    )
    assert torch.allclose(
        model.log_prob(obs, u), dist.log_prob(action).sum(-1), atol=1e-4
    )
print(
    "Passed: bootstrap masks, reset boundaries, action bounds, probability ratios, transformed log probabilities."
)
