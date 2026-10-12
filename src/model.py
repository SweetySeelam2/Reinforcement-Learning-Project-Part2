"""Actor, critic, and training-only observation statistics."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class RunningNorm:
    def __init__(self, size):
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)
        x = x.reshape(-1, len(self.mean))
        n = len(x)
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.var = (
            self.var * self.count + x.var(0) * n + delta**2 * self.count * n / total
        ) / total
        self.mean += delta * n / total
        self.count = total

    def apply(self, x):
        return np.clip(
            (np.asarray(x) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10
        ).astype(np.float32)

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class ActorCritic(nn.Module):
    def __init__(self, observations=10, actions=2):
        super().__init__()

        def network(out):
            return nn.Sequential(
                nn.Linear(observations, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, out),
            )

        self.actor = network(actions)
        self.critic = network(1)
        self.log_std = nn.Parameter(torch.zeros(actions))
        for net in [self.actor, self.critic]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-5, 2).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def log_prob(self, obs, latent):
        # Stable log(1 - tanh(u)^2), with latent u retained in each rollout.
        correction = 2 * (
            np.log(2) - latent - torch.nn.functional.softplus(-2 * latent)
        )
        return (self.distribution(obs).log_prob(latent) - correction).sum(-1)

    def action(self, obs, deterministic=False):
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.sample()
        return torch.tanh(latent), latent, self.log_prob(obs, latent), self.value(obs)
