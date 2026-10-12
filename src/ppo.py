"""PPO objective and GAE with separate bootstrap and episode masks."""

import numpy as np
import torch


def advantages(rewards, values, next_values, terminated, ended, gamma, lam):
    out = np.zeros_like(rewards)
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - ended[t]) * carry
        out[t] = carry
    return out


def update(model, optimizer, batch, config):
    obs, latent, old_lp, adv, target = [
        torch.as_tensor(x, dtype=torch.float32) for x in batch
    ]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    stats = []
    for _ in range(config["epochs"]):
        for idx in torch.randperm(len(obs)).split(config["minibatch"]):
            logp = model.log_prob(obs[idx], latent[idx])
            ratio = (logp - old_lp[idx]).exp()
            unclipped = ratio * adv[idx]
            clipped = ratio.clamp(1 - config["clip"], 1 + config["clip"]) * adv[idx]
            policy = -torch.minimum(unclipped, clipped).mean()
            value = 0.5 * (model.value(obs[idx]) - target[idx]).square().mean()
            # Monte Carlo entropy of transformed policy; differentiable sample.
            u = model.distribution(obs[idx]).rsample()
            entropy = -model.log_prob(obs[idx], u).mean()
            loss = (
                policy + config["value_coef"] * value - config["entropy_coef"] * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad"])
            optimizer.step()
            with torch.no_grad():
                stats.append(
                    [
                        float(loss),
                        float(((ratio - 1) - (logp - old_lp[idx])).mean()),
                        float((abs(ratio - 1) > config["clip"]).float().mean()),
                    ]
                )
    return np.mean(stats, axis=0).tolist()
