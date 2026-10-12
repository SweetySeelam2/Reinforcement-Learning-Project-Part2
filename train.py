"""Run matched PPO and one-step-advantage experiments from a JSON config."""

import argparse, json, time, random, platform
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from src.model import ActorCritic, RunningNorm
from src.ppo import advantages, update
from src.evaluate import evaluate


def run(config, seed, lam, root):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("Reacher-v5")
    obs, _ = env.reset(seed=seed)
    model = ActorCritic()
    norm = RunningNorm(10)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-5)
    folder = Path(root) / f"lambda_{lam}_seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    log = [dict(steps=0, seconds=0, **evaluate(model, norm, range(10000, 10020)))]
    start = time.perf_counter()
    steps = 0
    next_eval = 20000
    for _ in range(int(np.ceil(config["steps"] / config["rollout"]))):
        # Freeze preprocessing within a rollout and its PPO update.
        states = []
        latents = []
        lps = []
        values = []
        rewards = []
        nextvs = []
        terms = []
        ends = []
        raw = []
        for t in range(config["rollout"]):
            raw.append(obs.copy())
            state = norm.apply(obs)
            with torch.no_grad():
                action, latent, lp, v = model.action(
                    torch.from_numpy(state).unsqueeze(0)
                )
            nxt, reward, term, trunc, _ = env.step(action[0].numpy())
            with torch.no_grad():
                nv = model.value(torch.from_numpy(norm.apply(nxt)).unsqueeze(0)).item()
            states.append(state)
            latents.append(latent[0].numpy())
            lps.append(lp.item())
            values.append(v.item())
            rewards.append(reward)
            nextvs.append(nv)
            terms.append(term)
            ends.append(term or trunc)
            obs = env.reset()[0] if term or trunc else nxt
        a = advantages(
            np.array(rewards),
            np.array(values),
            np.array(nextvs),
            np.array(terms),
            np.array(ends),
            config["gamma"],
            lam,
        )
        loss, kl, clipfrac = update(
            model,
            optimizer,
            [
                np.array(states),
                np.array(latents),
                np.array(lps),
                a,
                a + np.array(values),
            ],
            config,
        )
        norm.update(np.array(raw))
        steps += config["rollout"]
        if steps >= next_eval or steps >= config["steps"]:
            item = dict(
                steps=steps,
                seconds=time.perf_counter() - start,
                loss=loss,
                approx_kl=kl,
                clip_fraction=clipfrac,
                **evaluate(model, norm, range(10000, 10020)),
            )
            log.append(item)
            print(
                f'lambda={lam} seed={seed} steps={steps} return={item["return_"]:.2f} success={item["success"]:.0%}',
                flush=True,
            )
            next_eval += 20000
            (folder / "validation.json").write_text(json.dumps(log, indent=2))
    train_seconds = time.perf_counter() - start
    torch.save(
        dict(
            model=model.state_dict(),
            norm=norm.state(),
            config=config,
            seed=seed,
            lam=lam,
        ),
        folder / "final.pt",
    )
    summary, episodes = evaluate(model, norm, range(20000, 20100), records=True)
    result = dict(
        seed=seed, lam=lam, steps=steps, training_seconds=train_seconds, **summary
    )
    (folder / "test.json").write_text(
        json.dumps(dict(summary=result, episodes=episodes), indent=2)
    )
    env.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    versions = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        gymnasium=gym.__version__,
    )
    import mujoco

    versions["mujoco"] = mujoco.__version__
    (Path(args.output) / "versions.json").write_text(json.dumps(versions, indent=2))
    if args.pilot:
        cfg["steps"] = 10240
        cfg["seeds"] = [0]
        cfg["lambdas"] = [0.95]
    results = []
    for lam in cfg["lambdas"]:
        for seed in cfg["seeds"]:
            results.append(run(cfg, seed, lam, args.output))
    for name in ["random", "zero"]:
        summary, rows = evaluate(
            None, None, range(20000, 20100), policy=name, records=True
        )
        (Path(args.output) / f"{name}_test.json").write_text(
            json.dumps(dict(summary=summary, episodes=rows), indent=2)
        )
    (Path(args.output) / "results.json").write_text(json.dumps(results, indent=2))
