"""Render fixed examples chosen by held-out success and distance."""

import argparse, json, os

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from src.model import ActorCritic, RunningNorm


def main(root):
    root = Path(root)
    # Preselected training seed 0, not the best-performing training run.
    folder = root / "lambda_0.95_seed_0"
    checkpoint = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["model"])
    norm = RunningNorm(10)
    norm.load(checkpoint["norm"])
    torch.set_num_threads(1)
    episodes = json.loads((folder / "test.json").read_text())["episodes"]
    ordered = sorted(episodes, key=lambda r: r["distance"])
    chosen = [ordered[len(ordered) // 2], ordered[0], ordered[-1]]
    titles = [
        "Median final distance",
        "Lowest final distance",
        "Highest final distance",
    ]
    out = root / "demo"
    out.mkdir(exist_ok=True)
    frames = []
    metadata = []
    font = ImageFont.truetype("/usr/local/lib/python3.13/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf", 20)
    for idx, (row, title) in enumerate(zip(chosen, titles)):
        seed = row["episode_seed"]
        rng = np.random.default_rng(seed + 500000)
        e1 = gym.make("Reacher-v5", render_mode="rgb_array", width=560, height=420)
        e2 = gym.make("Reacher-v5", render_mode="rgb_array", width=560, height=420)
        o1, _ = e1.reset(seed=seed)
        o2, _ = e2.reset(seed=seed)
        local = []
        for step in range(51):
            frame = Image.new("RGB", (1120, 540), "white")
            frame.paste(Image.fromarray(e1.render()), (0, 95))
            frame.paste(Image.fromarray(e2.render()), (560, 95))
            draw = ImageDraw.Draw(frame)
            draw.text(
                (18, 10),
                f"{title} | episode seed {seed} | step {step}/50",
                fill="black",
                font=font,
            )
            draw.text(
                (18, 42), "PPO GAE 0.95, training seed 0", fill="black", font=font
            )
            draw.text(
                (580, 42), "Uniform random torques, same reset", fill="black", font=font
            )
            draw.text(
                (18, 72),
                f"Distance {np.linalg.norm(o1[-2:]):.3f} m",
                fill="black",
                font=font,
            )
            draw.text(
                (580, 72),
                f"Distance {np.linalg.norm(o2[-2:]):.3f} m",
                fill="black",
                font=font,
            )
            local.append(np.array(frame))
            if step == 25:
                frame.save(out / f"example_{idx+1}.png")
            if step == 50:
                break
            with torch.no_grad():
                a = model.action(torch.from_numpy(norm.apply(o1)).unsqueeze(0), True)[
                    0
                ][0].numpy()
            o1, *_ = e1.step(a)
            o2, *_ = e2.step(rng.uniform(-1, 1, 2).astype(np.float32))
        # 10 frames/s shows each 1-second simulated episode over 5 seconds.
        frames.extend([local[0]] * 20 + local + [local[-1]] * 20)
        e1.close()
        e2.close()
        metadata.append(dict(selection=title, **row))
    imageio.mimsave(out / "Reacher_PPO_Demo.mp4", frames, fps=10, macro_block_size=1)
    (out / "selection.json").write_text(json.dumps(metadata, indent=2))
    print("Saved measured demo examples and selection metadata.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs")
    main(p.parse_args().output)
