# PPO for simulated robotic reaching

This repository continues Part 1's Reacher-v5 project. It includes a custom PyTorch PPO implementation, GAE, independent seeds, fixed validation and held-out evaluation, measured outputs, and a recorded simulation demo. The environment has 10 observations and two bounded continuous joint-torque actions. No physical robot is used.

## Setup

Use Python 3.11 or later in a virtual environment. From this folder:

```sh
python -m venv .venv
```

&nbsp;

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

&nbsp;

macOS or Linux:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

&nbsp;

The accompanying `outputs/versions.json` records the actual execution versions. CPU execution is sufficient. On Windows, if PyTorch installation fails, use its official installation selector for a compatible CPU wheel. A new installation can have slightly different numerical behavior.

## Run

```sh
python checks.py
python train.py --pilot --output pilot_outputs
python train.py
python analyze.py
python record_demo.py
```

&nbsp;

The full experiment executes six runs. Each uses 98 rollouts of 2,048 steps, totaling 200,704 transitions. Validation occurs after the rollout crossing each 20,000-step boundary, plus the final rollout. Package versions, runtime, checkpoints, raw per-episode evaluations, and raw validation logs are retained. No favorable checkpoint selection is used. Re-running training overwrites the local output files. Copy the supplied outputs before running if you want to preserve them.

For a short installation check, use `--pilot`. It does not replace the full experiment. On Linux without a display, rendering uses `MUJOCO_GL=egl`; the demo script sets this by default. Windows users can set `MUJOCO_GL=glfw` before running if EGL is unavailable. Training itself does not require a display.

## Project files

- `src/model.py`: separate actor and critic, stable transformed Gaussian log probability, observation statistics.  
- `src/ppo.py`: PPO clipped policy objective, critic loss, transformed-policy entropy estimate, GAE masks.  
- `src/evaluate.py`: frozen normalization and deterministic evaluation.  
- `train.py`: rollouts, optimization, checkpoint and raw log creation.  
- `checks.py`: probability and episode-mask correctness checks.  
- `analyze.py`: descriptive summaries, CSV and learning curves.  
- `record_demo.py`: actual simulation footage for three transparent examples.  
- `config.json`: the complete matched experimental configuration.  
- `outputs/`: measured outputs and six final model checkpoints.

## Evaluation protocol

Training seeds: 0, 1, 2\. Validation reset seeds: 10000–10019. Test reset seeds: 20000–20099. Baselines use the same test reset seeds. Uniform-random actions use an independently fixed action RNG. Success requires distance at most 0.03 m at every one of the final ten steps. The success threshold is a project choice, not an official Gymnasium criterion. Report standard deviation across the three training runs, not across pooled episodes. The test seeds sample new initial states and targets from the same task distribution.

The demo uses training seed 0, specified before selecting footage. It displays the median, lowest, and highest final-ten-step distances from that policy's 100 test episodes. Labels disclose this selection. Playback is slowed to ten frames per second and contains brief pauses. It is illustrative; the aggregate table is the performance evidence.

## Connection to Part 1

The environment, actor and critic sizes, Gaussian policy with tanh transformation, PPO settings, GAE comparison, three seeds, default reward, and success threshold match the plan. Necessary implementation details include rollout-aligned evaluation (20,480 steps for the first checkpoint), 200,704 transitions for complete rollouts, frozen preprocessing during each rollout/update, and a Monte Carlo estimate of the bounded policy's entropy. Observation statistics update only from training data after each update. The source code includes the tanh Jacobian correction and separate masks for bootstrapping and stopping advantage recursion. No Stable-Baselines3 runtime dependency or comparison is claimed.

## AI use disclosure

OpenAI ChatGPT assisted with the project design, source-code development, execution of the supplied experiments, result analysis, slide drafting, and narration preparation. The student must review, revise, understand, and verify the materials before submission. The supplied outputs were executed in the assistant's environment. They should not be described as independently run by the student unless the student reruns the code.

## References

See `REFERENCES.md`. Environment code and research sources retain their own licenses. This project contains an original implementation prepared with AI assistance. No claim of human-only authorship is made.

## Execution provenance and limitations

The supplied full outputs were computed in the assistant's Linux CPU environment on October 5, 2026\. A separate pilot verified operation before full training. The pilot did not trigger any hyperparameter tuning. Full training retained the original configuration. Some independent runs executed concurrently to shorten total completion time. Per-run wall-clock times therefore reflect shared CPU load and should not be interpreted as a controlled comparison of computational efficiency. No physical robot or live deployment was tested.

To verify the supplied summaries and a reloaded checkpoint without retraining:

```sh
python verify_outputs.py
```

&nbsp;

The archive also includes a Git bundle. To restore the local commit history:

```sh
git clone Reacher_PPO_Project.bundle restored_project
```

&nbsp;