# Reading the supplied experiment

The fixed configuration comes from the Part 1 plan: PPO, Reacher-v5, two separate 64-unit hidden layers per network, a bounded Gaussian policy, GAE, and three training seeds. Both advantage configurations use exactly 200,704 environment transitions per run.

The final mean should be calculated across the three run-level summaries. Every run-level summary itself averages 100 test episodes. The baseline summaries each average 100 episodes. They do not have training-seed standard deviations because they are not trained models.

Validation is descriptive and uses fixed reset seeds. The first checkpoint is 20,480 transitions, the next is 40,960, and the final is 200,704. These offsets arise from completing whole rollouts before evaluation.

The source separates true termination from time-limit truncation. A truncation still bootstraps from its final state, but the GAE recursion stops at the reset boundary. The sampled latent Gaussian variable is stored and reused in probability calculations, with the tanh Jacobian correction included.

The pilot was separate and used no tuning. The fixed final checkpoint determines test performance. One interrupted validation log was rebuilt by rerunning the same seed and configuration. The repeated run was not selected for a better outcome. Its deterministic test metrics reproduced the earlier run. Some full runs executed concurrently. Wall-clock times have this resource-sharing limitation and are not controlled speed comparisons.

The supplied outputs were executed with AI assistance in the assistant environment. After reviewing the methods, you can rerun all commands to obtain your own outputs. Use your own results in the presentation if you rerun, and update every chart and table together. Changes to hardware, versions, or random-number libraries can affect exact values.
