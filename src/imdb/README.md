# IMDb experiments

The benchmark compares five PPO methods: `vanilla`, `maxmin`, `sumgrad`, `pcgrad`, and `mgda`.

See the [main README](../../README.md) for the experimental setting, setup, and complete training and evaluation commands. Run commands from the repository root.

## Components

| Directory | Purpose |
|---|---|
| [dataset](dataset) | Generate candidate responses, score sentiment and length, and construct group-labeled preference pairs. |
| [reward_training](reward_training) | Train the pooled Vanilla reward model and separate minority/majority reward models; optionally normalize rewards. |
| [ppo_training](ppo_training) | Train policies with pooled rewards, batch-level worst-group selection, or group-gradient combination. |
| [evaluation](evaluation) | Generate and score policy responses, and summarize evaluation datasets. |
| [diagnostics](diagnostics) | Inspect gradient alignment, reward scales, and calibration. |

SumGrad, PCGrad, and MGDA reuse the same minority and majority reward models as MaxMin. Sentiment and response-length scoring provide the oracle evaluation metrics for all five methods.
