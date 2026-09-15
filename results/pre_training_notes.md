# Pre-training notes

Recorded 2026-09-14 17:05 PDT, before the final training run started.

## Settings

| Setting | Value |
|---|---|
| Exploration | 0.10 |
| Episodes | 1000 |
| Learning rate | 0.00025 |

Other change: `SHOW_POPUPS = False` (display only; no effect on training or evaluation).

## Prediction

**Modest improvement.** I expect the trained mean score to beat the untrained baseline, but not by a lot, because the replay memory is small (5,000 decisions) and exploration stays constant.

## Setup check

A 5-episode smoke test (exploration 0.10, learning rate 0.00025) completed on Apple M4 (MPS), Python 3.13.15: 2,790 decisions, 448 learning updates, finite losses, all evidence files produced. Its scores are a pipeline check only, not evidence of learning.

## Learning-rate decision

I considered 0.0001 (smaller, steadier updates) against 0.00025 (the original DQN paper's value). The case for 0.0001 was strongest for a long run: with only 5,000 decisions of replay memory, and the leaderboard scoring the final model, big updates risk overwriting good behavior late in training. Adam also takes larger effective steps than the paper's RMSProp at the same nominal rate.

I chose a fixed budget of **1,000 episodes** to keep training time reasonable, and with that budget I chose **0.00025**. A shorter run leaves less time for late-run forgetting, and 1,000 episodes is a small update budget, so the bigger risk is ending under-trained. I accepted a higher risk of unstable updates in exchange for learning more within limited training.

## Correction (added after training, 2026-09-14)

The learning-rate section above says Adam "takes larger effective steps than the paper's RMSProp at the same nominal rate." That was stated too broadly: how large each optimizer's updates are depends on gradient history and optimizer settings. The accurate point is that the original paper used RMSProp, while this notebook uses Adam, so the same numerical learning rate does not imply equivalent updates. The original text above is left as written before training; the settings and prediction are unchanged.
