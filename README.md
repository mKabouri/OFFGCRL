# OFFGCRL: OFFline Goal-Conditioned Reinforcement Learning

This repository contains the implementation of CRL, HIQL, and FQL algorithms for goal-conditioned offline reinforcement learning. It should support both image-based and proprioceptive state-based environments from [OGBench](https://seohong.me/projects/ogbench/).

## TODO:

- Add more algorithms: DQC, TDInfoNCE, Infom, HILP

## Run Training Example:

```bash
python main.py --agent hiql --env antmaze-large-navigate-v0
```


## Credits:

- This repository is inspired by [OGBench](https://seohong.me/projects/ogbench/).
