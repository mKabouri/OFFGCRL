import argparse
import importlib
import os
import sys
from datetime import datetime

import jax
import ml_collections
import numpy as np
import ogbench
from tqdm import tqdm

from agents import agents
from utils.augmentations import random_shift_image_batch
from utils.logging_utils import evaluate, init_wandb, log_metrics, record_video, save_checkpoint
from utils.device_replay import make_device_replay_sampler
from utils.replay_buffer import HierarchicalGCRLReplayBuffer, GCRLReplayBuffer

AGENT_REGISTRY = {
    "hiql": "agents.hiql",
    "crl": "agents.crl",
}


def arg_was_provided(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def apply_env_defaults(cfg: ml_collections.ConfigDict):
    if cfg.agent != "crl" or not cfg.env.startswith("visual-antmaze-"):
        return

    if not arg_was_provided("--batch_size"):
        cfg.batch_size = 256
        print("Defaulting visual AntMaze CRL to --batch_size=256.")
    if not arg_was_provided("--data_augmentation"):
        cfg.data_augmentation = False
        print("Defaulting visual AntMaze CRL to --data_augmentation=False.")


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            env="visual-antmaze-large-navigate-v0",
            seed=42,
            agent="hiql",
            num_steps=2_000_000,
            batch_size=1024,
            encoder="drqv2",
            encoder_feature_dim=512,
            encoder_num_features=32,
            data_augmentation=True,
            data_augmentation_padding=3,
            data_augmentation_probability=1.0,
            log_interval=10_000,
            eval_interval=200_000,
            num_eval_episodes=50,
            record_video=True,
            save_checkpoints=True,
            checkpoint_dir=os.path.abspath("checkpoints"),
            reward_type="neg_one_zero",
            wandb_project="gcrl",
            wandb_name="",
            agent_alpha=0.0,
        )
    )


def parse_args(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
    parser = argparse.ArgumentParser()
    for key, val in cfg.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    parser.add_argument("--env_name", type=str, default=None, help="Alias for --env.")
    args, _ = parser.parse_known_args()
    args = vars(args)
    env_name = args.pop("env_name")
    cfg.update(args)
    if env_name is not None:
        cfg.env = env_name
    return cfg


def main():
    cfg = parse_args(get_config())
    apply_env_defaults(cfg)
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    agent_module = importlib.import_module(AGENT_REGISTRY[cfg.agent])
    agent_cfg = agent_module.get_default_config()
    if cfg.agent_alpha > 0.0 and hasattr(agent_cfg, "alpha"):
        agent_cfg.alpha = cfg.agent_alpha

    init_wandb(
        cfg,
        agent_cfg=agent_cfg,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    np.random.seed(cfg.seed)

    backend = jax.default_backend()
    print(f"JAX backend: {backend}, devices: {jax.devices()}")
    if cfg.env.startswith("visual-") and backend != "gpu":
        print("WARNING: visual training is not using GPU. Launch with JAX_PLATFORMS=cuda for CUDA training.")
    env, train_dataset, _ = ogbench.make_env_and_datasets(cfg.env)
    use_device_replay = train_dataset["observations"].ndim < 4
    if use_device_replay:
        cfg.data_augmentation = False
    encoders = agent_module.build_encoders(cfg, train_dataset["observations"])
    replay_buffers = {
        "hiql": HierarchicalGCRLReplayBuffer,
        "crl": GCRLReplayBuffer,
    }
    subgoal_steps = 0
    if hasattr(agent_cfg, "subgoal_steps"):
        subgoal_steps = agent_cfg.subgoal_steps

    replay_buffer = replay_buffers[cfg.agent].create(
        observations=train_dataset["observations"],
        actions=train_dataset["actions"],
        next_observations=train_dataset["next_observations"],
        dones=train_dataset["terminals"],
        subgoal_steps=subgoal_steps,
        current_goal_probability=getattr(agent_cfg, "value_p_curgoal", 0.0),
    )
    agent = agents[cfg.agent].create(
        rng,
        env.observation_space.shape,
        env.action_space.shape[0],
        agent_cfg,
        encoders=encoders,
    )

    if use_device_replay:
        sample_device_batch = make_device_replay_sampler(
            replay_buffer,
            cfg.agent,
            cfg.batch_size,
            cfg.reward_type,
        )

        @jax.jit
        def train_step(agent):
            rng, sample_rng = jax.random.split(agent.rng)
            batch = sample_device_batch(sample_rng)
            agent = agent.replace(rng=rng)
            return agent.update(batch)

    else:

        @jax.jit
        def train_step(agent, batch, rng):
            if cfg.data_augmentation:
                rng, augmentation_rng = jax.random.split(rng)
                batch = random_shift_image_batch(
                    batch,
                    augmentation_rng,
                    padding=cfg.data_augmentation_padding,
                    probability=cfg.data_augmentation_probability,
                )

            agent = agent.replace(rng=rng)
            return agent.update(batch)

    for i in tqdm(range(1, cfg.num_steps + 1)):
        if use_device_replay:
            agent, update_logs = train_step(agent)
        else:
            rng, step_rng = jax.random.split(rng)
            batch = replay_buffer.sample(
                cfg.batch_size,
                reward_type=cfg.reward_type,
            )
            agent, update_logs = train_step(agent, batch, step_rng)

        if i % cfg.log_interval == 0:
            log_metrics(update_logs, step=i, prefix="train")

        if i % cfg.eval_interval == 0:
            eval_rng = agent.rng if use_device_replay else rng
            eval_logs = evaluate(agent, env, num_episodes=cfg.num_eval_episodes, rng=eval_rng)
            log_metrics(eval_logs, step=i, prefix="eval")

            if cfg.record_video:
                record_video(agent, env, eval_rng, step=i, key="eval/video")
            if cfg.save_checkpoints:
                save_checkpoint(agent, cfg.checkpoint_dir, step=i)

    if cfg.save_checkpoints:
        save_checkpoint(agent, cfg.checkpoint_dir, step=cfg.num_steps)


if __name__ == "__main__":
    main()
