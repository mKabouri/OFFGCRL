import argparse
import os
from datetime import datetime

import jax
import numpy as np
import ogbench
from tqdm import tqdm

from agents import agents
from agents import hiql, crl
from agents.flow import fql as fql_module
from utils.augmentations import random_shift_image_batch
from utils.device_replay import make_device_replay_sampler
from utils.logging_utils import evaluate, init_wandb, log_metrics, record_video, save_checkpoint
from utils.replay_buffer import GCRLReplayBuffer, HierarchicalGCRLReplayBuffer

REPLAY_BUFFERS = {
    "hiql": HierarchicalGCRLReplayBuffer,
    "crl": GCRLReplayBuffer,
    "fql": GCRLReplayBuffer,
}

AGENT_MODULES = {
    "hiql": hiql,
    "crl": crl,
    "fql": fql_module,
}


def main():
    parser = argparse.ArgumentParser(description="Offline Goal-Conditioned RL on OGBench")

    parser.add_argument("--env", default="visual-antmaze-large-navigate-v0",
                        help="OGBench environment (for example: visual-antmaze-large-navigate-v0, antmaze-large-navigate-v0)")
    parser.add_argument("--agent", default="hiql", choices=list(AGENT_MODULES))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder", default="drqv2", choices=["drqv2", "resnet", "none"],
                        help="Image encoder for visual observations (ignored for state-based envs)")
    parser.add_argument("--encoder_feature_dim", type=int, default=512)
    parser.add_argument("--encoder_num_features", type=int, default=32)

    parser.add_argument("--num_steps", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--reward_type", default="neg_one_zero", choices=["neg_one_zero", "zero_one"])
    parser.add_argument("--data_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_augmentation_padding", type=int, default=3)
    parser.add_argument("--data_augmentation_probability", type=float, default=1.0)
    parser.add_argument("--agent_alpha", type=float, default=0.0,
                        help="Override the agent's alpha hyperparameter (e.g. CRL BC weight)")

    parser.add_argument("--log_interval", type=int, default=10_000)
    parser.add_argument("--eval_interval", type=int, default=200_000)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--record_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--wandb_project", default="gcrl")
    parser.add_argument("--wandb_name", default="")

    args = parser.parse_args()
    checkpoint_dir = os.path.join(args.checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M"))

    agent_module = AGENT_MODULES[args.agent]
    agent_cfg = agent_module.get_default_config()
    if args.agent_alpha > 0.0 and hasattr(agent_cfg, "alpha"):
        agent_cfg.alpha = args.agent_alpha
    agent_cfg.encoder = args.encoder
    agent_cfg.encoder_feature_dim = args.encoder_feature_dim
    agent_cfg.encoder_num_features = args.encoder_num_features

    init_wandb(vars(args), agent_cfg=agent_cfg, project=args.wandb_project, name=args.wandb_name)

    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    backend = jax.default_backend()
    print(f"JAX backend: {backend}, devices: {jax.devices()}")
    if args.env.startswith("visual-") and backend != "gpu":
        print("WARNING: visual training without GPU. Set JAX_PLATFORMS=cuda for GPU.")

    env, train_dataset, _ = ogbench.make_env_and_datasets(args.env)
    is_visual = train_dataset["observations"].ndim == 4

    subgoal_steps = getattr(agent_cfg, "subgoal_steps", 0)
    current_goal_probability = getattr(agent_cfg, "value_p_curgoal", 0.0)

    replay_buffer = REPLAY_BUFFERS[args.agent].create(
        observations=train_dataset["observations"],
        actions=train_dataset["actions"],
        next_observations=train_dataset["next_observations"],
        dones=train_dataset["terminals"],
        subgoal_steps=subgoal_steps,
        current_goal_probability=current_goal_probability,
    )

    agent = agents[args.agent].create(
        rng,
        env.observation_space.shape,
        env.action_space.shape[0],
        agent_cfg,
    )

    if not is_visual:
        sample_batch = make_device_replay_sampler(replay_buffer, args.batch_size, args.reward_type)

        @jax.jit
        def train_step(agent):
            rng, sample_rng = jax.random.split(agent.rng)
            batch = sample_batch(sample_rng)
            agent = agent.replace(rng=rng)
            return agent.update(batch)

    else:
        @jax.jit
        def train_step(agent, batch, rng):
            if args.data_augmentation:
                rng, aug_rng = jax.random.split(rng)
                batch = random_shift_image_batch(
                    batch, aug_rng,
                    padding=args.data_augmentation_padding,
                    probability=args.data_augmentation_probability,
                )
            agent = agent.replace(rng=rng)
            return agent.update(batch)

    for i in tqdm(range(1, args.num_steps + 1)):
        if not is_visual:
            agent, update_logs = train_step(agent)
        else:
            rng, step_rng = jax.random.split(rng)
            batch = replay_buffer.sample(args.batch_size, reward_type=args.reward_type)
            agent, update_logs = train_step(agent, batch, step_rng)

        if i % args.log_interval == 0:
            log_metrics(update_logs, step=i, prefix="train")

        if i % args.eval_interval == 0:
            eval_rng = agent.rng if not is_visual else rng
            eval_logs = evaluate(agent, env, num_episodes=args.num_eval_episodes, rng=eval_rng)
            log_metrics(eval_logs, step=i, prefix="eval")

            if args.record_video:
                record_video(agent, env, eval_rng, step=i, key="eval/video")
            if args.save_checkpoints:
                save_checkpoint(agent, checkpoint_dir, step=i)

    if args.save_checkpoints:
        save_checkpoint(agent, checkpoint_dir, step=args.num_steps)


if __name__ == "__main__":
    main()
