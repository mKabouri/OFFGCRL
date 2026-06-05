import jax
import jax.numpy as jnp


def make_device_replay_sampler(replay_buffer, batch_size: int, reward_type: str):
    observations = jax.device_put(jnp.asarray(replay_buffer.observations))
    actions = jax.device_put(jnp.asarray(replay_buffer.actions))
    next_observations = jax.device_put(jnp.asarray(replay_buffer.next_observations))
    valid_indices = jax.device_put(jnp.asarray(replay_buffer.valid_indices, dtype=jnp.int32))
    episode_end_idx = jax.device_put(jnp.asarray(replay_buffer.episode_end_idx, dtype=jnp.int32))
    current_goal_probability = float(replay_buffer.current_goal_probability)

    subgoal_observations = None
    if hasattr(replay_buffer, "subgoal_observations"):
        subgoal_observations = jax.device_put(jnp.asarray(replay_buffer.subgoal_observations))

    def sample(rng):
        idx_rng, goal_rng, cur_goal_rng = jax.random.split(rng, 3)
        valid_pos = jax.random.randint(idx_rng, (batch_size,), 0, valid_indices.shape[0])
        idx = valid_indices[valid_pos]

        episode_ends = episode_end_idx[idx]
        steps_remaining = jnp.maximum(episode_ends - idx, 1)
        offsets = jnp.floor(jax.random.uniform(goal_rng, (batch_size,)) * steps_remaining).astype(jnp.int32) + 1
        goal_idx = idx + offsets

        if current_goal_probability > 0.0:
            use_current_goal = jax.random.uniform(cur_goal_rng, (batch_size,)) < current_goal_probability
            goal_idx = jnp.where(use_current_goal, idx, goal_idx)

        achieved = (idx == goal_idx).astype(jnp.float32)
        rewards = achieved - 1.0 if reward_type == "neg_one_zero" else achieved

        batch = {
            "observations": observations[idx],
            "actions": actions[idx],
            "goals": observations[goal_idx],
            "next_observations": next_observations[idx],
            "rewards": rewards,
            "masks": 1.0 - achieved,
            "dones": achieved,
            "indices": idx,
            "goal_indices": goal_idx,
        }

        if subgoal_observations is not None:
            batch["subgoal_observations"] = subgoal_observations[idx]

        return batch

    return sample
