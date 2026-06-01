from typing import Literal

import jax.numpy as jnp
import numpy as np

from utils.datasets import Dataset, make_episode_indices


def rewards_from_indices(
    idx: np.ndarray,
    goal_idx: np.ndarray,
    reward_type: Literal["neg_one_zero", "zero_one"],
) -> jnp.ndarray:
    achieved = jnp.asarray(idx == goal_idx, dtype=jnp.float32)
    if reward_type == "neg_one_zero":
        return achieved - 1.0
    return achieved


def goal_dones_from_indices(idx: np.ndarray, goal_idx: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(idx == goal_idx, dtype=jnp.float32)


class GCRLReplayBuffer(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        rewards: np.ndarray | None = None,
        current_goal_probability: float = 0.0,
    ):
        super().__init__(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )
        self.episode_end_idx, self.valid_indices = make_episode_indices(self.dones)
        self.current_goal_probability = current_goal_probability

    def sample_goal_indices(self, idx: np.ndarray) -> np.ndarray:
        episode_ends = self.episode_end_idx[idx]
        steps_remaining = episode_ends - idx
        offsets = np.floor(np.random.random(len(idx)) * steps_remaining).astype(np.int32) + 1
        goal_idx = idx + offsets

        if self.current_goal_probability > 0.0:
            use_current_goal = np.random.random(len(idx)) < self.current_goal_probability
            goal_idx = np.where(use_current_goal, idx, goal_idx)

        return goal_idx.astype(np.int32)

    def sample_indices_and_goals(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.sample_indices(batch_size, valid_indices=self.valid_indices, replace=True)
        goal_idx = self.sample_goal_indices(idx)
        goals = self.observations[goal_idx]
        return idx, goal_idx, goals

    def make_goal_info(
        self,
        idx: np.ndarray,
        goal_idx: np.ndarray,
        reward_type: Literal["neg_one_zero", "zero_one"],
    ) -> dict[str, jnp.ndarray]:
        dones = goal_dones_from_indices(idx, goal_idx)
        return {
            "indices": jnp.asarray(idx),
            "goal_indices": jnp.asarray(goal_idx),
            "rewards": rewards_from_indices(idx, goal_idx, reward_type),
            "dones": dones,
            "masks": 1.0 - dones,
        }

    def sample(
        self,
        batch_size: int,
        reward_type: Literal["neg_one_zero", "zero_one"] = "neg_one_zero",
    ) -> dict[str, jnp.ndarray]:
        idx, goal_idx, goals = self.sample_indices_and_goals(batch_size)
        batch = self.get_batch(idx, "observations", "actions")
        batch["goals"] = jnp.asarray(goals)
        batch["indices"] = jnp.asarray(idx)
        batch["goal_indices"] = jnp.asarray(goal_idx)
        return batch

    @classmethod
    def create(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        subgoal_steps: int = 0,  # ignored
        current_goal_probability: float = 0.0,
    ) -> "GCRLReplayBuffer":
        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            current_goal_probability=current_goal_probability,
        )


class HierarchicalGCRLReplayBuffer(GCRLReplayBuffer):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        subgoal_steps: int = 20,
        current_goal_probability: float = 0.0,
    ):
        super().__init__(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            current_goal_probability=current_goal_probability,
        )
        subgoal_idx = np.minimum(np.arange(len(self), dtype=np.int32) + subgoal_steps, self.episode_end_idx)
        self.subgoal_observations = self.observations[subgoal_idx]

    def sample(
        self,
        batch_size: int,
        reward_type: Literal["neg_one_zero", "zero_one"] = "neg_one_zero",
    ) -> dict[str, jnp.ndarray]:
        idx, goal_idx, goals = self.sample_indices_and_goals(batch_size)
        batch = self.get_transition_batch(idx, include_rewards=False)
        batch["goals"] = jnp.asarray(goals)
        batch.update(self.make_goal_info(idx, goal_idx, reward_type))
        batch["subgoal_observations"] = jnp.asarray(self.subgoal_observations[idx])
        return batch

    @classmethod
    def create(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        subgoal_steps: int = 20,
        current_goal_probability: float = 0.0,
    ) -> "HierarchicalGCRLReplayBuffer":
        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            subgoal_steps=subgoal_steps,
            current_goal_probability=current_goal_probability,
        )
