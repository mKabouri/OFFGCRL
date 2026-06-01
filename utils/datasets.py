from typing import Any

import jax.numpy as jnp
import numpy as np


def make_episode_indices(dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dones = np.asarray(dones, dtype=bool)
    done_positions = np.where(dones)[0]
    if len(done_positions) == 0:
        done_positions = np.array([len(dones) - 1])

    episode_end_idx = np.zeros(len(dones), dtype=np.int32)
    start = 0
    for end in done_positions:
        episode_end_idx[start : end + 1] = end
        start = end + 1
    if start < len(dones):
        episode_end_idx[start:] = len(dones) - 1

    valid_indices = np.where(np.arange(len(dones)) < episode_end_idx)[0].astype(np.int32)
    return episode_end_idx, valid_indices


class Dataset:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        rewards: np.ndarray | None = None,
        **extra_arrays: np.ndarray,
    ):
        self.observations = np.asarray(observations)
        self.actions = np.asarray(actions)
        self.next_observations = np.asarray(next_observations)
        self.dones = np.asarray(dones)
        self.rewards = None if rewards is None else np.asarray(rewards)
        self.extra_arrays = {name: np.asarray(value) for name, value in extra_arrays.items()}

    def __getitem__(self, name: str) -> Any:
        if name in self.extra_arrays:
            return self.extra_arrays[name]
        return getattr(self, name)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def sample_indices(
        self,
        batch_size: int,
        valid_indices: np.ndarray | None = None,
        replace: bool = True,
    ) -> np.ndarray:
        indices = np.arange(len(self), dtype=np.int32) if valid_indices is None else valid_indices
        return np.random.choice(indices, size=batch_size, replace=replace).astype(np.int32)

    def get_batch(self, idx: np.ndarray, *names: str) -> dict[str, jnp.ndarray]:
        batch = {}
        for name in names:
            value = self[name]
            if value is not None:
                batch[name] = jnp.asarray(value[idx])
        return batch

    def get_transition_batch(self, idx: np.ndarray, include_rewards: bool = True) -> dict[str, jnp.ndarray]:
        names = ["observations", "actions", "next_observations", "dones"]
        if include_rewards and self.rewards is not None:
            names.append("rewards")
        return self.get_batch(idx, *names)

    @classmethod
    def create(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        rewards: np.ndarray | None = None,
        **extra_arrays: np.ndarray,
    ):
        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            **extra_arrays,
        )
