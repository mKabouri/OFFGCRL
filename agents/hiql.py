import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from models.networks import ActorNetwork, Identity, ValueNetwork


class HIQLAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Any
    cfg: ml_collections.ConfigDict = flax.struct.field(pytree_node=False)

    def encode(self, image: jnp.ndarray, encoder_params: Any) -> jnp.ndarray:
        return self.train_states["encoder"].apply_fn({"params": encoder_params}, image)

    def encode_pair(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        encoder_params: Any | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if encoder_params is None:
            encoder_params = self.train_states["encoder"].params

        return self.encode(observations, encoder_params), self.encode(goals, encoder_params)

    def expectile_loss(self, diff: jnp.ndarray) -> jnp.ndarray:
        weights = jnp.where(diff > 0, self.cfg.expectile_coeff, 1 - self.cfg.expectile_coeff)
        return (weights * diff**2).mean()

    def update_value(self, batch: Any):
        next_observations, target_goals = self.encode_pair(
            batch["next_observations"],
            batch["goals"],
        )
        v_tp1 = self.train_states["target_value"].apply_fn(
            {"params": self.train_states["target_value"].params},
            next_observations,
            target_goals,
        )
        if v_tp1.ndim == 2:
            v_tp1 = v_tp1.mean(axis=0)
        td_target = batch["rewards"] + self.cfg.discount * (1.0 - batch["dones"]) * v_tp1
        td_target = jax.lax.stop_gradient(td_target)

        def loss_fn(params):
            observations, goals = self.encode_pair(
                batch["observations"],
                batch["goals"],
                params["encoder"],
            )

            v = self.train_states["value"].apply_fn({"params": params["value"]}, observations, goals)
            loss = self.expectile_loss(td_target - v)
            v_ensemble_std = jnp.array(0.0)
            if v.ndim == 2:
                v_ensemble_std = v.std(axis=0).mean()

            return loss, {
                "value_loss": loss,
                "v_mean": v.mean(),
                "v_ensemble_std": v_ensemble_std,
            }

        params = {
            "value": self.train_states["value"].params,
            "encoder": self.train_states["encoder"].params,
        }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        train_states = dict(self.train_states)
        train_states["value"] = self.train_states["value"].apply_gradients(grads=grads["value"])
        train_states["encoder"] = self.train_states["encoder"].apply_gradients(grads=grads["encoder"])
        return self.replace(train_states=train_states), metrics

    def update_high_level(self, batch: Any):
        observations, goals = self.encode_pair(batch["observations"], batch["goals"])
        subgoals, _ = self.encode_pair(batch["subgoal_observations"], batch["goals"])

        v_s = self.train_states["value"].apply_fn(
            {"params": self.train_states["value"].params},
            observations,
            goals,
        )
        v_subgoal = self.train_states["value"].apply_fn(
            {"params": self.train_states["value"].params},
            subgoals,
            goals,
        )
        if v_s.ndim == 2:
            v_s = v_s.mean(axis=0)
            v_subgoal = v_subgoal.mean(axis=0)
        advantage = v_subgoal - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(params):
            observations, goals = self.encode_pair(
                batch["observations"],
                batch["goals"],
                params["encoder"],
            )
            subgoals, _ = self.encode_pair(
                batch["subgoal_observations"],
                batch["goals"],
                params["encoder"],
            )
            subgoals = jax.lax.stop_gradient(subgoals)

            dist = self.train_states["high_level_actor"].apply_fn(
                {"params": params["high_level_actor"]},
                observations,
                goals,
            )
            log_prob = dist.log_prob(subgoals)
            loss = -(weights * log_prob).mean()
            return loss, {"high_level_actor_loss": loss}

        params = {
            "high_level_actor": self.train_states["high_level_actor"].params,
            "encoder": self.train_states["encoder"].params,
        }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        train_states = dict(self.train_states)
        train_states["high_level_actor"] = self.train_states["high_level_actor"].apply_gradients(
            grads=grads["high_level_actor"]
        )
        train_states["encoder"] = self.train_states["encoder"].apply_gradients(grads=grads["encoder"])
        return self.replace(train_states=train_states), metrics

    def update_low_level(self, batch: Any):
        observations, subgoals = self.encode_pair(batch["observations"], batch["subgoal_observations"])
        next_observations, _ = self.encode_pair(batch["next_observations"], batch["subgoal_observations"])

        v_s = self.train_states["value"].apply_fn(
            {"params": self.train_states["value"].params},
            observations,
            subgoals,
        )
        v_tp1 = self.train_states["value"].apply_fn(
            {"params": self.train_states["value"].params},
            next_observations,
            subgoals,
        )
        if v_s.ndim == 2:
            v_s = v_s.mean(axis=0)
            v_tp1 = v_tp1.mean(axis=0)
        advantage = v_tp1 - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(params):
            observations, subgoals = self.encode_pair(
                batch["observations"],
                batch["subgoal_observations"],
                params["encoder"],
            )

            dist = self.train_states["low_level_actor"].apply_fn(
                {"params": params["low_level_actor"]},
                observations,
                subgoals,
            )
            log_prob = dist.log_prob(batch["actions"])
            loss = -(weights * log_prob).mean()
            return loss, {"low_level_actor_loss": loss}

        params = {
            "low_level_actor": self.train_states["low_level_actor"].params,
            "encoder": self.train_states["encoder"].params,
        }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        train_states = dict(self.train_states)
        train_states["low_level_actor"] = self.train_states["low_level_actor"].apply_gradients(
            grads=grads["low_level_actor"]
        )
        train_states["encoder"] = self.train_states["encoder"].apply_gradients(grads=grads["encoder"])
        return self.replace(train_states=train_states), metrics

    def soft_update_target_value(self):
        tau = self.cfg.tau
        new_params = jax.tree_util.tree_map(
            lambda target, online: tau * target + (1 - tau) * online,
            self.train_states["target_value"].params,
            self.train_states["value"].params,
        )
        new_ts = self.train_states["target_value"].replace(params=new_params)
        return self.replace(train_states={**self.train_states, "target_value": new_ts})

    @jax.jit
    def update(self, batch: Any):
        agent, value_logs = self.update_value(batch)
        agent, low_logs = agent.update_low_level(batch)
        agent, high_logs = agent.update_high_level(batch)
        agent = agent.soft_update_target_value()
        return agent, {**value_logs, **low_logs, **high_logs}

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self, obs: jnp.ndarray, goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        rng, key_h, key_l = jax.random.split(rng, 3)
        obs, goal = self.encode_pair(obs, goal)

        high_dist = self.train_states["high_level_actor"].apply_fn(
            {"params": self.train_states["high_level_actor"].params},
            obs,
            goal,
        )
        if deterministic:
            subgoal = high_dist.mode()
        else:
            subgoal = high_dist.sample(seed=key_h)

        low_dist = self.train_states["low_level_actor"].apply_fn(
            {"params": self.train_states["low_level_actor"].params},
            obs,
            subgoal,
        )
        if deterministic:
            return low_dist.mode()
        return low_dist.sample(seed=key_l)

    @classmethod
    def create(
        cls,
        rng: jax.random.PRNGKey,
        obs_dim: tuple[int, ...],
        action_dim: int,
        cfg: ml_collections.ConfigDict,
        encoders: dict[str, nn.Module] | None = None,
    ):
        obs_shape = tuple(obs_dim)
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_goal = jnp.zeros((1, *obs_shape))

        if encoders is None:
            encoders = {"encoder": Identity()}

        encoder = encoders["encoder"]

        rng, *keys = jax.random.split(rng, 6)
        train_states = {}

        encoder_params = encoder.init(keys[4], dummy_obs).get("params", {})
        dummy_obs = encoder.apply({"params": encoder_params}, dummy_obs)
        dummy_goal = encoder.apply({"params": encoder_params}, dummy_goal)
        train_states["encoder"] = TrainState.create(
            apply_fn=encoder.apply,
            params=encoder_params,
            tx=optax.adam(cfg.value_lr),
        )
        high_level_action_dim = dummy_obs.shape[-1]

        networks = {
            "value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.value_ensemble_size,
            ),
            "target_value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.value_ensemble_size,
            ),
            "high_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=high_level_action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
            ),
            "low_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
            ),
        }

        params = {
            "value": networks["value"].init(keys[0], dummy_obs, dummy_goal)["params"],
            "target_value": networks["target_value"].init(keys[1], dummy_obs, dummy_goal)["params"],
            "high_level_actor": networks["high_level_actor"].init(keys[2], dummy_obs, dummy_goal)["params"],
            "low_level_actor": networks["low_level_actor"].init(keys[3], dummy_obs, dummy_goal)["params"],
        }

        train_states["value"] = TrainState.create(
            apply_fn=networks["value"].apply, params=params["value"], tx=optax.adam(cfg.value_lr)
        )
        train_states["target_value"] = TrainState.create(
            apply_fn=networks["target_value"].apply, params=params["target_value"], tx=optax.set_to_zero()
        )
        train_states["high_level_actor"] = TrainState.create(
            apply_fn=networks["high_level_actor"].apply, params=params["high_level_actor"], tx=optax.adam(cfg.actor_lr)
        )
        train_states["low_level_actor"] = TrainState.create(
            apply_fn=networks["low_level_actor"].apply, params=params["low_level_actor"], tx=optax.adam(cfg.actor_lr)
        )
        return cls(rng=rng, train_states=train_states, cfg=cfg)


def get_default_config():
    return ml_collections.ConfigDict(
        dict(
            # networks
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            activations=nn.gelu,
            kernel_init=nn.initializers.orthogonal(),
            value_ensemble_size=2,
            # training
            actor_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile_coeff=0.75,
            beta=3.0,
            # hierarchy
            subgoal_steps=25,
            value_p_curgoal=0.2,
        )
    )
