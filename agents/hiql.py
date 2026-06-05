import functools
from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from models.networks import ActorNetwork, ValueNetwork
from models.encoders import make_observation_encoder


class HIQLAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Dict[str, TrainState]
    cfg: ml_collections.ConfigDict = flax.struct.field(pytree_node=False)

    def expectile_loss(self, diff: jnp.ndarray) -> jnp.ndarray:
        weights = jnp.where(diff > 0, self.cfg.expectile_coeff, 1 - self.cfg.expectile_coeff)
        return (weights * diff**2).mean()

    def update_value(self, batch: Any):
        v_tp1 = self.train_states["target_value"].apply_fn(
            {"params": self.train_states["target_value"].params},
            batch["next_observations"], batch["goals"]
        )
        if v_tp1.ndim == 2:
            v_tp1 = v_tp1.mean(axis=0)
        td_target = batch["rewards"] + self.cfg.discount * (1.0 - batch["dones"]) * v_tp1
        td_target = jax.lax.stop_gradient(td_target)

        def loss_fn(value_params):
            v = self.train_states["value"].apply_fn(
                {"params": value_params},
                batch["observations"], batch["goals"]
            )
            loss = self.expectile_loss(td_target - v)
            v_ensemble_std = jnp.array(0.0)
            if v.ndim == 2:
                v_ensemble_std = v.std(axis=0).mean()
            return loss, {"value_loss": loss, "v_mean": v.mean(), "v_ensemble_std": v_ensemble_std}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.train_states["value"].params
        )
        train_states = dict(self.train_states)
        train_states["value"] = self.train_states["value"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def update_high_level(self, batch: Any):
        ts = self.train_states

        v_s = ts["value"].apply_fn(
            {"params": ts["value"].params}, batch["observations"], batch["goals"]
        )
        v_subgoal = ts["value"].apply_fn(
            {"params": ts["value"].params}, batch["subgoal_observations"], batch["goals"]
        )
        if v_s.ndim == 2:
            v_s = v_s.mean(axis=0)
            v_subgoal = v_subgoal.mean(axis=0)
        advantage = v_subgoal - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(high_level_actor_params):
            enc_subgoals = ts["high_level_actor"].apply_fn(
                {"params": high_level_actor_params},
                batch["subgoal_observations"],
                method=ActorNetwork.encode_observation,
            )
            enc_subgoals = jax.lax.stop_gradient(enc_subgoals)
            dist = ts["high_level_actor"].apply_fn(
                {"params": high_level_actor_params}, batch["observations"], batch["goals"]
            )
            loss = -(weights * dist.log_prob(enc_subgoals)).mean()
            return loss, {"high_level_actor_loss": loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts["high_level_actor"].params
        )
        train_states = dict(ts)
        train_states["high_level_actor"] = ts["high_level_actor"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def update_low_level(self, batch: Any):
        ts = self.train_states

        v_s = ts["value"].apply_fn(
            {"params": ts["value"].params}, batch["observations"], batch["subgoal_observations"]
        )
        v_tp1 = ts["value"].apply_fn(
            {"params": ts["value"].params}, batch["next_observations"], batch["subgoal_observations"]
        )
        if v_s.ndim == 2:
            v_s = v_s.mean(axis=0)
            v_tp1 = v_tp1.mean(axis=0)
        advantage = v_tp1 - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(low_level_actor_params):
            enc_subgoals = ts["high_level_actor"].apply_fn(
                {"params": ts["high_level_actor"].params},
                batch["subgoal_observations"],
                method=ActorNetwork.encode_observation,
            )
            enc_subgoals = jax.lax.stop_gradient(enc_subgoals)
            dist = ts["low_level_actor"].apply_fn(
                {"params": low_level_actor_params},
                batch["observations"],
                enc_subgoals,
                goal_encoded=True,
            )
            loss = -(weights * dist.log_prob(batch["actions"])).mean()
            return loss, {"low_level_actor_loss": loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts["low_level_actor"].params
        )
        train_states = dict(ts)
        train_states["low_level_actor"] = ts["low_level_actor"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def soft_update_target_value(self):
        tau = self.cfg.tau
        new_params = jax.tree.map(
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
        ts = self.train_states

        high_dist = ts["high_level_actor"].apply_fn(
            {"params": ts["high_level_actor"].params}, obs, goal
        )
        subgoal = high_dist.mode() if deterministic else high_dist.sample(seed=key_h)

        low_dist = ts["low_level_actor"].apply_fn(
            {"params": ts["low_level_actor"].params}, obs, subgoal, goal_encoded=True
        )
        return low_dist.mode() if deterministic else low_dist.sample(seed=key_l)

    @classmethod
    def create(
        cls,
        rng: jax.random.PRNGKey,
        obs_dim: tuple[int, ...],
        action_dim: int,
        cfg: ml_collections.ConfigDict,
    ):
        obs_shape = tuple(obs_dim)
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_goal = jnp.zeros((1, *obs_shape))

        encoder = make_observation_encoder(cfg, dummy_obs)

        rng, enc_key, *keys = jax.random.split(rng, 6)
        enc_params = encoder.init(enc_key, dummy_obs).get("params", {})
        high_level_action_dim = encoder.apply({"params": enc_params}, dummy_obs).shape[-1]

        networks = {
            "value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.value_ensemble_size,
                encoder=encoder,
            ),
            "target_value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.value_ensemble_size,
                encoder=encoder,
            ),
            "high_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=high_level_action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                encoder=encoder,
            ),
            "low_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                encoder=encoder,
            ),
        }

        params = {
            "value": networks["value"].init(keys[0], dummy_obs, dummy_goal)["params"],
            "target_value": networks["target_value"].init(keys[1], dummy_obs, dummy_goal)["params"],
            "high_level_actor": networks["high_level_actor"].init(keys[2], dummy_obs, dummy_goal)["params"],
            "low_level_actor": networks["low_level_actor"].init(keys[3], dummy_obs, dummy_goal)["params"],
        }

        train_states = {
            "value": TrainState.create(
                apply_fn=networks["value"].apply,
                params=params["value"],
                tx=optax.adam(cfg.value_lr)
            ),
            "target_value": TrainState.create(
                apply_fn=networks["target_value"].apply,
                params=params["target_value"],
                tx=optax.set_to_zero(),
            ),
            "high_level_actor": TrainState.create(
                apply_fn=networks["high_level_actor"].apply,
                params=params["high_level_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "low_level_actor": TrainState.create(
                apply_fn=networks["low_level_actor"].apply,
                params=params["low_level_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
        }
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
            # encoder (overridden by main.py --encoder args)
            encoder="drqv2",
            encoder_feature_dim=512,
            encoder_num_features=32,
        )
    )
