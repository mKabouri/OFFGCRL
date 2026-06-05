import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from models.networks import ActorVectorFieldNetwork, QNetwork
from models.encoders import make_observation_encoder


class FQLAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Any
    cfg: ml_collections.ConfigDict = flax.struct.field(pytree_node=False)

    def update_critic(self, batch: Any, rng: jax.random.PRNGKey):
        ts = self.train_states
        rng, noise_rng = jax.random.split(rng)

        noise = jax.random.normal(noise_rng, (batch["observations"].shape[0], self.cfg.action_dim))
        next_actions = ts["one_step_actor"].apply_fn(
            {"params": ts["one_step_actor"].params},
            batch["next_observations"], batch["goals"],
            actions=noise,
        )
        next_actions = jnp.clip(next_actions, -1, 1)

        next_q = ts["target_critic"].apply_fn(
            {"params": ts["target_critic"].params},
            batch["next_observations"], batch["goals"], next_actions,
        )
        next_q = next_q.min(axis=0)

        target_q = batch["rewards"] + self.cfg.discount * batch["masks"] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        def loss_fn(critic_params):
            q = ts["critic"].apply_fn(
                {"params": critic_params},
                batch["observations"], batch["goals"], batch["actions"],
            )
            loss = jnp.mean((q - target_q) ** 2)
            return loss, {
                "critic_loss": loss,
                "q_mean": q.mean(),
                "q_max": q.max(),
                "q_min": q.min(),
                "target_q_mean": target_q.mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts["critic"].params)
        train_states = dict(ts)
        train_states["critic"] = ts["critic"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def update_n_step_flow_actor(self, batch: Any, rng: jax.random.PRNGKey):
        ts = self.train_states
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        batch_size = batch["actions"].shape[0]

        x_0 = jax.random.normal(x_rng, (batch_size, self.cfg.action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel_target = x_1 - x_0

        def loss_fn(actor_params):
            pred = ts["n_step_actor"].apply_fn(
                {"params": actor_params},
                batch["observations"], batch["goals"],
                actions=x_t, times=t,
            )
            loss = jnp.mean((pred - vel_target) ** 2)
            return loss, {
                "bc_flow_loss": loss,
                "vel_norm": jnp.linalg.norm(vel_target, axis=-1).mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts["n_step_actor"].params)
        train_states = dict(ts)
        train_states["n_step_actor"] = ts["n_step_actor"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def update_one_step_flow_actor(self, batch: Any, rng: jax.random.PRNGKey):
        ts = self.train_states
        rng, noise_rng = jax.random.split(rng)
        batch_size = batch["actions"].shape[0]

        noise = jax.random.normal(noise_rng, (batch_size, self.cfg.action_dim))
        target_actions = jax.lax.stop_gradient(
            self._sample_flow_actions(batch["observations"], batch["goals"], noise)
        )

        def loss_fn(actor_params):
            actor_actions = ts["one_step_actor"].apply_fn(
                {"params": actor_params},
                batch["observations"], batch["goals"],
                actions=noise,
            )
            distill_loss = jnp.mean((actor_actions - target_actions) ** 2)

            clipped = jnp.clip(actor_actions, -1, 1)
            q = ts["critic"].apply_fn(
                {"params": ts["critic"].params},
                batch["observations"], batch["goals"], clipped,
            )
            q = q.mean(axis=0)
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = -lam * q.mean()

            loss = self.cfg.alpha * distill_loss + q_loss
            return loss, {
                "one_step_actor_loss": loss,
                "distill_loss": distill_loss,
                "q_loss": q_loss,
                "q_mean": q.mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts["one_step_actor"].params)
        train_states = dict(ts)
        train_states["one_step_actor"] = ts["one_step_actor"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def soft_update_target_critic(self):
        tau = self.cfg.tau
        new_params = jax.tree.map(
            lambda target, online: tau * target + (1 - tau) * online,
            self.train_states["target_critic"].params,
            self.train_states["critic"].params,
        )
        new_ts = self.train_states["target_critic"].replace(params=new_params)
        return self.replace(train_states={**self.train_states, "target_critic": new_ts})

    @jax.jit
    def update(self, batch: Any):
        new_rng, critic_rng, n_step_rng, one_step_rng = jax.random.split(self.rng, 4)
        agent, critic_metrics = self.update_critic(batch, critic_rng)
        agent, n_step_metrics = agent.update_n_step_flow_actor(batch, n_step_rng)
        agent, one_step_metrics = agent.update_one_step_flow_actor(batch, one_step_rng)
        agent = agent.soft_update_target_critic()
        return agent.replace(rng=new_rng), {**critic_metrics, **n_step_metrics, **one_step_metrics}

    def _sample_flow_actions(
        self, obs: jnp.ndarray, goals: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        ts = self.train_states

        def step(i, actions):
            t = jnp.full((obs.shape[0], 1), i / self.cfg.flow_steps)
            vels = ts["n_step_actor"].apply_fn(
                {"params": ts["n_step_actor"].params},
                obs, goals, actions=actions, times=t,
            )
            return actions + vels / self.cfg.flow_steps

        actions = jax.lax.fori_loop(0, self.cfg.flow_steps, step, noise)
        return jnp.clip(actions, -1, 1)

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self, obs: jnp.ndarray, goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        noise_shape = obs.shape[:-1] + (self.cfg.action_dim,)
        noise = jax.random.normal(rng, noise_shape)
        actions = self.train_states["one_step_actor"].apply_fn(
            {"params": self.train_states["one_step_actor"].params},
            obs, goal, actions=noise,
        )
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        rng: jax.random.PRNGKey,
        obs_dim: tuple[int, ...],
        action_dim: int,
        cfg: ml_collections.ConfigDict,
    ):
        cfg['action_dim'] = action_dim

        obs_shape = tuple(obs_dim)
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_goal = jnp.zeros((1, *obs_shape))
        dummy_action = jnp.zeros((1, action_dim))
        dummy_times = jnp.zeros((1, 1))

        encoder = make_observation_encoder(cfg, dummy_obs)
        rng, onestep_actor_key, nstep_actor_key, critic_key = jax.random.split(rng, 4)

        networks = {
            "one_step_actor": ActorVectorFieldNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                encoder=encoder,
            ),
            "n_step_actor": ActorVectorFieldNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                encoder=encoder,
            ),
            "critic": QNetwork(
                hidden_dims=cfg.critic_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.critic_ensemble_size,
                encoder=encoder,
            ),
        }

        critic_params = networks["critic"].init(critic_key, dummy_obs, dummy_goal, dummy_action)["params"]
        params = {
            "one_step_actor": networks["one_step_actor"].init(
                onestep_actor_key, dummy_obs, dummy_goal, actions=dummy_action
            )["params"],
            "n_step_actor": networks["n_step_actor"].init(
                nstep_actor_key, dummy_obs, dummy_goal, actions=dummy_action, times=dummy_times
            )["params"],
            "critic": critic_params,
            "target_critic": critic_params,
        }

        train_states = {
            "one_step_actor": TrainState.create(
                apply_fn=networks["one_step_actor"].apply,
                params=params["one_step_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "n_step_actor": TrainState.create(
                apply_fn=networks["n_step_actor"].apply,
                params=params["n_step_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "critic": TrainState.create(
                apply_fn=networks["critic"].apply,
                params=params["critic"],
                tx=optax.adam(cfg.value_lr),
            ),
            "target_critic": TrainState.create(
                apply_fn=networks["critic"].apply,
                params=params["target_critic"],
                tx=optax.set_to_zero(),
            ),
        }
        return cls(rng=rng, train_states=train_states, cfg=cfg)


def get_default_config():
    return ml_collections.ConfigDict(
        dict(
            # network
            actor_hidden_dims=(512, 512, 512),
            critic_hidden_dims=(512, 512, 512),
            activations=nn.gelu,
            kernel_init=nn.initializers.orthogonal(),
            critic_ensemble_size=2,
            # training
            actor_lr=3e-4,
            value_lr=3e-4,
            alpha=3.0,
            discount=0.99,
            tau=0.005,
            flow_steps=10,
            action_dim=0,
            # encoder
            encoder="drqv2",
            encoder_feature_dim=512,
            encoder_num_features=32,
        )
    )
