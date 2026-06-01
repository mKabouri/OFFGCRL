import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from models.networks import ActorNetwork, BilinearCriticNetwork, Identity


class CRLAgent(flax.struct.PyTreeNode):
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

    def _compute_contrastive_loss(
        self,
        batch: Any,
        critic_params: Any,
        encoder_params: Any | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        batch_size = batch["observations"].shape[0]
        observations, goals = self.encode_pair(
            batch["observations"],
            batch["goals"],
            encoder_params,
        )
        q, phi, psi = self.train_states["critic"].apply_fn(
            {"params": critic_params},
            observations,
            goals,
            batch["actions"],
        )
        if phi.ndim == 2:
            phi = phi[jnp.newaxis]
            psi = psi[jnp.newaxis]

        logits = jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])
        labels = jnp.eye(batch_size)
        loss = jax.vmap(
            lambda member_logits: optax.sigmoid_binary_cross_entropy(logits=member_logits, labels=labels),
            in_axes=-1,
            out_axes=-1,
        )(logits).mean()

        q_exp = jnp.exp(q)
        mean_logits = logits.mean(axis=-1)
        correct = jnp.argmax(mean_logits, axis=1) == jnp.argmax(labels, axis=1)
        logits_pos = (mean_logits * labels).sum() / labels.sum()
        logits_neg = (mean_logits * (1 - labels)).sum() / (1 - labels).sum()

        return loss, {
            "critic_loss": loss,
            "critic_q_mean": q_exp.mean(),
            "critic_q_max": q_exp.max(),
            "critic_q_min": q_exp.min(),
            "critic_binary_accuracy": ((mean_logits > 0) == labels).mean(),
            "critic_categorical_accuracy": correct.mean(),
            "critic_logits_pos": logits_pos,
            "critic_logits_neg": logits_neg,
            "critic_logits": mean_logits.mean(),
        }

    def update_value(self, batch: Any):
        def loss_fn(params):
            return self._compute_contrastive_loss(
                batch,
                params["critic"],
                params["encoder"],
            )

        params = {
            "critic": self.train_states["critic"].params,
            "encoder": self.train_states["encoder"].params,
        }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        train_states = dict(self.train_states)
        train_states["critic"] = self.train_states["critic"].apply_gradients(grads=grads["critic"])
        train_states["encoder"] = self.train_states["encoder"].apply_gradients(grads=grads["encoder"])
        return self.replace(train_states=train_states), metrics

    def update_actor(self, batch: Any):
        observations, goals = self.encode_pair(
            batch["observations"],
            batch["goals"],
            self.train_states["encoder"].params,
        )
        observations = jax.lax.stop_gradient(observations)
        goals = jax.lax.stop_gradient(goals)

        def loss_fn(params):
            dist = self.train_states["actor"].apply_fn({"params": params["actor"]}, observations, goals)
            policy_actions = jnp.clip(dist.mode(), -1, 1)
            critic_q, _, _ = self.train_states["critic"].apply_fn(
                {"params": self.train_states["critic"].params},
                observations,
                goals,
                policy_actions,
            )
            if critic_q.ndim == 2:
                critic_q = critic_q.min(axis=0)
            q_loss = -critic_q.mean() / jax.lax.stop_gradient(jnp.abs(critic_q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(self.cfg.alpha * log_prob).mean()
            actor_loss = q_loss + bc_loss
            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_q_loss": q_loss,
                "actor_bc_loss": bc_loss,
                "actor_q_mean": critic_q.mean(),
                "actor_q_abs_mean": jnp.abs(critic_q).mean(),
                "actor_bc_log_prob": log_prob.mean(),
                "actor_mse": ((dist.mode() - batch["actions"]) ** 2).mean(),
                "actor_std": dist.stddev().mean(),
            }

        params = {
            "actor": self.train_states["actor"].params,
        }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        train_states = dict(self.train_states)
        train_states["actor"] = self.train_states["actor"].apply_gradients(grads=grads["actor"])
        return self.replace(train_states=train_states), metrics

    @jax.jit
    def update(self, batch: Any):
        agent, critic_metrics = self.update_value(batch)
        agent, actor_metrics = agent.update_actor(batch)
        metrics = {**critic_metrics, **actor_metrics}
        return agent, metrics

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self, obs: jnp.ndarray, goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        rng, key_l = jax.random.split(rng)
        obs, goal = self.encode_pair(obs, goal)
        dist = self.train_states["actor"].apply_fn({"params": self.train_states["actor"].params}, obs, goal)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample(seed=key_l)
        return jnp.clip(action, -1, 1)

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
        dummy_action = jnp.zeros((1, action_dim))

        if encoders is None:
            encoders = {"encoder": Identity()}

        rng, actor_key, critic_key, encoder_key = jax.random.split(rng, 4)
        train_states = {}

        encoder = encoders["encoder"]
        encoder_params = encoder.init(encoder_key, dummy_obs).get("params", {})
        dummy_obs = encoder.apply({"params": encoder_params}, dummy_obs)
        dummy_goal = encoder.apply({"params": encoder_params}, dummy_goal)
        train_states["encoder"] = TrainState.create(
            apply_fn=encoder.apply,
            params=encoder_params,
            tx=optax.adam(cfg.value_lr),
        )

        networks = {
            "actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                const_std=cfg.const_std,
            ),
            "critic": BilinearCriticNetwork(
                hidden_dims=cfg.critic_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.critic_ensemble_size,
                latent_dim=cfg.latent_dim,
            ),
        }

        train_states["actor"] = TrainState.create(
            apply_fn=networks["actor"].apply,
            params=networks["actor"].init(actor_key, dummy_obs, dummy_goal)["params"],
            tx=optax.adam(cfg.actor_lr),
        )
        train_states["critic"] = TrainState.create(
            apply_fn=networks["critic"].apply,
            params=networks["critic"].init(critic_key, dummy_obs, dummy_goal, dummy_action)["params"],
            tx=optax.adam(cfg.value_lr),
        )
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
            latent_dim=512,
            alpha=0.1,
            const_std=True,
        )
    )
