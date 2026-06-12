import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from models.networks import ActorNetwork, BilinearCriticNetwork
from models.encoders import make_observation_encoder


class TDINFONCEAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Any
    cfg: ml_collections.ConfigDict = flax.struct.field(pytree_node=False)

    def _compute_contrastive_loss(
        self, batch: Any, critic_params: Any
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        batch_size = batch["observations"].shape[0]
        gamma = self.cfg.discount

        _, phi, psi_next = self.train_states["critic"].apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["next_observations"],
            batch["actions"],
        )

        _, _, psi_goals = self.train_states["critic"].apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["goals"],
            batch["actions"],
        )

        if phi.ndim == 2:
            phi = phi[jnp.newaxis]
            psi_next = psi_next[jnp.newaxis]
            psi_goals = psi_goals[jnp.newaxis]

        d = phi.shape[-1]
        F_next = jnp.einsum("eik,ejk->ije", phi, psi_next) / jnp.sqrt(d)  # [N,N,E]
        F_future = jnp.einsum("eik,ejk->ije", phi, psi_goals) / jnp.sqrt(d)  # [N,N,E]

        next_dist = self.train_states["actor"].apply_fn(
            {"params": self.train_states["actor"].params},
            batch["next_observations"],
            batch["goals"],
        )
        next_actions = jnp.clip(next_dist.mode(), -1, 1)

        _, phi_tgt, psi_tgt = self.train_states["target_critic"].apply_fn(
            {"params": self.train_states["target_critic"].params},
            batch["next_observations"],
            batch["goals"],
            next_actions,
        )
        if phi_tgt.ndim == 2:
            phi_tgt = phi_tgt[jnp.newaxis]
            psi_tgt = psi_tgt[jnp.newaxis]

        F_w = jnp.einsum("eik,ejk->ije", phi_tgt, psi_tgt) / jnp.sqrt(phi_tgt.shape[-1])
        W = jax.lax.stop_gradient(
            batch_size * jax.nn.softmax(F_w.mean(axis=-1), axis=-1)
        )  # [N, N]

        log_p_next = jax.nn.log_softmax(F_next, axis=1)  # [N, N, E]
        loss_next = -log_p_next[jnp.arange(batch_size), jnp.arange(batch_size), :].mean()

        log_p_future = jax.nn.log_softmax(F_future, axis=1)  # [N, N, E]
        loss_future = -(W[..., jnp.newaxis] * log_p_future).sum(axis=1).mean()

        loss = (1 - gamma) * loss_next + gamma * loss_future

        mean_next = F_next.mean(axis=-1)  # [N, N]
        labels = jnp.eye(batch_size)
        correct = jnp.argmax(mean_next, axis=1) == jnp.arange(batch_size)
        logits_pos = (mean_next * labels).sum() / labels.sum()
        logits_neg = (mean_next * (1 - labels)).sum() / (1 - labels).sum()
        # Q value: diagonal of F_future = f(s_t, a_t, goal)
        q_diag = F_future[jnp.arange(batch_size), jnp.arange(batch_size), :]  # [N, E]

        return loss, {
            "critic_loss": loss,
            "critic_next_loss": loss_next,
            "critic_future_loss": loss_future,
            "critic_q_mean": jnp.exp(q_diag).mean(),
            "critic_q_max": jnp.exp(q_diag).max(),
            "critic_q_min": jnp.exp(q_diag).min(),
            "critic_categorical_accuracy": correct.mean(),
            "critic_logits_pos": logits_pos,
            "critic_logits_neg": logits_neg,
            "critic_logits": mean_next.mean(),
        }


    def update_value(self, batch: Any):
        def loss_fn(critic_params):
            return self._compute_contrastive_loss(batch, critic_params)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.train_states["critic"].params
        )
        train_states = dict(self.train_states)
        train_states["critic"] = self.train_states["critic"].apply_gradients(grads=grads)
        return self.replace(train_states=train_states), metrics

    def update_actor(self, batch: Any):
        def loss_fn(actor_params):
            dist = self.train_states["actor"].apply_fn(
                {"params": actor_params}, batch["observations"], batch["goals"]
            )
            policy_actions = jnp.clip(dist.mode(), -1, 1)

            _, phi_goal, psi_goals = self.train_states["critic"].apply_fn(
                {"params": self.train_states["critic"].params},
                batch["observations"],
                batch["goals"],
                policy_actions,
            )
            if phi_goal.ndim == 2:
                phi_goal  = phi_goal[jnp.newaxis]
                psi_goals = psi_goals[jnp.newaxis]

            batch_size = batch["observations"].shape[0]
            d = phi_goal.shape[-1]
            F_goal = jnp.einsum("eik,ejk->ije", phi_goal, psi_goals) / jnp.sqrt(d)  # [N,N,E]

            log_p_goal = jax.nn.log_softmax(F_goal, axis=1)  # [N, N, E]
            goal_loss = -log_p_goal[jnp.arange(batch_size), jnp.arange(batch_size), :].mean()

            mse_bc = ((policy_actions - batch["actions"]) ** 2).mean()

            alpha = self.cfg.alpha
            actor_loss = (1 - alpha) * goal_loss + alpha * mse_bc

            q_diag = F_goal[jnp.arange(batch_size), jnp.arange(batch_size), :]
            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_goal_loss": goal_loss,
                "actor_bc_loss": mse_bc,
                "actor_q_mean": jnp.exp(q_diag).mean(),
                "actor_mse": ((dist.mode() - batch["actions"]) ** 2).mean(),
                "actor_std": dist.stddev().mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.train_states["actor"].params
        )
        train_states = dict(self.train_states)
        train_states["actor"] = self.train_states["actor"].apply_gradients(grads=grads)
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
        agent, critic_metrics = self.update_value(batch)
        agent, actor_metrics = agent.update_actor(batch)
        agent = agent.soft_update_target_critic()
        return agent, {**critic_metrics, **actor_metrics}

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self, obs: jnp.ndarray, goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        rng, key_l = jax.random.split(rng)
        dist = self.train_states["actor"].apply_fn(
            {"params": self.train_states["actor"].params}, obs, goal
        )
        action = dist.mode() if deterministic else dist.sample(seed=key_l)
        return jnp.clip(action, -1, 1)

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
        dummy_action = jnp.zeros((1, action_dim))

        encoder = make_observation_encoder(cfg, dummy_obs)

        rng, actor_key, critic_key = jax.random.split(rng, 3)

        critic_network = BilinearCriticNetwork(
            hidden_dims=cfg.critic_hidden_dims,
            activations=cfg.activations,
            kernel_init=cfg.kernel_init,
            ensemble_size=cfg.critic_ensemble_size,
            latent_dim=cfg.latent_dim,
            encoder=encoder,
        )
        critic_params = critic_network.init(critic_key, dummy_obs, dummy_goal, dummy_action)["params"]

        networks = {
            "actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                const_std=cfg.const_std,
                encoder=encoder,
            ),
            "critic": critic_network,
            "target_critic": critic_network,
        }

        train_states = {
            "actor": TrainState.create(
                apply_fn=networks["actor"].apply,
                params=networks["actor"].init(actor_key, dummy_obs, dummy_goal)["params"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "critic": TrainState.create(
                apply_fn=networks["critic"].apply,
                params=critic_params,
                tx=optax.adam(cfg.value_lr),
            ),
            "target_critic": TrainState.create(
                apply_fn=networks["target_critic"].apply,
                params=critic_params,
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
            latent_dim=512,
            alpha=0.1,
            discount=0.99,
            tau=0.005,
            const_std=True,
            # encoder (can be override by --encoder args)
            encoder="drqv2",
            encoder_feature_dim=512,
            encoder_num_features=32,
        )
    )
