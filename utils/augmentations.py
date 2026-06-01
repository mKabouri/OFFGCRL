import functools

import jax
import jax.numpy as jnp


def is_image_array(x: jnp.ndarray) -> bool:
    return x.ndim == 4


def get_image_keys(batch: dict[str, jnp.ndarray]) -> tuple[str, ...]:
    image_keys = []
    for key, value in batch.items():
        if not is_image_array(value):
            continue
        image_keys.append(key)
    return tuple(image_keys)


def random_crop(image: jnp.ndarray, crop_from: jnp.ndarray, padding: int) -> jnp.ndarray:
    padded_image = jnp.pad(
        image,
        pad_width=((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )
    start = (crop_from[0], crop_from[1], 0)
    return jax.lax.dynamic_slice(padded_image, start, image.shape)


@functools.partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(images: jnp.ndarray, crop_from: jnp.ndarray, padding: int = 3) -> jnp.ndarray:
    return jax.vmap(random_crop, in_axes=(0, 0, None))(images, crop_from, padding)


def sample_crop_from(rng: jax.random.PRNGKey, batch_size: int, padding: int) -> jnp.ndarray:
    return jax.random.randint(
        rng,
        shape=(batch_size, 2),
        minval=0,
        maxval=2 * padding + 1,
    )


def random_shift_batch(
    batch: dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    padding: int = 3,
    probability: float = 1.0,
) -> dict[str, jnp.ndarray]:
    new_batch = dict(batch)
    crop_from = None
    image_keys = get_image_keys(batch)

    for key in image_keys:
        images = new_batch[key]
        if crop_from is None:
            crop_from = sample_crop_from(rng, images.shape[0], padding)

        shifted_images = batched_random_crop(images, crop_from, padding)
        new_batch[key] = shifted_images

    if probability < 1.0:
        use_aug = jax.random.uniform(rng, ()) < probability
        return jax.lax.cond(
            use_aug,
            lambda _: new_batch,
            lambda _: batch,
            operand=None,
        )

    return new_batch


def random_shift_image_batch(
    batch: dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    padding: int = 3,
    probability: float = 1.0,
) -> dict[str, jnp.ndarray]:
    image_keys = get_image_keys(batch)
    if len(image_keys) == 0:
        return batch

    return random_shift_batch(
        batch,
        rng,
        padding=padding,
        probability=probability,
    )
