import jax.numpy as jnp

from jax.nn import log_softmax

from collections import defaultdict


def get_accuracy(y, logits):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(y == preds)


def eval_step(env, state, predict_fn, prior=None, rotations=None):
    metrics = defaultdict(lambda: jnp.array([]))
    if prior is None:
        mean, *_ = state
        num_classes = len(mean)
        prior = log_softmax(jnp.ones((num_classes,)))

    if rotations is None:
        rotations = jnp.argwhere(env.rot_mat.sum(axis=0) > 0).flatten().tolist()

    for batch in env.test_data:
        input, label = jnp.array(batch['image']), jnp.array(batch['label'])
        if input.shape[-1] == 1:
            input = jnp.repeat(input, axis=-1, repeats=3)

        for degrees in rotations:
            if degrees:
                x = env.apply(input, degrees)
            else:
                x = input
            logits = predict_fn(state, x, prior)
            metrics[degrees] = jnp.append(metrics[degrees], get_accuracy(label, logits))

    return {k: jnp.mean(v) for k, v in metrics.items()}
