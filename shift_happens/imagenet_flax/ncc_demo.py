from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import train
import agents.nearest_centroid_classifier as agent
from environment import Environment

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    local_device_count = jax.local_device_count()

    ds_name, num_classes = 'mnist', 10
    batch_size, num_pulls = 256, 10
    T = 200

    trans_mat, rot_mat = train.init_trans_mat_and_rot_mat(key, num_classes=num_classes)
    env = Environment('mnist', class_trans_mat=trans_mat, rot_mat=rot_mat, batch_size=batch_size)
    inputs, labels = env.warmup(num_pulls=num_pulls)

    state = agent.init(inputs, labels)
    prior = jax.nn.log_softmax(jnp.ones((num_classes)))

    accuracies, results = [defaultdict(list)] * num_classes, []
    for t in range(T):
        batch = env.get_data()
        logits = agent.predict(state, batch['image'], prior)
        loss = train.cross_entropy_loss(logits, batch['label'], num_classes)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        accuracies[env.current_class][env.rot.item()].append(accuracy)
        results.append(accuracy)
        state = agent.update(state, batch['image'], env.current_class)

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(results)), results)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.savefig('./ncc_training.png')
    plt.show()

    '''fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    rotations = sorted(accuracies[0].keys())
    for i, ax in enumerate(axes.flatten()):
        for rot, acc in accuracies[i].items():
            ax.plot(np.arange(len(acc)), np.array(acc), 'o-', label=str(rot))
        ax.legend()
        ax.set_title(f'Class {i}')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig('./ncc_cls_acc.png')
    plt.show()'''
