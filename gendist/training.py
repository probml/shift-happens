import jax
import optax
import jax.numpy as jnp


def make_loss_func(model, X, y):
    def loss_fn(params):
        y_hat = model.apply(params, X)
        loss = optax.softmax_cross_entropy(y_hat, y).mean()
        return loss
    return loss_fn


class TrainingConfig:
    def __init__(self, model, processor, loss_generator):
        self.model = model
        self.processor = processor
        self.loss_generator = loss_generator
    
    @jax.jit
    def train_step(params, opt_state, X_batch, y_batch):
        loss_fn = self.loss_generator(self.model, X_batch, y_batch)
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss_val, params

    def get_batch_train_ixs(self, key, num_samples, batch_size):
        """
        Obtain the training indices to be used in an epoch of
        mini-batch optimisation.
        """
        steps_per_epoch = num_samples // batch_size
        batch_ixs = jax.random.permutation(key, num_samples)
        batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
        batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
        
        return batch_ixs


    def train_epoch(self, key, params, opt_step, X, y, batch_size, epoch):
        num_samples, *_ = X.shape
        batch_ixs = self.get_batch_train_ixs(key, num_samples, batch_size)
        
        for batch_ix in batch_ixs:
            X_batch = X[batch_ix, ...]
            y_batch = y[batch_ix, ...]
            loss, params = self.train_step(params, opt_step, X_batch, y_batch)
        
        return params, opt_step


    def train_model_config(self, key, X_train, y_train, config, tx, num_epochs):
        """
        Train a flax.linen model by transforming the data according to
        process_config.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random number generator key.
        model: flax.nn.Module
            Model to train.
        X_train: jnp.array(N, ...)
            Training data.
        y_train: jnp.array(N)
            Training target values
        processor: function
            Function that takes a single sample and returns a transformed sample.
        config: dict
            Dictionary containing the training configuration to be passed to
            the processor.
        tx: optax optimizer
            Optimizer for training the model.
        num_epochs: int
            Number of epochs to train the model.
        """
        X_train_proc = self.processor(X_train, **config)
        _, *input_shape = X_train_proc.shape

        batch = jnp.ones((1, *input_shape))
        params = self.model.init(key, batch)
        optimiser_state = tx.init(params)

        for e in range(num_epochs):
            print(f"@epoch {e+1:03}", end="\r")
            _, key = jax.random.split(key)
            params, opt_state = self.train_epoch(key, params, optimiser_state,
                                                 X_train_proc, y_train, batch_size, e)

        final_train_acc = (y_train == self.model.apply(params, X_train_ravel).argmax(axis=1)).mean().item()
        print(f"@{radius=:0.4f}, {final_train_acc=:0.4f}")
        
        return params, final_train_acc
