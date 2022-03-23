import jax
import optax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm.auto import tqdm


def make_cross_entropy_loss_func(model, X, y):
    """
    Make a loss function for a multi-output classifier, i.e.,
    model: R^M -> (y1, y2, ..., yK) = y_hat; where y_hat is a
    probability distribution over K classes

    Make a loss function for a multi-output model, i.e.,
    model: R^M -> R^K

    Parameters
    ----------
    model: Flax model
        Flax model that takes X and returns y_hat
    X: array(N, ...)
        N samples of arbitrary shape
    y: array(N, K)
        N samples of K-dimensional outputs
    """
    def loss_fn(params):
        y_hat = model.apply(params, X)
        loss = optax.softmax_cross_entropy(y_hat, y).mean()
        return loss
    return loss_fn


def make_multi_output_loss_func(model, X, y):
    """
    Make a loss function for a multi-output model, i.e.,
    model: R^M -> R^K

    Parameters
    ----------
    model: Flax model
        Flax model that takes X and returns y_hat
    X: array(N, ...)
        N samples of arbitrary shape
    y: array(N, K)
        N samples of K-dimensional outputs
    """
    def loss_fn(params):
        y_hat = model.apply(params, X)
        loss = jnp.linalg.norm(y - y_hat, axis=-1) ** 2
        return loss.mean()
    return loss_fn

class TrainingBase:
    """
    Class to train a neural network model that transforms the input data
    given a processor function.
    """
    def __init__(self, model, processor, loss_generator, tx):
        self.model = model
        self.processor = processor
        self.loss_generator = loss_generator
        self.tx = tx

    def fit(self, key, X_train, y_train, config, num_epochs, batch_size, evalfn=None):
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
        config: dict
            Dictionary containing the training configuration to be passed to
            the processor.
        num_epochs: int
            Number of epochs to train the model.
        """
        X_train_proc = self.processor(X_train, config)
        _, *input_shape = X_train_proc.shape

        batch = jnp.ones((1, *input_shape))
        params = self.model.init(key, batch)
        optimiser_state = self.tx.init(params)

        losses = []
        for e in tqdm(range(num_epochs), leave=False):
            _, key = jax.random.split(key)
            params, optimiser_state, avg_loss = self.train_epoch(key, params, optimiser_state,
                                                 X_train_proc, y_train, batch_size, e)
            losses.append(avg_loss)

        if evalfn is not None:
            yhat = self.model.apply(params, X_train_proc)
            metric = evalfn(y_train, yhat)

        training_output = {
            "losses": jnp.array(losses),
            "metric": metric,
            "params": params,
        }

        return training_output

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, X_batch, y_batch):
        loss_fn = self.loss_generator(self.model, X_batch, y_batch)
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = self.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return loss_val, params, opt_state

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
        
        epoch_loss = 0.0
        for batch_ix in batch_ixs:
            X_batch = X[batch_ix, ...]
            y_batch = y[batch_ix, ...]
            loss, params, opt_step = self.train_step(params, opt_step, X_batch, y_batch)
            epoch_loss += loss
        
        epoch_loss = epoch_loss / len(batch_ixs)
        return params, opt_step, epoch_loss


class TrainingSnapshot(TrainingBase):
    """
    Extension of Training base class that saves the model parameters
    every snapshot_interval epochs. For this class, it is better to consider
    an optimiser that fluctuates the learning rate.
    """
    def __init__(self, model, processor, loss_generator, tx, snapshot_interval):
        super().__init__(model, processor, loss_generator, tx)
        self.snapshot_interval = snapshot_interval

    def fit(self, key, X_train, y_train, config, num_epochs, batch_size, evalfn=None):
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
        config: dict
            Dictionary containing the training configuration to be passed to
            the processor.
        num_epochs: int
            Number of epochs to train the model.
        """
        X_train_proc = self.processor(X_train, config)
        _, *input_shape = X_train_proc.shape

        batch = jnp.ones((1, *input_shape))
        params = self.model.init(key, batch)
        optimiser_state = self.tx.init(params)

        losses = []
        params_hist = []
        metrics_hist = []
        for e in tqdm(range(num_epochs), leave=False):
            _, key = jax.random.split(key)

            # Store the parameters and evaluate the model on
            # the train set
            if (e+1) % self.snapshot_interval == 0:
                params_hist.append(params)
                if evalfn is not None:
                    yhat = self.model.apply(params, X_train_proc)
                    metric = evalfn(y_train, yhat)
                    metrics_hist.append(metric)


            params, optimiser_state, avg_loss = self.train_epoch(key, params, optimiser_state,
                                                 X_train_proc, y_train, batch_size, e)
            losses.append(avg_loss)

        training_output = {
            "params": params_hist,
            "losses": jnp.array(losses),
            "metrics": metrics_hist,
        }

        return training_output


class TrainingMeta(TrainingBase):
    """
    Training class of model parameters. We consider an input of the form NxMx..., and a target
    variable of the form KxMxW, wher
    * N: number of observations
    * M: number of transformations per observation
    * ...: Dimension specification of a single instance
    * K: number of samples per configuration. 
    * W: number of parameters per configuration
    """
    def __init__(self, model, loss_generator, tx):
        super().__init__(model, lambda x, _: x, loss_generator, tx)

    def fit(self, key, X_train, y_train, num_epochs, batch_size, leave_pb=True):
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
        num_epochs: int
            Number of epochs to train the model.
        batch_size: int
            Number of samples per batch.
        leave_pb: bool
            If True, the progress bar is left open.
        """
        _, *input_shape = X_train.shape

        key, key_params = jax.random.split(key)
        batch = jnp.ones((1, *input_shape))
        params = self.model.init(key_params, batch)
        optimiser_state = self.tx.init(params)

        losses = []
        for e in tqdm(range(num_epochs), leave=leave_pb):
            _, key = jax.random.split(key)
            params, optimiser_state, avg_loss = self.train_epoch(key, params, optimiser_state,
                                                                 X_train, y_train, batch_size, e)
            losses.append(avg_loss)

        training_output = {
            "params": params,
            "losses": jnp.array(losses),
        }

        return training_output

    
    def train_epoch(self, key, params, opt_step, X, y, batch_size, epoch):
        """
        Train an model considering an input of the form NxMx..., and a target
        variable of the form KxMxW.
        """
        num_samples, num_configs_X, *_ = X.shape
        num_cycles, num_configs_y, _ = y.shape
        if num_configs_X != num_configs_y:
            raise ValueError("The number of configurations in X and y must be the same.")
        num_configs = num_configs_X
        num_elements = num_samples * num_configs * num_cycles
        
        batch_ixs = self.get_batch_train_ixs(key, num_elements, batch_size)
        
        epoch_loss = 0.0
        for batch_ix in batch_ixs:
            X_batch = X[batch_ix % num_samples, batch_ix // (num_samples * num_cycles), ...]
            y_batch = y[(batch_ix // num_samples) % num_cycles, batch_ix // (num_samples * num_cycles), ...]
            loss, params, opt_step = self.train_step(params, opt_step, X_batch, y_batch)
            epoch_loss += loss
        
        epoch_loss = epoch_loss / len(batch_ixs)
        return params, opt_step, epoch_loss
