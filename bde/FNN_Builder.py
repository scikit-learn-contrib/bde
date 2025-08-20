# FNN_Builder.py
import jax
import jax.numpy as jnp
import optax


class FNN():
    'Builds a single FNN'
    def __init__(self, sizes):
        self.sizes = sizes
        self.params = None  # will hold initialized weights

    def init_mlp(self):
        sizes = self.sizes
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, len(sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        self.params = params
        return params


class FNN_Trainer():
    @staticmethod
    def mlp_forward(params, X):
        for (W, b) in params[:-1]:
            X = jnp.dot(X, W) + b
            X = jnp.tanh(X)          
        W, b = params[-1]  # Fixed indentation - this should be outside the loop
        return jnp.dot(X, W) + b     
    
    @staticmethod
    def mse_loss(params, X, y):
        pred = FNN_Trainer.mlp_forward(params, X)
        return jnp.mean((pred - y) ** 2)
    
    def create_train_step(self, optimizer):
        """Factory function to create a jitted train_step with the optimizer"""
        @jax.jit
        def train_step(params, opt_state, X, y):
            grads = jax.grad(self.mse_loss)(params, X, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        return train_step
        
    def fit(self, model, X, y, optimizer, epochs=100):
        if model.params is None:
            model.init_mlp()
        opt_state = optimizer.init(model.params)
        params = model.params
        
        # Create the jitted training step function
        train_step_fn = self.create_train_step(optimizer)
        
        for step in range(epochs):
            params, opt_state = train_step_fn(params, opt_state, X, y)
            if step % 200 == 0:
                loss = float(self.mse_loss(params, X, y))
                print(step, loss)
        model.params = params

def main():
    # generate True data for test purposes
    main_key = jax.random.PRNGKey(0)
    k_X, k_W, k_eps = jax.random.split(main_key, 3)
    X_true = jax.random.normal(k_X, (1024, 10))
    true_W = jax.random.normal(k_W, (10, 1))
    y_true = X_true @ true_W + 0.1 * jax.random.normal(k_eps, (1024, 1))

    sizes = [10, 64, 64, 1]

    model = FNN(sizes)
    trainer = FNN_Trainer()
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=0.01)
    
    trainer.fit(
        model=model,
        X=X_true,
        y=y_true,
        optimizer=optimizer,
        epochs=1000
    )

if __name__ == "__main__":
    main()