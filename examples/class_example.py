from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from bde._bdecore import _BdeCore
from bde.task import TaskType
from bde.loss.loss import CategoricalCrossEntropy

# Load data
iris = load_iris()
X = iris.data.astype("float32")
y = iris.target.astype("int32")  # 0, 1, 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to JAX
Xtr, Xte = jnp.array(X_train), jnp.array(X_test)
ytr, yte = jnp.array(y_train), jnp.array(y_test)

# Define network: 4 inputs, hidden, 3 outputs (classes)
sizes = [4, 16, 16, 3]

bde = _BdeCore(
    n_members=5,
    sizes=sizes,
    seed=0,
    task=TaskType.CLASSIFICATION,
    loss=CategoricalCrossEntropy(),
    activation="relu"
)

bde.train(
    X=Xtr,
    y=ytr,
    epochs=50,
    lr=1e-3,
    warmup_steps=200,
    n_samples=50,
    n_thinning=5,
)

probs, preds = bde.evaluate(Xte)
print("Predicted class probabilities:\n", probs)
print("Predicted class labels:\n", preds)
print("True labels:\n", yte)
