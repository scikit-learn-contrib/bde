import argparse
import os
import sys
from bde import BdeRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from sklearn.neural_network import MLPRegressor
from bde import BdeRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


# ------------- configuration of models -------------
def bde_regressor():
    return BdeRegressor()


def mlp_regressor():
    return MLPRegressor(hidden_layer_sizes=(2, 12))


def linear_regressor():
    return LinearRegression()


# ------------- configuration of data -------------
def config_data(data_file):
    data = pd.read_csv(data_file, sep=",", header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X = jnp.array(X)
    y = jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test


# ------------- main runner -------------
def runner_regression(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    if isinstance(model, BdeRegressor):
        mus, sigma = model.predict(X_test)
    else:
        mus = model.predict(X_test)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=mus)
    pass


# ------------- CLI arguments -------------
def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choises=["airfoil", "concrete"],
        required=True,
        help="Benchmark dataset.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bde", "linear", "mlp"],
        help="Models to run. Example: --models bde linear rf",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Runs use seed0 + i.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path for per-run results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print()
    # runner_regression()
