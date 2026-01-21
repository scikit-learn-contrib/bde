import argparse
import os
import sys


from bde import BdeRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from bde import BdeRegressor
from sklearn.linear_model import LinearRegression
import time
from pathlib import Path
from bde.loss.loss import GaussianNLL
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from catboost import CatBoostRegressor
import torch

torch.set_num_threads(10)
torch.set_num_interop_threads(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


# ------------- configuration of models -------------
def bde_regressor(seed) -> BdeRegressor:
    return BdeRegressor(
        n_members=10,
        hidden_layers=[16, 16, 16, 16],
        loss=GaussianNLL(),
        epochs=1200,
        validation_split=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_steps=50000,
        n_samples=1000,
        n_thinning=10,
        patience=20,
        seed=seed,
    )


def linear_regressor() -> LinearRegression:
    return LinearRegression()


def rand_forest(seed) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=1000,
        criterion="squared_error",
        random_state=seed,
        n_jobs=10,
    )


def xgb_reg(seed) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        tree_method="hist",
        n_estimators=2000,
        learning_rate=0.05,
        subsample=0.8,
        max_depth=6,
        random_state=seed,
        n_jobs=10,
    )


def catboost_reg(seed) -> CatBoostRegressor:
    return CatBoostRegressor(
        random_seed=seed,
        depth=6,
        iterations=1000,
        learning_rate=0.05,
        loss_function="RMSEWithUncertainty",
        allow_writing_files=False,
        verbose=False,
        thread_count=10,
    )


def tabpfn_reg(seed) -> TabPFNRegressor:
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, ignore_pretraining_limits=True, random_state=seed
    )


MODEL_FACTORY = {
    "bde": lambda seed: bde_regressor(seed),
    "linear": lambda seed: linear_regressor(),
    "rf": lambda seed: rand_forest(seed),
    "xgb": lambda seed: xgb_reg(seed),
    "catboost": lambda seed: catboost_reg(seed),
    "tabpfn": lambda seed: tabpfn_reg(seed),
}


# ------------- configuration of data -------------

DATA_DIR = Path(__file__).resolve().parents[1] / "bde" / "data"
DATASETS = {
    "airfoil": {"file": "airfoil.csv", "sep": ",", "header": 0},
    "concrete": {"file": "concrete.data", "sep": " ", "header": None},
    "bikesharing": {"file": "bike_sharing_dataset/hour.csv", "sep": ",", "header": 0},
}


def config_data(dataset, split_seed: int):
    spec = DATASETS[dataset]
    path = DATA_DIR / spec["file"]
    data = pd.read_csv(path, sep=spec["sep"], header=spec["header"])

    if dataset == "bikesharing":
        y = data["cnt"].to_numpy(dtype=float)

        X = data.drop(
            columns=["cnt", "casual", "registered", "instant", "dteday"]
        ).to_numpy(dtype=float)
    else:
        X = data.iloc[:, :-1].to_numpy(dtype=float)
        y = data.iloc[:, -1].to_numpy(dtype=float)

    X = jnp.array(X)
    y = jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test


# ------------- Store results -------------


def save_results(out_dir: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    path = out_dir / "results.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ------------- Negative Log Likelihood Computation -------------


def neg_log_likelihood(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-8)

    return 0.5 * (np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / sigma**2)


# ------------- main runner -------------

IQR_TO_STD = 1.3489795


def runner_regression(
    model, model_name, dataset, run_idx, seed, X_train, X_test, y_train, y_test
):
    logger.info(
        "Start: dataset=%s model=%s run=%d seed=%d", dataset, model_name, run_idx, seed
    )

    nll_default = np.nan
    sigma = None

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0  # time took to train

    if isinstance(model, BdeRegressor):
        mus, sigma = model.predict(X_test, mean_and_std=True)
    elif isinstance(model, TabPFNRegressor):
        out = model.predict(X_test, output_type="main", quantiles=[0.25, 0.75])
        mus = out["mean"]
        qs_25, qs_75 = out["quantiles"]
        sigma = (qs_75 - qs_25) / IQR_TO_STD
    elif isinstance(model, CatBoostRegressor) and hasattr(
        model, "virtual_ensembles_predict"
    ):
        tu = model.virtual_ensembles_predict(
            X_test,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=10,
            thread_count=10,
        )  # shape (n_samples, 3) for RMSEWithUncertainty: [mean, knowledge_unc, data_unc]

        mus = tu[:, 0]
        knowledge = tu[:, 1]  # epistemic uncertainty (model)
        data = tu[:, 2]  # aleatoric uncertainty (data)
        sigma = np.sqrt(np.maximum(knowledge + data, 1e-8))
    else:
        mus = model.predict(X_test)

    total_time = time.perf_counter() - t0  # time took to trian and predict

    rmse = root_mean_squared_error(y_true=y_test, y_pred=mus)

    nll_mean = float(
        np.mean(neg_log_likelihood(y_test, mus, rmse))
    )  # baseline with global sigma, using the rmse evaluated on the test set
    # this leakage is acceptable for benchmarking purposes as it is consistent across
    # all models and acts in favor of the baselines

    if sigma is not None:
        sigma = np.maximum(sigma, 1e-8)
        nll_default = float(np.mean(neg_log_likelihood(y_test, mus, sigma)))

    logger.info(
        (
            "Done:  dataset=%s model=%s run=%d mean=%.6f rmse=%.6f nll_default=%.6f"
            " nll_mean=%.6f  train_time=%.6f total_time=%.6f"
        ),
        dataset,
        model_name,
        run_idx,
        mus.mean(),
        rmse,
        nll_default,
        nll_mean,
        train_time,
        total_time,
    )
    return (
        float(mus.mean()),
        float(rmse),
        float(nll_default),
        float(nll_mean),
        float(train_time),
        float(total_time),
    )


# ------------- CLI arguments -------------
def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bde", "linear", "rf", "xgb", "catboost"],
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
        default="scripts/results",
        help="Output CSV path for per-run results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()
    X_train, X_test, y_train, y_test = config_data(args.dataset, split_seed=args.seed)

    rows = []
    for model_name in args.models:
        for run_idx in range(args.n_runs):
            seed = args.seed + run_idx
            model = MODEL_FACTORY[model_name](seed)
            mus, rmse, nll_default, nll_mean, train_time, total_time = (
                runner_regression(
                    model,
                    model_name,
                    args.dataset,
                    run_idx,
                    seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
            )

            row = {
                "dataset": args.dataset,
                "model": model_name,
                "run": run_idx,
                "mean": mus,
                "rmse": rmse,
                "nll_default": nll_default,
                "nll_mean": nll_mean,
                "train_time": train_time,
                "total_time": total_time,
            }
            rows.append(row)

        if args.out:
            save_results(args.out, rows)
