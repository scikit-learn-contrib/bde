---
title: 'bde: A Python Package for Bayesian Deep Ensembles via MILE'
tags:
  - Python
  - machine learning
  - MCMC
  - Bayesian deep learning
  - uncertainty quantification
authors:
  - name: Vyron Arvanitis
    orcid: 0009-0001-2290-5084
    equal-contrib: true
    affiliation: 1
    email: vyronas.arvanitis@gmail.com
  - name: Angelos Aslanidis
    orcid: 0009-0009-6699-2691
    equal-contrib: true
    affiliation: 1
    email: a.aslanidis@campus.lmu.de
  - name: Emanuel Sommer
    orcid: 0000-0002-1606-7547
    equal-contrib: true
    corresponding: true
    affiliation: "2, 3"
    email: emanuel.sommer@stat.uni-muenchen.de
  - name: David Rügamer
    orcid: 0000-0002-8772-9202
    affiliation: 3
affiliations:
 - name: Faculty of Physics, LMU Munich, Munich, Germany
   index: 1
 - name: Department of Statistics, LMU Munich, Munich, Germany
   index: 2
 - name: Munich Center for Machine Learning, Munich, Germany
   index: 3
date: 10 December 2025
bibliography: paper.bib

---

# Summary

`bde` is a Python package designed to bring state-of-the-art sampling-based Bayesian Deep Learning (BDL) to practitioners and researchers. The package combines the speed and high-performance capabilities of JAX and `blackjax` [@jax2018github; @cabezas2024blackjax] with the user-friendly API of scikit-learn [@scikit-learn]. It specifically targets tabular supervised learning tasks, including distributional regression and (multi-class) classification, providing a seamless interface for Bayesian Deep Ensembles (BDEs) [@sommer2024connecting] via **Microcanonical Langevin Ensembles (MILE)** [@sommer2025mile].

The workflow of `bde` implements the robust two-stage BDE inference process of MILE. First, it optimizes `n_members` many (usually 8) independent, flexibly configurable feed-forward neural networks using regularized empirical risk minimization (with the negative log-likelihood as loss) via the AdamW optimizer [@loshchilov2018decoupled]. Second, it transitions to a sampling phase using Microcanonical Langevin Monte Carlo [@robnik2023microcanonical; @robnik2024fluctuation], enhanced with a tuning phase adapted for Bayesian Neural Networks [@sommer2025mile]. In essence optimization finds diverse high-likelihood modes; sampling explores local posterior structure. This process generates an ensemble of samples (models) that constitute an implicit posterior approximation.

**Key Software Design Feature.**
Because optimization and sampling across ensemble members are independent, bde exploits JAX’s parallelization and just-in-time compilation to scale efficiently across CPUs, GPUs, and TPUs. Given new test data, the package approximates the posterior predictive, enabling point predictions, credible intervals, coverage estimates, and other uncertainty metrics through a unified interface.

# Statement of Need and State of the Field

Reliable uncertainty quantification (UQ) is increasingly viewed as a critical component of modern machine learning systems, and Bayesian Deep Learning provides a principled framework for achieving it [@papamarkou2024position]. While several libraries support optimization-based approaches such as variational inference or classical Bayesian modeling, accessible tools for sampling-based inference in Bayesian neural networks remain scarce. Existing probabilistic programming frameworks offer general-purpose MCMC but require substantial manual configuration to achieve competitive performance on neural network models.

`bde` addresses this gap by providing the first user-friendly implementation of MILE-a hybrid sampling technique shown to deliver strong predictive accuracy and calibrated uncertainty for Bayesian neural networks [@sommer2025mile]. By providing full scikit-learn compatibility, the package enables seamless integration into existing machine learning workflows, allowing users to obtain principled Bayesian uncertainty estimates without specialized knowledge of MCMC dynamics, initialization strategies, or JAX internals.

Through automated orchestration of optimization, sampling, parallelization, and predictive inference, `bde` offers a fast, reproducible, and practical solution for applying sampling-based BDL methods to tabular supervised learning tasks.

**Research Impact.**
`bde` bridges the gap between high-performance MCMC research and practical data science. By providing a curated implementation of MILE for tabular data, it enables researchers and practitioners alike to easily apply Bayesian Deep Ensembles. Its inclusion in the `scikit-learn-contrib` organization ensures adherence to rigorous software standards and API consistency, making it a trusted, community-ready tool for reproducible uncertainty quantification in tabular machine learning.

# Usage Example
The following example illustrates a regression task with uncertainty quantification using `bde` in only a few lines of code. Training inputs are assumed to be preprocessed (e.g., normalized). The workflow consists of
(i) specifying the ensemble model and sampling hyperparameters, (ii) fitting the model, and
(iii) obtaining posterior predictive quantities, including predictive moments, credible intervals, and raw (non-aggregated) ensemble outputs.

```python
from bde import BdeRegressor

regressor = BdeRegressor(
        n_members=8,
        hidden_layers=[16, 16],
        epochs=1000,
        validation_split=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        patience=20,
        warmup_steps=5000,
        n_samples=200,
        n_thinning=10,
        desired_energy_var_start=0.5, # key MILE hyperparameter
        desired_energy_var_end=0.1 # key MILE hyperparameter
)

regressor.fit(x=X_train, y=y_train)

means, sigmas = regressor.predict(X_test, mean_and_std=True)
means, intervals = regressor.predict(X_test, credible_intervals=[0.1, 0.9])
raw = regressor.predict(X_test, raw=True)
```

Classification follows analogously using `BdeClassifier`.

# Benchmark for Regression

<h3 style="margin-left: 24%;">Airfoil dataset
</h3>



|               | RMSE                | NLL (distr. regr.)   | NLL (mean regr.)     | Runtime (s) |
| ------------- | ------------------- | -------------------- | -------------------- | ----------- |
| Linear Model  | 0.6598 ± 0.0000 | -                    | 1.0032 ± 0.0000  | 0.0009 ± 0.0004  |
| Random Forest | 0.2560 ± 0.0015 | -                    | 0.0567 ± 0.0057 | 1.4610 ± 0.0127    |
| XGBoost       | 0.2025 ± 0.0055 | -                    | -0.1782 ± 0.0269 | 1.2466 ± 0.0305   |
| CatBoost      | 0.2393 ± 0.0036 | 0.1479 ± 0.1414   | -0.0109 ± 0.0152 | 0.8474 ± 0.0246   |
| TabPFN (V2)   | 0.1359 ± 0.0028 | -0.9338 ± 0.0195  | -0.5769 ± 0.0205 | 9.3150 ± 0.7825    |
| BDE($10, 16\times4$)       | **0.1215 ± 0.0042** | **-0.9126 ± 0.0131**  | **-0.6888 ± 0.0347** | 240.8765 ± 4.4666  |

<h3 style="margin-left: 23%;">Bikesharing dataset
</h3>

|               | RMSE               | NLL (distr. regr.) | NLL (mean regr.)  | Runtime (s) |
| ------------- | -------------------| -------------------- | -------------------- | ----------- |
| Linear Model  | 0.7796 ± 0.0000    | -                  | 1.1700 ± 0.000  | 0.0035 ± 0.0000 |
| Random Forest | 0.2440 ± 0.002     | -                   | 0.0085 ± 0.0009  | 6.8065 ±  0.0009|
| XGBoost       | 0.2143 ±  0.0010   | -                   | -0.1215 ± 0.0049 | 1.8980 ± 0.0049 |
| CatBoost      | 0.2652 ±  0.0021   | -0.3737 ± 0.0229      | 0.0918 ±  0.0080 | 2.5237 ± 0.0080 |
| TabPFN (V2)   | **0.2103 ± 0.0008**| -0.6856 ±  0.0063    | **-0.1400 ± 0.0041** | 1245.9870 ±  0.0041|
| BDE($10, 16\times4$)       | 0.2261 ± 0.0016    | **-0.7315 ± 0.0098**   | -0.0679 ± 0.0071  | 2555.0797 ± 0.0071 |


*Caption: Mean performance (RMSE), uncertainty-quantification metrics (NLL for distributional and mean regression), and total runtime (training + prediction), reported as mean ± standard deviation over 5 independent runs (different random seeds)*.

For all the models, 10 CPU cores were allocated and no additional tunning was performed. For **BDE**, we used $10$ MLP members (4 layers with 16 neurons) and we drew $100$ posterior samples for prediction (via $n_{samples} =1000$ using a thinning factor of 10 ).

\*default

# AI usage

Generative AI was used for smart code autocompletion via GitHub Copilot, using the Claude Sonnet 3.7 and Claude Sonnet 4 models. Its use was limited to local, line- or block-level completion during software development. Further Codex was used to assist in the generation of the first draft of the package documentation and for few refactors ensuring the compatibility with the `scikit-learn` API. No AI tools were used for ideation, architectural or design decisions, code review or testing strategy. All generated suggestions were critically reviewed, modified where necessary, and fully validated by the authors, who retain complete responsibility for the correctness, originality, licensing compliance, and ethical integrity of all materials.

# References
