# StaMojo <!-- omit in toc -->

![icon](stamojo.png)

A statistical computing library for [Mojo](https://www.modular.com/mojo), inspired by `scipy.stats` and `statsmodels` in Python.

**[Repository on GitHub»](https://github.com/mojomath/stamojo)**　|　**[Discord channel»](https://discord.gg/3rGH87uZTk)**

- [Overview](#overview)
- [Status](#status)
- [Background](#background)
- [Installation](#installation)
- [Examples](#examples)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

StaMojo (Statistics + Mojo) brings comprehensive statistical computing to the Mojo ecosystem. The library is organized into two parts:

| Part                                          | Scope                                                                                   | Dependencies                                                                                                            |
| --------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Part I — Statistical Computing Foundation** | Special functions, distributions, descriptive statistics, hypothesis tests, correlation | Mojo stdlib only                                                                                                        |
| **Part II — Statistical Modeling**            | OLS, GLM, logistic regression, and model diagnostics                                    | [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) + [MatMojo](https://github.com/mojomath/matmojo)  |

Part I is **available now** with zero external dependencies. Part II will begin once the upstream linear-algebra ecosystem stabilizes on a compatible Mojo release.

### Why a separate library? <!-- omit in toc -->

In the Python ecosystem, `scipy` bundles statistics, optimization, signal processing, integration, interpolation, and more into one giant package. For Mojo, a modular approach is more appropriate:

| Python package                  | Mojo equivalent                                                            | Focus                             |
| ------------------------------- | -------------------------------------------------------------------------- | --------------------------------- |
| `numpy`                         | [**NuMojo**](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) | N-dimensional arrays, basic math  |
| `decimal` / `mpmath`            | [**DeciMojo**](https://github.com/mojomath/decimojo)                       | Arbitrary-precision arithmetic    |
| `numpy.linalg` / `scipy.linalg` | [**MatMojo**](https://github.com/mojomath/matmojo)                         | Linear algebra                    |
| `scipy.stats`                   | **StaMojo** (distributions + tests)                                        | Statistical distributions & tests |
| `statsmodels`                   | **StaMojo** (models)                                                       | Statistical models & econometrics |

Placing `scipy.stats`-like functionality and `statsmodels`-like regression in **one library** is intentional: regression models inherently depend on distribution functions (for p-values, confidence intervals, etc.), so co-locating them avoids circular dependencies, simplifies versioning, and provides a cohesive API.

## Status

**v0.1** — Part I is complete and ready for use. The current release provides:

| Category          | Functions                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Special functions | `gammainc`, `gammaincc`, `beta`, `lbeta`, `betainc`, `erfinv`, `ndtri`                          |
| Distributions     | `Normal`, `StudentT`, `ChiSquared`, `FDist` — each with PDF, log-PDF, CDF, SF, PPF, `rvs`       |
| Descriptive stats | `mean`, `variance`, `std`, `median`, `quantile`, `skewness`, `kurtosis`, `data_min`, `data_max` |
| Correlation       | `pearsonr`, `spearmanr`, `kendalltau` (with p-values)                                           |
| Hypothesis tests  | `ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2_gof`, `chi2_ind`, `ks_1samp`, `f_oneway`         |

All 30 functions are self-contained (Mojo stdlib only) and covered by unit tests validated against SciPy reference values.

> **What about Part II (statistical models)?**
> OLS regression and GLMs require matrix operations that depend on [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) and [MatMojo](https://github.com/mojomath/matmojo). These upstream libraries are still fast evolving. Part II will resume once the ecosystem catches up. See the [full roadmap](docs/roadmap.md) for details.

## Background

Due to my academic and professional background, I work extensively with hypothesis testing and regression models on a daily basis, and have been a long-time user of Stata and `statsmodels`. It has been two years since Mojo first appeared, and [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) now has its core functionality in place. Driven by my enthusiasm for Mojo, I felt it was time to start migrating some of my personal research projects to the Mojo ecosystem — and that is precisely how StaMojo was born.

The library is designed around two pillars:

1. **Part I — Statistical computing foundation** (self-contained) — special functions, probability distributions, descriptive statistics, hypothesis tests, and correlation.
2. **Part II — Statistical modeling** (depends on NuMojo and MatMojo) — OLS, GLM, logistic regression, and related diagnostics.

At the moment I am still building out the project scaffolding and solidifying the core functionality. Because Mojo has not yet reached v1.0, breaking changes are frequent across compiler releases, so **pull requests are not preferred at this time**. If you have any suggestions, questions, or feedback, please feel free to open an [issue](https://github.com/mojomath/stamojo/issues), start a [discussion](https://github.com/mojomath/stamojo/discussions), or reach out on our [Discord channel](https://discord.gg/3rGH87uZTk). Thank you for your understanding!

## Installation

StaMojo is available in the modular-community `https://repo.prefix.dev/modular-community` package repository. To access this repository, add it to your `channels` list in your `pixi.toml` file:

```toml
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]
```

Then, you can install StaMojo using any of these methods:

1. From the `pixi` CLI, run the command ```pixi add stamojo```. This fetches the latest version and makes it immediately available for import.

1. In the `mojoproject.toml` file of your project, add the following dependency:

    ```toml
    stamojo = "==0.1.0"
    ```

    Then run `pixi install` to download and install the package.

1. For the latest development version in the `main` branch, clone [this GitHub repository](https://github.com/mojomath/stamojo) and build the package locally using the command `pixi run package`.

    ```bash
    git clone https://github.com/mojomath/stamojo.git
    cd stamojo
    pixi install
    pixi run package
    ```

The following table summarizes the package versions and their corresponding Mojo versions:

| `stamojo` | `mojo`   | package manager |
| --------- | -------- | --------------- |
| v0.1.0    | ==0.26.1 | pixi            |

## Examples

The file [`examples/examples.mojo`](examples/examples.mojo) demonstrates the key APIs available in Part I. Run it with:

```bash
mojo run -I src examples/examples.mojo
```

```mojo
from stamojo.special import gammainc, gammaincc, beta, lbeta, betainc, erfinv, ndtri
from stamojo.distributions import Normal, StudentT, ChiSquared, FDist
from stamojo.stats import (
    mean, variance, std, median, quantile, skewness, kurtosis,
    pearsonr, spearmanr, kendalltau,
    ttest_1samp, ttest_ind, ttest_rel,
    chi2_gof, chi2_ind, ks_1samp, f_oneway,
)


fn main() raises:
    # --- Special functions ---------------------------------------------------
    print("gammainc(1, 2) =", gammainc(1.0, 2.0))           # 0.8646647167628346
    print("gammaincc(1, 2) =", gammaincc(1.0, 2.0))         # 0.13533528323716537
    print("beta(2, 3)     =", beta(2.0, 3.0))               # 0.08333333333323925
    print("betainc(2, 3, 0.5) =", betainc(2.0, 3.0, 0.5))   # 0.6875000000000885
    print("erfinv(0.5)    =", erfinv(0.5))                  # 0.4769362762044701
    print("ndtri(0.975)   =", ndtri(0.975))                 # 1.9599639845400543

    # --- Distributions -------------------------------------------------------
    var n = Normal(0.0, 1.0)
    print("Normal(0,1).pdf(0)   =", n.pdf(0.0))             # 0.3989422804014327
    print("Normal(0,1).cdf(1.96)=", n.cdf(1.96))            # 0.9750021048517795
    print("Normal(0,1).ppf(0.975)=", n.ppf(0.975))          # 1.9599639845400543
    print("Normal(0,1).sf(1.96) =", n.sf(1.96))             # 0.02499789514822043

    var t = StudentT(10.0)
    print("StudentT(10).cdf(2.0)=", t.cdf(2.0))             # 0.9633059826444078
    print("StudentT(10).ppf(0.975)=", t.ppf(0.975))         # 2.2281388540534057

    var c = ChiSquared(5.0)
    print("ChiSquared(5).cdf(11.07)=", c.cdf(11.07))        # 0.9499903814759155

    var f = FDist(5.0, 10.0)
    print("FDist(5,10).cdf(3.33)=", f.cdf(3.33))            # 0.9501687242532277

    # --- Descriptive statistics ----------------------------------------------
    var data: List[Float64] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print("mean    =", mean(data))              # 5.0
    print("variance=", variance(data, ddof=0))  # 4.0
    print("std     =", std(data, ddof=0))       # 2.0
    print("median  =", median(data))            # 4.5
    print("Q(0.25) =", quantile(data, 0.25))    # 4.0
    print("skewness=", skewness(data))          # 0.8184875533567997
    print("kurtosis=", kurtosis(data))          # 0.940625

    # --- Correlation ---------------------------------------------------------
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var y: List[Float64] = [2.1, 3.8, 6.0, 7.9, 10.1]
    var pr = pearsonr(x, y)
    print("pearsonr  r=", pr[0], " p=", pr[1])    # 0.9991718425080479, 2.8605484175113625e-05
    var sr = spearmanr(x, y)
    print("spearmanr ρ=", sr[0], " p=", sr[1])    # 1.0, 0.0
    var kt = kendalltau(x, y)
    print("kendalltau τ=", kt[0], " p=", kt[1])   # 1.0, 0.014305878435429659

    # --- Hypothesis tests ----------------------------------------------------
    var sample: List[Float64] = [5.1, 4.8, 5.3, 5.0, 4.9, 5.2]
    var res = ttest_1samp(sample, 5.0)
    print("ttest_1samp  t=", res[0], " p=", res[1])   # 0.654653670707975, 0.5416045608507769

    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var b: List[Float64] = [4.0, 5.0, 6.0, 7.0, 8.0]
    var res2 = ttest_ind(a, b)
    print("ttest_ind    t=", res2[0], " p=", res2[1])  # -3.0, 0.0170716812337895

    var obs: List[Float64] = [16.0, 18.0, 16.0, 14.0, 12.0, 14.0]
    var exp: List[Float64] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    var res3 = chi2_gof(obs, exp)
    print("chi2_gof   χ²=", res3[0], " p=", res3[1])  # 1.4666666666666666, 0.9168841203537823

    var g1: List[Float64] = [3.0, 4.0, 5.0]
    var g2: List[Float64] = [6.0, 7.0, 8.0]
    var g3: List[Float64] = [9.0, 10.0, 11.0]
    var groups = List[List[Float64]]()
    groups.append(g1^)
    groups.append(g2^)
    groups.append(g3^)
    var res4 = f_oneway(groups)
    print("f_oneway   F=", res4[0], " p=", res4[1])   # 27.0, 0.0010000000005315757
```

## Architecture

```txt
src/stamojo/
├── __init__.mojo              # Package root (re-exports distributions & stats)
├── prelude.mojo               # Convenient re-exports
├── special/                   # Special mathematical functions (cf. scipy.special)
│   ├── __init__.mojo
│   ├── _gamma.mojo            # gammainc, gammaincc
│   ├── _beta.mojo             # beta, lbeta, betainc
│   └── _erf.mojo              # erfinv, ndtri
├── distributions/             # Probability distributions
│   ├── __init__.mojo
│   ├── normal.mojo            # Normal (Gaussian) — PDF, logPDF, CDF, SF, PPF, rvs
│   ├── t.mojo                 # Student's t
│   ├── chi2.mojo              # Chi-squared
│   └── f.mojo                 # F-distribution
├── stats/                     # Descriptive stats & hypothesis tests
│   ├── __init__.mojo
│   ├── descriptive.mojo       # mean, variance, std, median, quantile, skewness, kurtosis
│   ├── correlation.mojo       # pearsonr, spearmanr, kendalltau
│   └── tests.mojo             # ttest_1samp, ttest_ind, ttest_rel, chi2_gof, chi2_ind, ks_1samp, f_oneway
└── models/                    # Statistical models (planned)
    ├── __init__.mojo
    └── ols.mojo               # Ordinary Least Squares (stub)
tests/
├── test_all.sh                # Run all test suites
├── test_special.mojo          # tests — special functions
├── test_distributions.mojo    # tests — Normal, t, χ², F
├── test_stats.mojo            # tests — descriptive statistics
└── test_hypothesis.mojo       # tests — hypothesis tests, correlation, ANOVA
```

## Roadmap

The project is organized into **Part I** (scipy.stats-equivalent, no external dependencies) and **Part II** (statsmodels-equivalent, requires NuMojo + MatMojo). Phases 0–2 are complete; see the [full roadmap](docs/roadmap.md) for all planned phases.

|             | Phase                                            | Status                    |
| ----------- | ------------------------------------------------ | ------------------------- |
| **Part I**  | Phase 0 — Special Functions                      | ✓                         |
|             | Phase 1 — Core Distributions & Descriptive Stats | ✓                         |
|             | Phase 2 — Hypothesis Testing & Correlation       | ✓                         |
|             | Phase 3 — Extended Distributions                 | planned                   |
|             | Phase 4 — Extended Tests & Utilities             | planned                   |
| **Part II** | Phase 5 — OLS Regression                         | awaiting NuMojo / MatMojo |
|             | Phase 6 — Generalized Linear Models              | awaiting NuMojo / MatMojo |
|             | Phase 7 — Extended Models                        | planned                   |
|             | Phase 8 — Advanced Topics                        | planned                   |

## License

This repository and its contributions are licensed under the Apache License v2.0.
