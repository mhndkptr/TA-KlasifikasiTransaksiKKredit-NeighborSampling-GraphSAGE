# EXP21E-v8 Mechanism Audit and Statistical Analysis

V8 is post-hoc only: no checkpoint, threshold, hyperparameter, or model weight was updated.

## Main results

| version   | window_id    | model       |   n_seeds |   validation_auprc_mean |   validation_auprc_std |   test_auprc_mean |   test_auprc_std |   f1_mean |   f1_std |   precision_mean |   recall_mean |   runtime_seconds_mean |
|:----------|:-------------|:------------|----------:|------------------------:|-----------------------:|------------------:|-----------------:|----------:|---------:|-----------------:|--------------:|-----------------------:|
| v5        | primary_2018 | importance  |         5 |                0.642827 |               0.002523 |          0.941093 |         0.008948 |  0.921056 | 0.007304 |         0.868457 |      0.980523 |            2385.486625 |
| v5        | primary_2018 | topology    |         5 |                0.643751 |               0.000679 |          0.953941 |         0.008620 |  0.908496 | 0.014520 |         0.845014 |      0.982423 |            2155.670725 |
| v5        | primary_2018 | uniform     |         5 |                0.643149 |               0.002360 |          0.941877 |         0.021432 |  0.922178 | 0.006467 |         0.869593 |      0.981948 |            2319.962722 |
| v7        | primary_2018 | tabular_mlp |         5 |                0.644944 |               0.001663 |          0.956840 |         0.003872 |  0.887077 | 0.023793 |         0.893480 |      0.882185 |             369.081302 |

## Rolling-window robustness

| window_id          | model      |   n_seeds |   validation_auprc_mean |   test_auprc_mean |   test_auprc_std |   f1_mean |   f1_std |
|:-------------------|:-----------|----------:|------------------------:|------------------:|-----------------:|----------:|---------:|
| rolling_minus_180d | importance |         3 |                0.992750 |          0.920991 |         0.002098 |  0.879038 | 0.004414 |
| rolling_minus_180d | topology   |         3 |                0.989002 |          0.918455 |         0.009750 |  0.872249 | 0.009420 |
| rolling_minus_180d | uniform    |         3 |                0.989019 |          0.924786 |         0.012607 |  0.867456 | 0.007863 |
| rolling_plus_180d  | importance |         3 |                0.999965 |          0.940012 |         0.008438 |  0.928059 | 0.008268 |
| rolling_plus_180d  | topology   |         3 |                0.999736 |          0.928854 |         0.014249 |  0.912504 | 0.004579 |
| rolling_plus_180d  | uniform    |         3 |                0.999961 |          0.933679 |         0.001356 |  0.921317 | 0.008120 |

## Paired daily-block bootstrap

The resampling unit is a calendar day. Effects are paired within seed against V5 Uniform.

### topology versus Uniform

- auprc: Δ=0.003154, 95% CI [-0.005958, 0.013929], P(Δ>0)=0.723.
- f1: Δ=0.000785, 95% CI [-0.010701, 0.014391], P(Δ>0)=0.502.

### importance versus Uniform

- auprc: Δ=0.001342, 95% CI [-0.007861, 0.011623], P(Δ>0)=0.587.
- f1: Δ=0.001501, 95% CI [-0.010613, 0.013847], P(Δ>0)=0.601.

### tabular_mlp versus Uniform

- auprc: Δ=-0.000023, 95% CI [-0.015790, 0.014599], P(Δ>0)=0.516.
- f1: Δ=-0.053414, 95% CI [-0.081569, -0.026952], P(Δ>0)=0.000.

## Mechanism audit

Empirical draws per destination group: 2000.

### transaction__rev_buys__user

- uniform: Spearman(weight, inclusion)=None.
- topology: Spearman(weight, inclusion)=0.7862876993217168.
- importance: Spearman(weight, inclusion)=0.8236631756687988.

### transaction__rev_sells__merchant

- uniform: Spearman(weight, inclusion)=None.
- topology: Spearman(weight, inclusion)=-0.023110415887389237.
- importance: Spearman(weight, inclusion)=0.20556942088869842.

## Embedding audit

| ('strategy', '')   |   ('centroid_distance', 'mean') |   ('centroid_distance', 'std') |   ('fisher_ratio', 'mean') |   ('fisher_ratio', 'std') |
|:-------------------|--------------------------------:|-------------------------------:|---------------------------:|--------------------------:|
| importance         |                        5.636348 |                       0.312593 |                   4.579382 |                  1.618111 |
| topology           |                        6.554016 |                       0.901678 |                   4.733089 |                  0.730922 |
| uniform            |                        5.617543 |                       0.800033 |                   4.119379 |                  1.464321 |

Checkpoint reproduction was enforced before representation statistics were accepted.

## Interpretation guardrail

A mechanism claim is supported only when sampling weights alter inclusion/neighborhood statistics and the corresponding representation or error statistic changes consistently. Performance association alone is not interpreted as causation.
