# DAGMA-NET: Algorithm Unrolling for DAG Structure Learning

By Rhythm Sachdeva, Antonio G. Marques, Sundeep Prabhakar Chepuri, and Gonzalo Mateos

## DAGMA-Net — code, tests, and results

This folder holds three experiments:

1. Train on ER1, test on ER1 (`test_synthetic.py --graph-level 1`)
2. Train on ER4, test on ER4 (`test_synthetic.py --graph-level 4`)
3. Train on ER (per-edge-probability), test on Sachs (`test_sachs.py`)

### Files

| File | Description |
|---|---|
| `dagma_net.py` | The core pipeline (DAGMA baseline, DAGMA-Net, training, metrics) plus two instance generators: <br>• `make_instances_erpedge` — ER graphs by per-edge probability (`random_dag_W`); used for the Sachs experiment's synthetic training data <br>• `make_instances_erlevel` — ER1/ER4-style graphs with an explicit expected-degree (`data_generation.simulate_dag`); used for the ER1/ER4 experiments |
| `data_generation.py` | `simulate_dag` / `simulate_parameter` / `simulate_linear_sem`, used by `make_instances_erlevel`. We thank the authors of the NOTEARS repo for making their code available. Part of our code is based on their implementation, specially the `utils.py` file. |
| `test_synthetic.py` | ER1 / ER4 experiment: imports `dagma_net`, generates data with `make_instances_erlevel`, trains and evaluates DAGMA-Net vs DAGMA on held-out ER1/ER4 test DAGs |
| `test_sachs.py` | Sachs experiment: imports `dagma_net`, trains DAGMA-Net on synthetic ER data (`make_instances_erpedge`, p=0.3), then evaluates it against the Sachs protein-signalling dataset (`sachs.data.txt`) using the 17-edge consensus benchmark |
| `sachs.data.txt` | Sachs protein-signalling flow cytometry dataset (n=853, d=11), tab-separated, used only by `test_sachs.py` |
| `requirements.txt` | Dependencies (`torch`, `numpy`, `matplotlib`, `networkx`, `igraph`, `pandas`) |

### Run with

```bash
python dagma_net.py --quick               # fast smoke test (d=6, K=20)
python dagma_net.py                       # full run (d=10, K=40); writes PNGs + run_summary.txt to ./out
python test_synthetic.py --graph-level 1  # ER1 experiment, writes to ER1_gauss/
python test_synthetic.py --graph-level 4  # ER4 experiment, writes to ER4_gauss/
python test_sachs.py                      # Sachs experiment, writes to results_sachs/
```

### Results

| Folder | Description |
|---|---|
| `ER1_gauss/` | ER1 experiment (train on ER1, test on ER1) |
| `ER4_gauss/` | ER4 experiment (train on ER4, test on ER4) |
| `results_sachs/` | Sachs experiment (train on ER, test on Sachs, n=853) |

Each results folder has:

| File | Description |
|---|---|
| `training_curves.png` | Train/val loss vs epoch, with early-stopping checkpoint marked |
| `learned_schedules.png` | Per-layer μ_k, barrier offset, learning rate η_k |
| `convergence.png` | Distance-to-truth and objective vs iteration/layer, DAGMA vs DAGMA-Net |
| `shd_comparison.png` | Structure-recovery accuracy (SHD) on the held-out test set (or Sachs) |
| `run_summary.txt` | Config, learned scalars, SHD numbers, and test-time runtime for both methods |

The vanilla-DAGMA baseline here is a faithful port of the reference implementation ([kevinsbello/dagma](https://github.com/kevinsbello/dagma), `src/dagma/linear.py`): same score/acyclicity-gradient math, the same per-outer-iteration fixed `s` schedule with backtracking, and the same default Adam hyperparameters (beta_1=0.99, lr=3e-4, mu_factor=0.1, lambda1=0.03).
