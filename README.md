DAGMA-NET: Algorithm Unrolling for DAG Structure Learning
By Rhythm Sachdeva, Sundeep Prabhakar Chepuri, Antonio G. Marques, and Gonzalo Mateos

Claude AI was used to assist in coding.


DAGMA-Net — latest code, tests, and results
=============================================

Files
-----
dagma_net.py       the full pipeline (data, DAGMA baseline, DAGMA-Net, training, plots)
test_dagma_net.py  pytest suite (16 tests, all passing); `test_training_reduces_loss` is the gate
requirements.txt   dependencies (torch, numpy, matplotlib, networkx, pytest)

Run with:
    python dagma_net.py --quick     # fast smoke test (d=6, K=20)
    python dagma_net.py             # full run (d=10, K=40); writes PNGs + run_summary.txt to ./out
    python dagma_net.py --precond   # same, with the per-layer preconditioner turned on
    pytest -q                       # run the test suite

Results
-------
results/            default config (K=40, d=10, precond OFF, 85 learnable params)


Each results folder has:
  training_curves.png    train/val loss vs epoch, with early-stopping checkpoint marked
  learned_schedules.png  per-layer mu_k, barrier offset, learning rate eta_k, mean preconditioner
  convergence.png        distance-to-truth and objective vs iteration/layer, DAGMA vs DAGMA-Net
  shd_comparison.png     structure-recovery accuracy (SHD) on 30 held-out test DAGs
  run_summary.txt        config, learned scalars, SHD numbers, and test-time runtime for both methods

Headline numbers (default config, results/run_summary.txt):
  DAGMA          : SHD 3.13 +/- 3.10   (~9000 solver iterations, ~2.28s/DAG at test time)
  DAGMA-Net (K=40): SHD 1.90 +/- 2.36   (40 unrolled layers,     ~0.10s/DAG at test time, ~22x faster)


The vanilla-DAGMA baseline here is a faithful port of the reference implementation
(kevinsbello/dagma, src/dagma/linear.py): same score/acyclicity-gradient math, the same
per-outer-iteration fixed `s` schedule with backtracking, and the same default Adam
hyperparameters (beta_1=0.99, lr=3e-4, mu_factor=0.1, lambda1=0.03).
