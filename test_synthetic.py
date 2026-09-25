"""
DAGMA-Net vs DAGMA on ER-graph synthetic data (ER1 / ER4), built on top of this
folder's dagma_net.py. Instances come from dagma_net.make_instances_erlevel,
which uses data_generation.simulate_dag with an explicit expected-degree
graph_level (ER1 -> ~1*d edges, ER4 -> ~4*d edges) instead of dagma_net's
per-edge-probability sampler (make_instances_erpedge). Everything else
(DAGMANet, dagma_solve, train_dagmanet, metrics) is reused as-is from
dagma_net.py -- no need to duplicate it here.

Run:
    python test_synthetic.py --graph-level 1   # ER1, writes to ER1_gauss/
    python test_synthetic.py --graph-level 4   # ER4, writes to ER4_gauss/
"""
import argparse
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dagma_net as dn

DEV = dn.DEV


# ----------------------------------------------------------------------------- CLI / pipeline
def main():
    ap = argparse.ArgumentParser(description="DAGMA-Net vs DAGMA on ER1/ER4 synthetic data")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (d=6, K=20)")
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--n", type=int, default=500, help="samples per SEM instance")
    ap.add_argument("--train", type=int, default=100, help="# training DAG instances")
    ap.add_argument("--val", type=int, default=20, help="# validation DAG instances")
    ap.add_argument("--test", type=int, default=30, help="# test DAG instances")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--outdir", type=str, default=None,
                     help="defaults to '<graph_type><graph_level>_<sem_type>', e.g. ER1_gauss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph-type", type=str, default="ER", help="'ER' or 'SF' (data_generation.simulate_dag)")
    ap.add_argument("--graph-level", type=int, default=1, help="expected edges = graph_level * d (1 -> ER1, 4 -> ER4)")
    ap.add_argument("--sem-type", type=str, default="gauss", help="noise type for simulate_linear_sem")
    args = ap.parse_args()

    if args.outdir is None:
        args.outdir = f"{args.graph_type}{args.graph_level}_{args.sem_type}"

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.quick:
        d = 6; K = 20; n = 200
        n_train = 20; n_val = 8; n_test = 8
        epochs = 20; patience = min(args.patience, 8)
    else:
        d = args.d; K = args.K; n = args.n
        n_train = args.train; n_val = args.val; n_test = args.test
        epochs = args.epochs; patience = args.patience

    os.makedirs(args.outdir, exist_ok=True)

    print("device:", DEV, "| torch", torch.__version__)
    print(f"config: d={d} K={K} n={n} train={n_train} val={n_val} test={n_test} epochs={epochs} "
          f"lr={args.lr} graph_type={args.graph_type} "
          f"graph_level={args.graph_level} sem_type={args.sem_type}\n")

    train_set = dn.make_instances_erlevel(d, n, count=n_train, seed=args.seed + 1,
                                           graph_type=args.graph_type, graph_level=args.graph_level, sem_type=args.sem_type)
    val_set   = dn.make_instances_erlevel(d, n, count=n_val,   seed=args.seed + 2,
                                           graph_type=args.graph_type, graph_level=args.graph_level, sem_type=args.sem_type)
    test_set  = dn.make_instances_erlevel(d, n, count=n_test,  seed=args.seed + 3,
                                           graph_type=args.graph_type, graph_level=args.graph_level, sem_type=args.sem_type)

    model = dn.DAGMANet(d, K=K).to(DEV)
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DAGMA-Net: K={K} layers, {n_learn} learnable parameters")

    model, hist = dn.train_dagmanet(model, train_set, val_set, epochs=epochs, lr=args.lr, patience=patience)

    # ---- training_curves.png
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.axvline(hist["best_epoch"] - 1, color="k", ls="--", lw=1, label=f"best epoch {hist['best_epoch']}")
    plt.xlabel("epoch"); plt.ylabel("loss  ($\\|W_K-W^*\\|^2 + \\gamma\\,h$)")
    plt.title("DAGMA-Net training (early stopping)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "training_curves.png")); plt.close()

    # ---- learned_schedules.png
    hp = model.hparams()
    mu0v = float(hp["mu0"]); alphav = float(hp["alpha"])
    mu_sched = mu0v * alphav ** np.arange(K)
    soff = hp["soff"].detach().cpu().numpy()
    eta = hp["eta"].detach().cpu().numpy()
    layers = np.arange(1, K + 1)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    ax[0].plot(layers, mu_sched, marker='.'); ax[0].set_title("$\\mu_k=\\mu_0\\,\\alpha^k$ (path weight)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("layer $k$")
    ax[1].plot(layers, soff, marker='.', color='tab:orange')
    ax[1].set_title("barrier offset  ($s_k=\\rho+$offset$_k$)"); ax[1].set_xlabel("layer $k$")
    ax[2].plot(layers, eta, marker='.', color='tab:green'); ax[2].set_title("learning rate $\\eta_k$")
    ax[2].set_xlabel("layer $k$")
    for a in ax.ravel(): a.grid(alpha=.3)
    plt.suptitle(f"Learned per-layer schedules ({args.graph_type}{args.graph_level})"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "learned_schedules.png")); plt.close()

    print(f"scalars:  lambda={float(hp['lam']):.3f}   alpha_decay={float(hp['alpha']):.3f}   "
          f"beta1={float(hp['b1']):.3f}   beta2={float(hp['b2']):.3f}   mu0={float(hp['mu0']):.3f}")

    # ---- convergence.png — mean +/- std over the full test set, same convention as dagma_net.py
    kk = np.arange(1, K + 1)
    net_score_all = np.zeros((len(test_set), K))
    net_dist_all  = np.zeros((len(test_set), K))
    net_rho_all   = np.zeros((len(test_set), K))
    with torch.no_grad():
        for i, (Sigma, Wst, _) in enumerate(test_set):
            _, tr = model(Sigma.to(DEV), trace=True, Wstar=Wst.to(DEV))
            net_score_all[i] = [r[1] for r in tr]
            net_dist_all[i]  = [r[2] for r in tr]
            net_rho_all[i]   = [r[4] for r in tr]
    net_score_mean, net_score_std = net_score_all.mean(0), net_score_all.std(0)
    net_dist_mean,  net_dist_std  = net_dist_all.mean(0),  net_dist_all.std(0)
    net_rho_mean = net_rho_all.mean(0)

    dag_traces = [dn.dagma_solve(Sigma.cpu(), K_outer=6, inner=1500, trace_every=25, Wstar=Wst.cpu())[1]
                  for Sigma, Wst, _ in test_set]
    m = min(len(t) for t in dag_traces)                    # guard against rare backtracking-retry drift
    it = [dag_traces[0][j][0] for j in range(m)]
    dag_score_all = np.array([[t[j][1] for j in range(m)] for t in dag_traces])
    dag_dist_all  = np.array([[t[j][2] for j in range(m)] for t in dag_traces])
    dag_score_mean, dag_score_std = dag_score_all.mean(0), dag_score_all.std(0)
    dag_dist_mean,  dag_dist_std  = dag_dist_all.mean(0),  dag_dist_all.std(0)

    def _band(ax, x, mean, std, color, **kw):
        ax.plot(x, mean, color=color, **kw)
        lo = np.clip(mean - std, 1e-12, None)
        ax.fill_between(x, lo, mean + std, color=color, alpha=0.15, linewidth=0)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    _band(ax[0], it, dag_dist_mean, dag_dist_std, "tab:gray", label="DAGMA (per iter)")
    _band(ax[0], kk, net_dist_mean, net_dist_std, "tab:blue", label="DAGMA-Net (per layer)", marker='.')
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration / layer (log scale)"); ax[0].set_ylabel("$\\|W-W^*\\|_F$")
    ax[0].set_title("distance to truth"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
    _band(ax[1], it, dag_score_mean, dag_score_std, "tab:gray", label="DAGMA")
    _band(ax[1], kk, net_score_mean, net_score_std, "tab:blue", label="DAGMA-Net", marker='.')
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("iteration / layer (log scale)"); ax[1].set_ylabel("least-squares score")
    ax[1].set_title("objective (fit) vs steps"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "convergence.png"), dpi=200); plt.close()

    # ---- convergence_paper.png — standalone least-squares-score panel, no title
    fig, axp = plt.subplots(1, 1, figsize=(3.5, 2.8))
    _band(axp, it, dag_score_mean, dag_score_std, "tab:gray", label="DAGMA")
    _band(axp, kk, net_score_mean, net_score_std, "tab:blue", label="DAGMA-Net", marker='.')
    axp.set_xscale("log"); axp.set_yscale("log")
    axp.set_xlabel("iteration / layer (log scale)"); axp.set_ylabel("least-squares score")
    axp.legend(); axp.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "convergence_paper.png"), dpi=200); plt.close()

    # ---- icassp_schedules.png — train/val curve, mu_k, eta_k, rho_k in one 1x4 row
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6), constrained_layout=True)
    ax[0].plot(hist["train"], label="train")
    ax[0].plot(hist["val"], label="val")
    ax[0].axvline(hist["best_epoch"] - 1, color="k", ls="--", lw=1, label="best")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("training")
    ax[0].legend(loc="upper right", framealpha=0.9, handlelength=1.4, borderaxespad=0.3)
    ax[0].grid(alpha=.3)

    ax[1].plot(layers, mu_sched, marker='.', color='tab:purple')
    ax[1].set_yscale("log"); ax[1].set_xlabel("layer $k$"); ax[1].set_ylabel("$\\mu_k$")
    ax[1].set_title("$\\mu_k=\\mu_0\\,\\alpha^k$"); ax[1].grid(alpha=.3)

    ax[2].plot(layers, eta, marker='.', color='tab:green')
    ax[2].set_xlabel("layer $k$"); ax[2].set_ylabel("$\\eta_k$")
    ax[2].set_title("learning rate"); ax[2].grid(alpha=.3)

    ax[3].plot(kk, net_rho_mean, marker='.', color='tab:red')
    ax[3].set_xlabel("layer $k$")
    ax[3].set_ylabel("$\\rho(\\mathbf{W}^{(k)}\\!\\circ\\!\\mathbf{W}^{(k)})$")
    ax[3].set_title("spectral radius"); ax[3].grid(alpha=.3)

    plt.savefig(os.path.join(args.outdir, "icassp_schedules.png"), dpi=200); plt.close()

    print(f"DAGMA-Net used {K} layers; DAGMA trace ran to {it[-1]} iterations.")

    # ---- shd_comparison.png (also times each method's test-time/inference cost)
    tau = 0.3
    net_shd, dag_shd = [], []
    net_time, dag_time = 0.0, 0.0
    model.eval()
    for Sigma, Wst, X in test_set:
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            Wn = model(Sigma.to(DEV)).cpu().numpy()
        if DEV == "cuda": torch.cuda.synchronize()
        net_time += time.time() - t0

        t0 = time.time()
        Wd, _ = dn.dagma_solve(Sigma, K_outer=6, inner=1500)
        dag_time += time.time() - t0

        Wd = Wd.numpy()
        Wt = Wst.numpy()
        net_shd.append(dn.shd(Wt, dn.threshold(Wn, tau)))
        dag_shd.append(dn.shd(Wt, dn.threshold(Wd, tau)))

    net_shd, dag_shd = np.array(net_shd), np.array(dag_shd)
    print(f"DAGMA         : SHD mean {dag_shd.mean():.2f} ± {dag_shd.std():.2f}  (K_outer*inner = 9000 iters)"
          f"  |  test-time {dag_time:.2f}s total, {dag_time/len(test_set)*1000:.1f} ms/DAG")
    print(f"DAGMA-Net (K={K}): SHD mean {net_shd.mean():.2f} ± {net_shd.std():.2f}  ({K} layers)"
          f"  |  test-time {net_time:.2f}s total, {net_time/len(test_set)*1000:.1f} ms/DAG"
          f"  ({dag_time/max(net_time, 1e-9):.1f}x faster than DAGMA)")

    # Raw per-DAG SHD values, saved so ER1/ER4/Sachs runs can later be combined into one box plot.
    np.savez(os.path.join(args.outdir, "shd_raw.npz"), dagma=dag_shd, dagma_net=net_shd)

    plt.figure(figsize=(6, 4))
    plt.bar(["DAGMA\n(~9000 iters)", f"DAGMA-Net\n({K} layers)"],
            [dag_shd.mean(), net_shd.mean()],
            yerr=[dag_shd.std(), net_shd.std()], capsize=6,
            color=["tab:gray", "tab:blue"])
    plt.ylabel("SHD (lower is better)")
    plt.title(f"Structure recovery on {len(test_set)} test DAGs ({args.graph_type}{args.graph_level}, d={d})")
    plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shd_comparison.png")); plt.close()

    # ---- run_summary.txt
    summary = (
        f"DAGMA-Net run summary ({args.graph_type}{args.graph_level}_{args.sem_type})\n"
        f"======================\n"
        f"config: d={d} K={K} n={n} train={n_train} val={n_val} test={n_test} epochs={epochs} "
        f"lr={args.lr} seed={args.seed} "
        f"graph_type={args.graph_type} graph_level={args.graph_level} sem_type={args.sem_type}\n"
        f"learnable parameters: {n_learn}\n\n"
        f"learned scalars:\n"
        f"  lambda      = {float(hp['lam']):.4f}\n"
        f"  alpha_decay = {float(hp['alpha']):.4f}\n"
        f"  beta1       = {float(hp['b1']):.4f}\n"
        f"  beta2       = {float(hp['b2']):.4f}\n"
        f"  mu0         = {float(hp['mu0']):.4f}\n\n"
        f"training:\n"
        f"  best_epoch = {hist['best_epoch']}\n"
        f"  best_val   = {min(hist['val']):.4f}\n\n"
        f"DAGMA-Net used {K} layers; DAGMA trace ran to {it[-1]} iterations.\n\n"
        f"SHD comparison ({len(test_set)} test DAGs, d={d}):\n"
        f"  DAGMA          : mean {dag_shd.mean():.2f} +/- {dag_shd.std():.2f}  (K_outer*inner = 9000 iters)\n"
        f"  DAGMA-Net (K={K}): mean {net_shd.mean():.2f} +/- {net_shd.std():.2f}  ({K} layers)\n\n"
        f"test-time (inference) runtime, {len(test_set)} test DAGs:\n"
        f"  DAGMA          : {dag_time:.2f}s total, {dag_time/len(test_set)*1000:.1f} ms/DAG\n"
        f"  DAGMA-Net (K={K}): {net_time:.2f}s total, {net_time/len(test_set)*1000:.1f} ms/DAG"
        f"  ({dag_time/max(net_time, 1e-9):.1f}x faster than DAGMA)\n"
    )
    with open(os.path.join(args.outdir, "run_summary.txt"), "w") as f:
        f.write(summary)

    print(f"\nwrote plots and run_summary.txt to {args.outdir}/")


if __name__ == "__main__":
    main()
