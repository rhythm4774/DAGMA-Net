"""
DAGMA-Net: unrolls the DAGMA DAG-structure-learning solver into a fixed-depth network
whose layers are Adam update steps, and learns DAGMA's hyper-parameters.

Run:
    python dagma_net.py --quick        # fast smoke test (d=6, K=20)
    python dagma_net.py                # default (d=10, K=40); writes PNGs to ./out
    python dagma_net.py --d 20 --K 60 --epochs 200
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generation import simulate_dag, simulate_parameter, simulate_linear_sem, set_random_seed


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data / metrics
def random_dag_W(d, p=0.3, wmin=0.5, wmax=2.0, rng=None):
    rng = rng or np.random.default_rng()
    perm = rng.permutation(d)
    U = np.triu((rng.random((d, d)) < p).astype(float), 1)
    W = U * rng.choice([-1., 1.], (d, d)) * rng.uniform(wmin, wmax, (d, d))
    P = np.eye(d)[perm]
    return P.T @ W @ P                                              # permute back (still a DAG)


def sample_sem(W, n, rng=None):
    rng = rng or np.random.default_rng()
    d = W.shape[0]
    E = rng.normal(0., 1., (n, d))
    X = E @ np.linalg.inv(np.eye(d) - W)
    X = X - X.mean(0)
    return X / (X.std() + 1e-8)                                     # global scalar: keeps W* and edge orientation


def gram(X):
    return (X.T @ X) / X.shape[0]


def threshold(W, tau=0.3):
    return W * (np.abs(W) > tau)


def shd(B_true, B_est, thr=1e-8):
    A = (np.abs(B_true) > thr).astype(int); B = (np.abs(B_est) > thr).astype(int)
    np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    rev = ((A == 1) & (B.T == 1) & (B == 0) & (A.T == 0)).sum()     # reversed edges
    missing = ((A == 1) & (B == 0)).sum(); extra = ((A == 0) & (B == 1)).sum()
    return int(missing + extra - rev)


def is_acyclic_support(W, thr=1e-8):
    """Is the support (nonzero pattern) of W a DAG? Uses networkx if available,
    else a pure-python Kahn's algorithm."""
    A = (np.abs(W) > thr).astype(int)
    d = A.shape[0]
    np.fill_diagonal(A, 0)
    try:
        import networkx as nx
        G = nx.DiGraph(A)
        return nx.is_directed_acyclic_graph(G)
    except Exception:
        indeg = A.sum(axis=0).tolist()
        stack = [i for i in range(d) if indeg[i] == 0]
        visited = 0
        while stack:
            i = stack.pop()
            visited += 1
            for j in range(d):
                if A[i, j]:
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        stack.append(j)
        return visited == d


def make_instances_erpedge(d, n, count, seed, p=0.3):
    """ER graphs sampled by per-edge probability (random_dag_W's `p`)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        W = random_dag_W(d, p=p, rng=rng); X = sample_sem(W, n, rng=rng)
        out.append((torch.tensor(gram(X)), torch.tensor(W), X))     # (Sigma, W*, X)
    return out


def make_instance_erlevel(d, n, graph_type, graph_level, sem_type):
    """One ER1/ER4-style instance: data_generation.simulate_dag with an explicit
    expected-degree graph_level (ER1 -> ~1*d edges, ER4 -> ~4*d edges), instead of
    random_dag_W's per-edge-probability sampler."""
    s0 = graph_level * d
    B_true = simulate_dag(d, s0, graph_type)
    W = simulate_parameter(B_true)
    X = simulate_linear_sem(W, n, sem_type)
    X = (X - X.mean(0)) / (X.std() + 1e-8)
    return W, X


def make_instances_erlevel(d, n, count, seed, graph_type="ER", graph_level=1, sem_type="gauss"):
    set_random_seed(seed)
    out = []
    for _ in range(count):
        W, X = make_instance_erlevel(d, n, graph_type, graph_level, sem_type)
        out.append((torch.tensor(gram(X)), torch.tensor(W), X))     # (Sigma, W*, X)
    return out


# ----------------------------------------------------------------------------- self-defending barrier
def spectral_radius(A):                     # rho(A) = max |eigenvalue|, detached scalar
    with torch.no_grad():
        return torch.linalg.eigvals(A).abs().max().real


def _barrier(W, s=None):
    # Sanitize W and build M = sI - W∘W guaranteed positive-definite, so downstream
    # inverses/log-dets can never see a singular or NaN matrix.
    W = torch.nan_to_num(torch.clamp(W, -1e3, 1e3))
    d = W.shape[0]; I = torch.eye(d, dtype=W.dtype, device=W.device)
    s_floor = 1.05 * spectral_radius(W * W) + 0.1          # strictly above rho => M is PD
    if s is None:
        s = s_floor
    else:
        s = torch.as_tensor(s, dtype=W.dtype, device=W.device)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e6, neginf=0.0)
        s = torch.maximum(s, s_floor)
    return W, s, s * I - W * W


def grad_h_ldet(W, s=None):                 # gradient of the log-det acyclicity term
    W, s, M = _barrier(W, s)
    return 2.0 * torch.linalg.inv(M).T * W


def h_ldet(W, s_off=1.0):                   # value of the log-det acyclicity term
    W, s, M = _barrier(W)
    sign, logabsdet = torch.linalg.slogdet(M)
    return -logabsdet + W.shape[0] * torch.log(s)


def lsq_score(Sigma, W):                    # 1/2n ||X - XW||^2 using the Gram matrix only
    d = W.shape[0]; I = torch.eye(d, dtype=W.dtype, device=W.device)
    return 0.5 * torch.trace((I - W).T @ Sigma @ (I - W))


def grad_lsq(Sigma, W):                     # = Sigma (W - I)
    d = W.shape[0]; I = torch.eye(d, dtype=W.dtype, device=W.device)
    return Sigma @ (W - I)


# ----------------------------------------------------------------------------- vanilla DAGMA baseline
def dagma_solve(Sigma, K_outer=6, inner=3000, mu0=1.0, alpha=0.1, lam=0.03,
                s_list=None, lr=3e-4, b1=0.99, b2=0.999, clip=3.0, Wstar=None, trace_every=0):
    """Faithful port of the reference DAGMA `fit()`/`minimize()` (kevinsbello/dagma linear.py):
    a FIXED per-outer-iteration `s` schedule (not DAGMANet's adaptive self-defending barrier),
    with backtracking (undo the last step, halve lr, retry) whenever an update leaves the
    M-matrix domain (inv(sI - W*W) picks up a negative entry), and a full outer-loop retry with
    a larger s if backtracking still can't recover. No backprop happens through this function.
    b1/lr/alpha(mu_factor)/lam/s_list match the paper's defaults: beta_1=0.99, lr=3e-4,
    mu_factor=0.1, lambda1=0.03, s=[1.0, .9, .8, .7, .6] (repeating the last value if K_outer
    is longer)."""
    d = Sigma.shape[0]; I = torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    mask = 1 - I
    if s_list is None:
        s_list = [1.0, 0.9, 0.8, 0.7, 0.6]
    s_list = list(s_list)
    if len(s_list) < K_outer:
        s_list = s_list + [s_list[-1]] * (K_outer - len(s_list))

    W = torch.zeros(d, d, dtype=Sigma.dtype, device=Sigma.device)
    mu = mu0
    t = 0
    tr = []                                                          # (iter, score, dist-to-truth)

    def _minimize(W0, mu, max_iter, s, lr):
        nonlocal t
        W = W0.clone()
        m = torch.zeros_like(W); v = torch.zeros_like(W)
        grad = torch.zeros_like(W)
        for it in range(1, max_iter + 1):
            M = s * I - W * W
            Minv = torch.linalg.inv(M) + 1e-16
            if torch.any(Minv < 0):                                  # W left the M-matrix domain
                if it == 1 or s <= 0.9:
                    return W, False                                  # signal caller to retry with larger s
                while True:
                    W = (W + lr * grad) * mask                       # undo the last step
                    lr = lr * 0.5
                    if lr <= 1e-16:
                        return W, True
                    W = (W - lr * grad) * mask                       # redo it with a smaller lr
                    M = s * I - W * W
                    Minv = torch.linalg.inv(M) + 1e-16
                    if not torch.any(Minv < 0):
                        break
            g_score = mu * grad_lsq(Sigma, W)
            Gobj = (g_score + mu * lam * torch.sign(W) + 2.0 * W * Minv.T) * mask
            m = b1 * m + (1 - b1) * Gobj; v = b2 * v + (1 - b2) * Gobj * Gobj
            mhat = m / (1 - b1**it); vhat = v / (1 - b2**it)
            grad = mhat / (torch.sqrt(vhat) + 1e-8)
            W = (W - lr * grad) * mask
            W = torch.nan_to_num(torch.clamp(W, -clip, clip))        # extra safety net, not in the paper
            t += 1
            if trace_every and t % trace_every == 0:
                dist = float(torch.norm(W - Wstar)) if Wstar is not None else np.nan
                tr.append((t, float(lsq_score(Sigma, W)), dist))
        return W, True

    for outer in range(K_outer):
        s = s_list[outer]; lr_cur = lr; success = False
        while not success:
            W_new, success = _minimize(W, mu, inner, s, lr_cur)
            if not success:
                lr_cur *= 0.5; s += 0.1
        W = W_new
        mu *= alpha
    return W, tr


# ----------------------------------------------------------------------------- DAGMA-Net
def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def _logit(y):
    return float(np.log(y / (1 - y)))


DEFAULT_FLAGS = dict(lam=True, mu=True, s=True, momentum=True, lr=True)


class DAGMANet(nn.Module):
    def __init__(self, d, K=40, flags=None, clip=3.0,
                 lam0=0.05, mu0=2.0, alpha0=0.9, soff0=1.0, eta0=0.06, b1_0=0.9, b2_0=0.99):
        super().__init__()
        self.d, self.K, self.clip = d, K, clip
        f = {**DEFAULT_FLAGS, **(flags or {})}; self.flags = f

        def P(init, learn):
            return nn.Parameter(torch.tensor(init), requires_grad=learn)

        # scalars
        self.raw_lam   = P(_inv_softplus(lam0),  f["lam"])
        self.raw_mu0   = P(_inv_softplus(mu0),   f["mu"])
        self.raw_alpha = P(_logit(alpha0),       f["mu"])
        self.raw_b1    = P(_logit(b1_0),         f["momentum"])
        self.raw_b2    = P(_logit(b2_0),         f["momentum"])
        # per-layer vectors
        self.raw_soff = P(np.full(K, _inv_softplus(soff0)), f["s"])
        self.raw_eta  = P(np.full(K, _inv_softplus(eta0)),  f["lr"])
        self.register_buffer("I",    torch.eye(d))
        self.register_buffer("mask", 1 - torch.eye(d))

    def hparams(self):                                   # decoded schedules (for plotting)
        sp = torch.nn.functional.softplus; sg = torch.sigmoid
        return dict(lam=sp(self.raw_lam), mu0=sp(self.raw_mu0), alpha=sg(self.raw_alpha),
                    b1=sg(self.raw_b1), b2=sg(self.raw_b2),
                    soff=sp(self.raw_soff), eta=sp(self.raw_eta))

    def forward(self, Sigma, trace=False, Wstar=None, collect=False):
        sp = torch.nn.functional.softplus; sg = torch.sigmoid
        lam, mu0, alpha = sp(self.raw_lam), sp(self.raw_mu0), sg(self.raw_alpha)
        b1, b2 = sg(self.raw_b1), sg(self.raw_b2)
        W = torch.zeros(self.d, self.d, dtype=Sigma.dtype, device=Sigma.device)
        m = torch.zeros_like(W); v = torch.zeros_like(W); tr = []; ws = []
        for k in range(self.K):
            mu    = mu0 * alpha**k
            rho_k = spectral_radius(W * W)                                  # rho(W^(k) o W^(k)), pre-update
            s     = 1.05 * rho_k + sp(self.raw_soff[k]) + 0.1                # margin above rho
            eta   = sp(self.raw_eta[k])
            g     = mu * (grad_lsq(Sigma, W) + lam * torch.sign(W)) + grad_h_ldet(W, s)
            g     = g * self.mask
            m    = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1**(k + 1)); vhat = v / (1 - b2**(k + 1))
            W    = (W - eta * mhat / torch.sqrt(vhat + 1e-8)) * self.mask
            W    = torch.nan_to_num(torch.clamp(W, -self.clip, self.clip))  # always finite & bounded
            if collect:
                ws.append(W)
            if trace:
                d2 = float(torch.norm(W - Wstar)) if Wstar is not None else np.nan
                tr.append((k + 1, float(lsq_score(Sigma, W)), d2, float(s), float(rho_k)))
        if trace:   return W, tr
        if collect: return W, ws
        return W


def train_dagmanet(model, train_set, val_set, epochs=150, lr=3e-2, gamma=1e-3,
                   patience=20, verbose=True):
    learnable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(learnable, lr=lr)
    wts = torch.arange(1, model.K + 1, dtype=torch.float64, device=DEV)     # deep-supervision ramp
    wts = wts / wts.sum()
    hist = {"train": [], "val": []}
    best_val, best_state, wait, best_ep = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for Sigma, Wst, _ in train_set:
            opt.zero_grad()
            Wt = Wst.to(DEV)
            W, ws = model(Sigma.to(DEV), collect=True)                      # every layer supervised
            deep = sum(wts[k] * ((ws[k] - Wt)**2).sum() for k in range(model.K)) + gamma * h_ldet(W)
            if not torch.isfinite(deep):
                continue                                                    # skip a pathological instance
            deep.backward()
            for p in learnable:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nn.utils.clip_grad_norm_(learnable, 10.0)
            opt.step()
            with torch.no_grad():                                         # params stay finite no matter what
                for p in learnable:
                    torch.nan_to_num_(p, nan=0.0, posinf=1e3, neginf=-1e3)
            tl += ((W - Wt)**2).sum().item()                              # log final-layer MSE (comparable to val)
        tl /= len(train_set)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for Sigma, Wst, _ in val_set:
                W = model(Sigma.to(DEV))
                vl += ((W - Wst.to(DEV))**2).sum().item()
        vl /= len(val_set)
        hist["train"].append(tl); hist["val"].append(vl)
        if vl < best_val - 1e-6:
            best_val, best_state, wait, best_ep = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            wait += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"epoch {ep:3d}  train {tl:8.3f}  val {vl:8.3f}  (best@{best_ep}, wait {wait})")
        if wait >= patience:
            print(f"early stop at epoch {ep} (best epoch {best_ep}, val {best_val:.3f})"); break
    model.load_state_dict(best_state)
    hist["best_epoch"] = best_ep
    return model, hist


# ----------------------------------------------------------------------------- CLI / pipeline
def main():
    ap = argparse.ArgumentParser(description="DAGMA-Net training/eval pipeline")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (d=6, K=20)")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--n", type=int, default=500, help="samples per SEM instance")
    ap.add_argument("--train", type=int, default=None, help="# training DAG instances")
    ap.add_argument("--val", type=int, default=20, help="# validation DAG instances")
    ap.add_argument("--test", type=int, default=30, help="# test DAG instances")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="./out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.quick:
        d = args.d or 6; K = args.K or 20; n = 200
        n_train = args.train or 20; n_val = 8; n_test = 8
        epochs = args.epochs or 20; patience = min(args.patience, 8)
    else:
        d = args.d or 10; K = args.K or 40; n = args.n
        n_train = args.train or 80; n_val = args.val; n_test = args.test
        epochs = args.epochs or 150; patience = args.patience

    os.makedirs(args.outdir, exist_ok=True)

    print("device:", DEV, "| torch", torch.__version__)
    print(f"config: d={d} K={K} n={n} train={n_train} val={n_val} test={n_test} epochs={epochs} "
          f"lr={args.lr}\n")

    train_set = make_instances_erpedge(d, n, count=n_train, seed=args.seed + 1)
    val_set   = make_instances_erpedge(d, n, count=n_val,   seed=args.seed + 2)
    test_set  = make_instances_erpedge(d, n, count=n_test,  seed=args.seed + 3)

    model = DAGMANet(d, K=K).to(DEV)
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DAGMA-Net: K={K} layers, {n_learn} learnable parameters")

    model, hist = train_dagmanet(model, train_set, val_set, epochs=epochs, lr=args.lr, patience=patience)

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
    plt.suptitle("Learned per-layer schedules"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "learned_schedules.png")); plt.close()

    print(f"scalars:  lambda={float(hp['lam']):.3f}   alpha_decay={float(hp['alpha']):.3f}   "
          f"beta1={float(hp['b1']):.3f}   beta2={float(hp['b2']):.3f}   mu0={float(hp['mu0']):.3f}")

    # ---- convergence.png — mean +/- std over the full test set (averaging removes the
    # per-instance Adam-overshoot noise that a single test DAG's trace would show)
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

    dag_traces = [dagma_solve(Sigma.cpu(), K_outer=6, inner=1500, trace_every=25, Wstar=Wst.cpu())[1]
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

    # ---- convergence_paper.png — standalone least-squares-score panel, no title, for direct
    # inclusion in the paper (single-column width)
    fig, axp = plt.subplots(1, 1, figsize=(3.5, 2.8))
    _band(axp, it, dag_score_mean, dag_score_std, "tab:gray", label="DAGMA")
    _band(axp, kk, net_score_mean, net_score_std, "tab:blue", label="DAGMA-Net", marker='.')
    axp.set_xscale("log"); axp.set_yscale("log")
    axp.set_xlabel("iteration / layer (log scale)"); axp.set_ylabel("least-squares score")
    axp.legend(); axp.grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "convergence_paper.png"), dpi=200); plt.close()

    # ---- icassp_schedules.png — train/val curve, mu_k, eta_k, rho_k in one 1x4 row sized to span
    # both columns of an ICASSP page
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
        Wd, _ = dagma_solve(Sigma, K_outer=6, inner=1500)
        dag_time += time.time() - t0

        Wd = Wd.numpy()
        Wt = Wst.numpy()
        net_shd.append(shd(Wt, threshold(Wn, tau)))
        dag_shd.append(shd(Wt, threshold(Wd, tau)))

    net_shd, dag_shd = np.array(net_shd), np.array(dag_shd)
    print(f"DAGMA         : SHD mean {dag_shd.mean():.2f} ± {dag_shd.std():.2f}  (K_outer*inner = 9000 iters)"
          f"  |  test-time {dag_time:.2f}s total, {dag_time/len(test_set)*1000:.1f} ms/DAG")
    print(f"DAGMA-Net (K={K}): SHD mean {net_shd.mean():.2f} ± {net_shd.std():.2f}  ({K} layers)"
          f"  |  test-time {net_time:.2f}s total, {net_time/len(test_set)*1000:.1f} ms/DAG"
          f"  ({dag_time/max(net_time, 1e-9):.1f}x faster than DAGMA)")

    plt.figure(figsize=(6, 4))
    plt.bar(["DAGMA\n(~9000 iters)", f"DAGMA-Net\n({K} layers)"],
            [dag_shd.mean(), net_shd.mean()],
            yerr=[dag_shd.std(), net_shd.std()], capsize=6,
            color=["tab:gray", "tab:blue"])
    plt.ylabel("SHD (lower is better)"); plt.title(f"Structure recovery on {len(test_set)} test DAGs (d={d})")
    plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "shd_comparison.png")); plt.close()

    # ---- run_summary.txt
    summary = (
        f"DAGMA-Net run summary\n"
        f"======================\n"
        f"config: d={d} K={K} n={n} train={n_train} val={n_val} test={n_test} epochs={epochs} "
        f"lr={args.lr} seed={args.seed}\n"
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
