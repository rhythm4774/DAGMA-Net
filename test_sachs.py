"""
DAGMA-Net vs DAGMA, trained on synthetic ER data (dagma_net.make_instances_erpedge,
per-edge probability p=0.3) and evaluated on the Sachs protein-signalling flow
cytometry dataset (sachs.data.txt, n=853, d=11), against the 17-edge consensus
benchmark (with 1 edge reversed), preprocessed with a single global scalar
(same convention as dagma_net.py's synthetic data).
"""

import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dagma_net as dn

OUTDIR = "results_sachs"
os.makedirs(OUTDIR, exist_ok=True)

print("device:", dn.DEV)


def h_dagma_np(W):
    ev = np.max(np.abs(np.linalg.eigvals(W * W)))
    s = ev + 1.0
    M = s * np.eye(W.shape[0]) - W * W
    sign, logdet = np.linalg.slogdet(M)
    return -logdet + W.shape[0] * np.log(s)


# ---------------------------------------------------------------------------
# 1) Train DAGMA-Net on synthetic d=11 ER data (make_instances_erpedge, p=0.3).
# ---------------------------------------------------------------------------
d = 11
K = 40
n = 800
seed = 0
p_edge = 0.3

np.random.seed(seed)
torch.manual_seed(seed)

train_set = dn.make_instances_erpedge(d, n, count=100, seed=seed + 1, p=p_edge)
val_set = dn.make_instances_erpedge(d, n, count=20, seed=seed + 2, p=p_edge)

model = dn.DAGMANet(d, K=K).to(dn.DEV).double()
n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DAGMA-Net: K={K} layers, {n_learn} learnable parameters")

t0 = time.perf_counter()
model, hist = dn.train_dagmanet(model, train_set, val_set, epochs=150, lr=1e-2, patience=20, verbose=True)
train_seconds = time.perf_counter() - t0
print(f"training took {train_seconds:.1f}s, best epoch {hist['best_epoch']}")

from pathlib import Path
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "sachs.data.txt", sep="\t")
X_raw = df.to_numpy(dtype=np.float32)
X_sachs = (X_raw - X_raw.mean(0)) / (X_raw.std() + 1e-8)

# 17-edge consensus Sachs DAG (1 edge reversed relative to the original protein-signalling paper)
sachs_edges_17 = [
    ("PKC", "PKA"), ("PKC", "Raf"), ("PKA", "Raf"),
    ("PKC", "Mek"), ("PKA", "Mek"), ("Raf", "Mek"),
    ("Mek", "Erk"), ("PKA", "Erk"),
    ("Erk", "Akt"), ("PKA", "Akt"),
    ("PKC", "P38"), ("PKA", "P38"),
    ("PKC", "Jnk"), ("PKA", "Jnk"),
    ("Plcg", "PIP3"), ("Plcg", "PIP2"), ("PIP3", "PIP2"),
]

node_names = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]
node_to_idx = {name: i for i, name in enumerate(node_names)}

W_true_sachs = np.zeros((d, d))
for src, dst in sachs_edges_17:
    W_true_sachs[node_to_idx[src], node_to_idx[dst]] = 1.0

print("true edges:", int(W_true_sachs.sum()), "| h(W_true):", h_dagma_np(W_true_sachs), "| is_acyclic:", dn.is_acyclic_support(W_true_sachs))
Wstar_sachs = torch.tensor(W_true_sachs, dtype=torch.float64)

Sigma_sachs = torch.tensor(dn.gram(X_sachs), dtype=torch.float64)

# ---------------------------------------------------------------------------
# 2) convergence.png
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    W_net, tr_net = model(Sigma_sachs.to(dn.DEV), trace=True, Wstar=Wstar_sachs.to(dn.DEV))
kk = [r[0] for r in tr_net]; net_score = [r[1] for r in tr_net]; net_dist = [r[2] for r in tr_net]

W_dag, tr_dag = dn.dagma_solve(Sigma_sachs, K_outer=6, inner=1500, trace_every=25, Wstar=Wstar_sachs)
it = [r[0] for r in tr_dag]; dag_score = [r[1] for r in tr_dag]; dag_dist = [r[2] for r in tr_dag]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(it, dag_dist, label="DAGMA (per iter)", color="tab:gray")
ax[0].plot(kk, net_dist, label="DAGMA-Net (per layer)", color="tab:blue", marker='.')
ax[0].set_xlabel("iteration / layer"); ax[0].set_ylabel("$\\|W-W^*\\|_F$")
ax[0].set_title("distance to Sachs truth (17-edge)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_yscale("log")
ax[1].plot(it, dag_score, label="DAGMA", color="tab:gray")
ax[1].plot(kk, net_score, label="DAGMA-Net", color="tab:blue", marker='.')
ax[1].set_xlabel("iteration / layer"); ax[1].set_ylabel("least-squares score")
ax[1].set_title("objective (fit) vs steps on Sachs"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "convergence.png")); plt.close()

print(f"DAGMA-Net used {K} layers; DAGMA trace ran to {it[-1]} iterations.")

# ---------------------------------------------------------------------------
# 3) shd_comparison.png -- Sachs is a single fixed dataset, so we replicate
#    each method N_REPS times on the same Sigma_sachs/W_true_sachs and report
#    mean +/- std (SHD should be ~deterministic; timing std reflects wall-clock
#    noise).
# ---------------------------------------------------------------------------
tau = 0.3
N_REPS = 30
model.eval()

net_shd, dag_shd = [], []
net_time, dag_time = [], []
for _ in range(N_REPS):
    if dn.DEV == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        Wn = model(Sigma_sachs.to(dn.DEV)).cpu().numpy()
    if dn.DEV == "cuda": torch.cuda.synchronize()
    net_time.append(time.time() - t0)

    t0 = time.time()
    Wd, _ = dn.dagma_solve(Sigma_sachs, K_outer=6, inner=1500)
    dag_time.append(time.time() - t0)
    Wd = Wd.numpy()

    net_shd.append(dn.shd(W_true_sachs, dn.threshold(Wn, tau)))
    dag_shd.append(dn.shd(W_true_sachs, dn.threshold(Wd, tau)))

net_shd, dag_shd = np.array(net_shd, dtype=float), np.array(dag_shd, dtype=float)
net_time, dag_time = np.array(net_time), np.array(dag_time)

print(f"DAGMA         : SHD mean {dag_shd.mean():.2f} +/- {dag_shd.std():.2f} | (K_outer*inner = {it[-1]} iters)"
      f"  |  test-time {dag_time.mean()*1000:.2f} +/- {dag_time.std()*1000:.2f} ms/DAG  ({N_REPS} reps)")
print(f"DAGMA-Net (K={K}): SHD mean {net_shd.mean():.2f} +/- {net_shd.std():.2f}  ({K} layers)"
      f"  |  test-time {net_time.mean()*1000:.2f} +/- {net_time.std()*1000:.2f} ms/DAG  ({N_REPS} reps)"
      f"  ({dag_time.mean()/max(net_time.mean(), 1e-9):.1f}x faster than DAGMA)")

plt.figure(figsize=(6, 4))
plt.bar([f"DAGMA\n(~{it[-1]} iters)", f"DAGMA-Net\n({K} layers)"],
        [dag_shd.mean(), net_shd.mean()],
        yerr=[dag_shd.std(), net_shd.std()], capsize=6,
        color=["tab:gray", "tab:blue"])
plt.ylabel("SHD"); plt.title(f"Structure recovery on Sachs, averaged over {N_REPS} reps")
plt.grid(alpha=.3, axis="y"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "shd_comparison.png")); plt.close()

# ---------------------------------------------------------------------------
# 4) run_summary.txt
# ---------------------------------------------------------------------------
summary = (
    f"DAGMA-Net on Sachs (17-edge benchmark) -- run summary\n"
    f"=====================================================================\n"
    f"config: d={d} K={K} n={n} train=100 val=20 epochs=150 lr=1e-2 seed={seed} p_edge={p_edge}\n"
    f"learnable parameters: {n_learn}\n\n"
    f"data shape {df.shape}\n"
    f"preprocessing: global-scalar division (X - mean) / std()\n"
    f"training: best_epoch = {hist['best_epoch']}, best_val = {min(hist['val']):.4f}, "
    f"{train_seconds:.1f}s\n\n"
    f"DAGMA-Net used {K} layers; DAGMA trace ran to {it[-1]} iterations.\n\n"
    f"SHD on Sachs (17 true edges, d=11, threshold={tau}), averaged over {N_REPS} reps:\n"
    f"  DAGMA          : mean {dag_shd.mean():.2f} +/- {dag_shd.std():.2f}  ({it[-1]} iterations)\n"
    f"  DAGMA-Net (K={K}): mean {net_shd.mean():.2f} +/- {net_shd.std():.2f}  ({K} layers)\n\n"
    f"test-time (inference) runtime on Sachs, averaged over {N_REPS} reps:\n"
    f"  DAGMA          : {dag_time.mean()*1000:.2f} +/- {dag_time.std()*1000:.2f} ms/DAG\n"
    f"  DAGMA-Net (K={K}): {net_time.mean()*1000:.2f} +/- {net_time.std()*1000:.2f} ms/DAG"
    f"  ({dag_time.mean()/max(net_time.mean(), 1e-9):.1f}x faster than DAGMA)\n"
)
with open(os.path.join(OUTDIR, "run_summary.txt"), "w") as f:
    f.write(summary)

print(f"\nwrote plots and run_summary.txt to {OUTDIR}/")
