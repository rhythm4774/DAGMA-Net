"""
Fast green/red signals for DAGMA-Net. Run:  pytest -q

Tiers:
  * cheap correctness       — SHD metric, data is a DAG, global-scale conditioning
  * crash regression        — the self-defending barrier must never blow up (the _LinAlgError saga)
  * forward sanity          — output finite, right shape, zero diagonal
  * *** the optimization gate *** :
      test_gradients_flow        — the deep-supervision loss has usable (finite, nonzero) gradient
      test_training_does_not_diverge — a short train run stays finite and never worsens val
      test_training_reduces_loss — a short train run improves val by >= 5%  <-- turn this GREEN

The last test is the thing to optimize: if training is flat/broken it goes red; when DAGMA-Net
actually learns, it goes green. Keep it as the target signal while tuning.
"""
import numpy as np
import pytest

import dagma_net as dn   # the module under test (dagma_net.py in the same folder)

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

torch_required = pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")


# ----------------------------------------------------------------------------- SHD metric
def _e(d, edges):
    """Build a d×d binary adjacency from a list of (i,j) directed edges."""
    A = np.zeros((d, d))
    for i, j in edges:
        A[i, j] = 1.0
    return A


def test_shd_identical_is_zero():
    A = _e(4, [(0, 1), (1, 2), (0, 3)])
    assert dn.shd(A, A) == 0


def test_shd_reversal_counts_one():
    A = _e(4, [(0, 1)]); B = _e(4, [(1, 0)])
    assert dn.shd(A, B) == 1


def test_shd_missing_and_extra_count_one_each():
    A = _e(4, [(0, 1)])
    assert dn.shd(A, _e(4, []))            == 1     # missing
    assert dn.shd(A, _e(4, [(0, 1), (2, 3)])) == 1  # one correct + one extra


def test_shd_multi_edge():
    A = _e(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    B = _e(5, [(0, 1), (2, 1), (2, 3)])            # 1 correct, 1 reversed(1<->2), 1 correct, 2 missing
    # (1->2) vs (2->1): reversal=1 ; (3->4),(?) missing... compute against implementation
    val = dn.shd(A, B)
    assert isinstance(val, int) and val >= 1


def test_shd_ignores_weights_and_diagonal():
    A = _e(3, [(0, 1)]) * 2.7
    B = _e(3, [(0, 1)]) * -0.4
    np.fill_diagonal(A, 5.0); np.fill_diagonal(B, -9.0)   # diagonal must be ignored
    assert dn.shd(A, B) == 0


# ----------------------------------------------------------------------------- data properties
def test_generated_graph_is_a_dag():
    rng = np.random.default_rng(0)
    for _ in range(20):
        W = dn.random_dag_W(8, rng=rng)
        assert dn.is_acyclic_support(W)


def test_global_scaling_keeps_covariance_bounded():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        W = dn.random_dag_W(10, p=0.4, wmin=0.5, wmax=1.5, rng=rng)   # amplifying
        X = dn.sample_sem(W, 800, rng=rng)
        assert np.isfinite(X).all()
        worst = max(worst, float(np.max(np.abs(X.T @ X / 800))))
    assert worst < 50.0, f"covariance too large ({worst:.1f}) — global scaling not effective"


@torch_required
def test_true_dag_has_zero_h():
    rng = np.random.default_rng(2)
    W = dn.random_dag_W(10, rng=rng)
    h = float(dn.h_ldet(torch.tensor(W)))
    assert abs(h) < 1e-6, f"h(W*) should be ~0, got {h}"


# ----------------------------------------------------------------------------- crash regression
@torch_required
@pytest.mark.parametrize("bad", ["nan", "inf", "huge", "boundary"])
def test_barrier_never_singular(bad):
    d = 10
    if bad == "nan":      W = torch.full((d, d), float("nan"))
    elif bad == "inf":    W = torch.full((d, d), float("inf"))
    elif bad == "huge":   W = torch.full((d, d), 1e9)
    else:                 W = torch.full((d, d), 3.0) * (1 - torch.eye(d))
    for s in (None, -50.0, float("nan")):                 # incl. a NaN barrier passed in
        g = dn.grad_h_ldet(W, s)
        assert torch.isfinite(g).all(), f"grad_h_ldet non-finite for bad={bad}, s={s}"


# ----------------------------------------------------------------------------- forward sanity
@torch_required
def test_forward_finite_shaped_and_masked():
    torch.manual_seed(0)
    d, K = 8, 15
    W = dn.random_dag_W(d, rng=np.random.default_rng(3))
    X = dn.sample_sem(W, 400, rng=np.random.default_rng(4))
    Sigma = torch.tensor(dn.gram(X)).to(dn.DEV)
    model = dn.DAGMANet(d, K=K).to(dn.DEV)
    with torch.no_grad():
        Wout = model(Sigma)
    assert Wout.shape == (d, d)
    assert torch.isfinite(Wout).all()
    assert torch.allclose(torch.diag(Wout), torch.zeros(d, dtype=Wout.dtype, device=Wout.device), atol=1e-9)


# ----------------------------------------------------------------------------- optimization gate
def _tiny_sets(seed=0, d=6, n=400, ntr=30, nva=10, K=12):
    tr = dn.make_instances(d, n, ntr, seed=seed + 1)
    va = dn.make_instances(d, n, nva, seed=seed + 2)
    return d, K, tr, va


def _val_mse(model, val_set):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for Sigma, Wst, _ in val_set:
            W = model(Sigma.to(dn.DEV))
            tot += float(((W - Wst.to(dn.DEV))**2).sum())
    return tot / len(val_set)


@torch_required
def test_gradients_flow():
    """The deep-supervision loss must produce finite, nonzero gradients (not vanishing/NaN)."""
    torch.manual_seed(0)
    d, K, tr, _ = _tiny_sets()
    model = dn.DAGMANet(d, K=K).to(dn.DEV)
    learnable = [p for p in model.parameters() if p.requires_grad]
    wts = torch.arange(1, K + 1, dtype=torch.float64, device=dn.DEV); wts = wts / wts.sum()
    Sigma, Wst, _ = tr[0]
    Wt = Wst.to(dn.DEV)
    W, ws = model(Sigma.to(dn.DEV), collect=True)
    loss = sum(wts[k] * ((ws[k] - Wt)**2).sum() for k in range(K))
    loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in learnable if p.grad is not None)
    assert np.isfinite(gnorm), "gradient is non-finite"
    assert gnorm > 0, "gradient vanished to exactly zero — no learning signal"


@torch_required
def test_training_does_not_diverge():
    """Short training must stay finite and never make val worse (catches divergence/NaN)."""
    torch.manual_seed(0)
    d, K, tr, va = _tiny_sets()
    model = dn.DAGMANet(d, K=K).to(dn.DEV)
    init_val = _val_mse(model, va)
    model, hist = train_short(model, tr, va, epochs=15)
    best_val = min(hist["val"])
    assert np.isfinite(best_val)
    assert best_val <= init_val * 1.01, f"training worsened val: {init_val:.3f} -> {best_val:.3f}"


@torch_required
def test_training_reduces_loss():
    """THE GATE: a short train run should improve val MSE by >= 5%. Turn this green while tuning."""
    torch.manual_seed(0)
    d, K, tr, va = _tiny_sets()
    model = dn.DAGMANet(d, K=K).to(dn.DEV)
    init_val = _val_mse(model, va)
    model, hist = train_short(model, tr, va, epochs=30)
    best_val = min(hist["val"])
    assert best_val < 0.95 * init_val, (
        f"training did not reduce val loss enough: {init_val:.3f} -> {best_val:.3f} "
        f"(need < {0.95 * init_val:.3f}). Deep supervision may not be moving the schedule params."
    )


def train_short(model, tr, va, epochs):
    """Thin wrapper around dn.train_dagmanet with patience high enough to run the full budget."""
    return dn.train_dagmanet(model, tr, va, epochs=epochs, patience=epochs, verbose=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
