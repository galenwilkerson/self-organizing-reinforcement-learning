#!/usr/bin/env python3
"""
Discrimination Exp 1 -- model competition for the swarm's coarse-grained patch-flow law.
(paper2_rl_v2, DISCRIMINATION_EXPERIMENT_DESIGN.md Exp 1.)

Referee point (verified): on equal-sharing congestion payoff f_i=Q_i/(N n_i), the replicator regressor
x_i=n_i(f_i-f_bar) is ALGEBRAICALLY the IFD gap: x_i=(Q_tot/N)(n_i*-n_i), n_i*=Q_i/Q_tot. So
corr(x_i, n_i*-n_i)~0.997 and "positive slope of n-dot vs x" only proves convergence to the IFD; it
cannot separate replicator from any convergent dynamics. => rep and linear-relaxation are ONE class here.
But that class (LINEAR in the gap) IS separable from best-response/logit and from payoff-blind contagion.

Candidate mean-field laws for the per-patch flow n-dot_i (fit free params):
  M_rep/relax : n-dot = lambda (n_i* - n_i)                      [1]  rep == linear relaxation (equal sharing)
  M_logit     : n-dot = kappa (sigma_i - n_i), sigma=softmax_b(f) [2]  best-response (payoff-sensitive)
  M_contagion : n-dot = alpha n_i (c Q_i - n_i)                   [2]  crowd-driven logistic + blocking cap
  M_const     : n-dot = const                                    [1]  null / noise floor

Per-tick n-dot is noise-dominated (all models R^2~0.03-0.04); we fit the DENOISED mean flow:
bin (patch,tick,replica) by richness-stratum x occupancy-bin, fit per-bin MEAN n-dot (weighted by count),
TRANSIENT ticks 1-150, occupied real patches. Model selection by leave-replicas-out CV (held-out bin-mean
R^2) + AIC/BIC. Reuses the B=64 bridge cache. CPU.

Outputs: figures/fig_discrim_modelcomp.{pdf,png} and printed numbers for DISCRIM_RESULTS.md.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = "/mnt/newssd/flocking_ltm_data/paper2_bridge/bridge_B64_T800_seed0.npz"
N = 200
LO, HI = 1, 150
NSTRAT, NBIN = 4, 8
N_FOLDS = 60
TEST_R = 16
BETA_GRID = np.concatenate([[0.0], np.geomspace(0.05, 100, 40)])


def load():
    z = np.load(BRIDGE)
    rich = z["rich"].astype(float)
    count = z["count"].astype(float)
    T, R, K = count.shape
    Q = np.broadcast_to(rich[None], count.shape).astype(float)
    n = count / N
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(count > 0, Q / count, 0.0)
    fbar = np.where(count > 0, Q, 0.0).sum(2) / N
    Qtot = np.where(count > 0, Q, 0.0).sum(2)
    nstar = np.where(Qtot[:, :, None] > 0, Q / Qtot[:, :, None], 0.0)
    dn = np.full_like(n, np.nan); dn[:-1] = n[1:] - n[:-1]
    occ_mask = (count > 0)
    # committed mass per (t,r) over occupied real patches (for logit sigma scaling)
    real = Q >= 1
    m_mass = np.where(occ_mask & real, n, 0.0).sum(2)                      # (T,R)
    return dict(T=T, R=R, K=K, Q=Q, n=n, f=f, fbar=fbar, nstar=nstar, dn=dn,
                occ_mask=occ_mask, real=real, m_mass=m_mass, rich=rich)


def sigma_logit(D, beta):
    """softmax_beta(f) over occupied real patches, scaled to the committed mass. (T,R,K)."""
    f = D["f"]; mask = D["occ_mask"] & D["real"]
    z = np.where(mask, beta * f, -np.inf)
    z = z - np.where(np.isfinite(z), z, -np.inf).max(2, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    s = e / e.sum(2, keepdims=True).clip(min=1e-12)
    return s * D["m_mass"][:, :, None]


def valid_mask(D):
    v = np.zeros_like(D["n"], bool); v[LO:HI] = True
    v &= D["occ_mask"] & D["real"] & np.isfinite(D["dn"]) & np.isfinite(D["n"])
    return v


# ---------- bin-mean weighted fitting ----------
def make_bins(Qv, nv):
    qs = np.quantile(Qv, np.linspace(0, 1, NSTRAT + 1)); qs[-1] += 1e-9
    ns = np.quantile(nv, np.linspace(0, 1, NBIN + 1)); ns[-1] += 1e-9
    return qs, ns


def binkey(Qv, nv, qs, ns):
    qb = np.clip(np.digitize(Qv, qs) - 1, 0, NSTRAT - 1)
    nb = np.clip(np.digitize(nv, ns) - 1, 0, NBIN - 1)
    return qb * NBIN + nb


def binmeans(key, weights_cols, nbins):
    """Return per-bin count and per-bin mean of each column in weights_cols (list of arrays)."""
    cnt = np.bincount(key, minlength=nbins).astype(float)
    means = [np.bincount(key, weights=c, minlength=nbins) / np.maximum(cnt, 1) for c in weights_cols]
    return cnt, means


def wls_no_intercept(X, y, w):
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * np.sqrt(w), rcond=None)
    return beta


def r2_weighted(y, yhat, w):
    ss_res = (w * (y - yhat) ** 2).sum()
    ss_tot = (w * (y - np.average(y, weights=w)) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def model_features(D, v, beta=None):
    """Return dict of per-sample feature matrices for each model at valid samples v."""
    n = D["n"][v]; Q = D["Q"][v]; gap = (D["nstar"] - D["n"])[v]
    feats = {
        "rep/relax": np.stack([gap], 1),
        "contagion": np.stack([n * Q, n * n], 1),
        "const": np.ones((v.sum(), 1)),
    }
    if beta is not None:
        sig = sigma_logit(D, beta)[v]
        feats["logit"] = np.stack([sig - n], 1)
    return feats


def fit_eval(D, train_v, test_v, qs, ns, beta_fit=True):
    """Fit each model on train bin-means, return held-out R^2 on test bin-means."""
    nbins = NSTRAT * NBIN
    # training bin assignment uses GLOBAL edges
    out = {}
    # precompute keys
    def prep(v):
        return binkey(D["Q"][v], D["n"][v], qs, ns), D["dn"][v]
    ktr, ytr = prep(train_v); kte, yte = prep(test_v)
    cnt_tr, (ym_tr,) = binmeans(ktr, [ytr], nbins)
    cnt_te, (ym_te,) = binmeans(kte, [yte], nbins)
    # fit beta for logit on train (grid minimizing train weighted SSE)
    best_beta, best_sse = 0.0, np.inf
    if beta_fit:
        for b in BETA_GRID:
            feats = model_features(D, train_v, beta=b)["logit"][:, 0]
            _, (fm,) = binmeans(ktr, [feats], nbins)
            mtr = cnt_tr > 0
            X = fm[mtr][:, None]; beta = wls_no_intercept(X, ym_tr[mtr], cnt_tr[mtr])
            sse = (cnt_tr[mtr] * (ym_tr[mtr] - X @ beta) ** 2).sum()
            if sse < best_sse:
                best_sse, best_beta = sse, b
    # fit each model on train bin-means, score on test bin-means
    for name in ["rep/relax", "contagion", "const", "logit"]:
        b = best_beta if name == "logit" else None
        feat_tr = model_features(D, train_v, beta=b)[name]
        feat_te = model_features(D, test_v, beta=b)[name]
        ncol = feat_tr.shape[1]
        cols_tr = [feat_tr[:, j] for j in range(ncol)]
        cols_te = [feat_te[:, j] for j in range(ncol)]
        _, fm_tr = binmeans(ktr, cols_tr, nbins)
        _, fm_te = binmeans(kte, cols_te, nbins)
        mtr = cnt_tr > 0; mte = cnt_te > 0
        if name == "const":
            beta = np.array([np.average(ym_tr[mtr], weights=cnt_tr[mtr])])
            Xte = np.ones((mte.sum(), 1))
        else:
            Xtr = np.stack([c[mtr] for c in fm_tr], 1)
            beta = wls_no_intercept(Xtr, ym_tr[mtr], cnt_tr[mtr])
            Xte = np.stack([c[mte] for c in fm_te], 1)
        yhat = Xte @ beta
        out[name] = r2_weighted(ym_te[mte], yhat, cnt_te[mte])
    out["_beta_logit"] = best_beta
    return out


def main():
    D = load()
    v = valid_mask(D)
    print(f"valid transient occupied samples: {v.sum()}")
    Qv, nv = D["Q"][v], D["n"][v]
    qs, ns = make_bins(Qv, nv)

    # ---- leave-replicas-out CV ----
    rng = np.random.default_rng(0)
    R = D["R"]
    rep_id = np.broadcast_to(np.arange(R)[None, :, None], D["n"].shape)
    rep_v = rep_id[v]
    scores = {k: [] for k in ["rep/relax", "logit", "contagion", "const"]}
    betas = []
    for fold in range(N_FOLDS):
        test_reps = set(rng.choice(R, TEST_R, replace=False).tolist())
        test_sel = np.isin(rep_v, list(test_reps))
        # build boolean masks over full array for train/test
        is_test = np.isin(rep_id, list(test_reps))
        train_v = v & (~is_test)
        test_vv = v & is_test
        res = fit_eval(D, train_v, test_vv, qs, ns)
        for k in scores:
            scores[k].append(res[k])
        betas.append(res["_beta_logit"])
    print("\n=== leave-16-replicas-out CV held-out bin-mean R^2 (mean [2.5,97.5] over %d folds) ===" % N_FOLDS)
    cv = {}
    for k in ["contagion", "rep/relax", "logit", "const"]:
        a = np.array(scores[k]); cv[k] = (a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5))
        print(f"  {k:10s}: {cv[k][0]:+.3f}  [{cv[k][1]:+.3f}, {cv[k][2]:+.3f}]")
    print(f"  logit beta (median over folds) = {np.median(betas):.3f}")

    # ---- in-sample AIC/BIC on all valid data (bin-mean level) ----
    print("\n=== in-sample bin-mean fit + AIC/BIC (n_bins effective points) ===")
    nbins = NSTRAT * NBIN
    key = binkey(Qv, nv, qs, ns); cnt, (ym,) = binmeans(key, [D["dn"][v]], nbins)
    mb = cnt > 0; neff = mb.sum()
    # refit logit beta in-sample
    bb, bsse = 0.0, np.inf
    for b in BETA_GRID:
        fe = model_features(D, v, beta=b)["logit"][:, 0]
        _, (fm,) = binmeans(key, [fe], nbins)
        X = fm[mb][:, None]; be = wls_no_intercept(X, ym[mb], cnt[mb])
        sse = (cnt[mb] * (ym[mb] - X @ be) ** 2).sum()
        if sse < bsse: bsse, bb = sse, b
    aic = {}
    fit_params = {}
    for name, k in [("contagion", 2), ("rep/relax", 1), ("logit", 2), ("const", 1)]:
        b = bb if name == "logit" else None
        feat = model_features(D, v, beta=b)[name]
        cols = [feat[:, j] for j in range(feat.shape[1])]
        _, fm = binmeans(key, cols, nbins)
        if name == "const":
            be = np.array([np.average(ym[mb], weights=cnt[mb])]); X = np.ones((mb.sum(), 1))
        else:
            X = np.stack([c[mb] for c in fm], 1); be = wls_no_intercept(X, ym[mb], cnt[mb])
        yhat = X @ be
        r2 = r2_weighted(ym[mb], yhat, cnt[mb])
        sse = (cnt[mb] * (ym[mb] - yhat) ** 2).sum() / cnt[mb].sum() * neff  # scale to neff points
        aicv = neff * np.log(sse / neff) + 2 * k
        bicv = neff * np.log(sse / neff) + k * np.log(neff)
        aic[name] = (r2, aicv, bicv, k)
        fit_params[name] = be
        print(f"  {name:10s}: in-sample R^2={r2:+.3f}  AIC={aicv:7.1f}  BIC={bicv:7.1f}  k={k}  beta={be}")
    best_aic = min(a[1] for a in aic.values())
    print("  dAIC:", {k: round(a[1] - best_aic, 1) for k, a in aic.items()})
    print(f"  logit beta (in-sample, saturates for beta>~10) = {bb:.2f}")

    # ---- per-sample predictions for the figure ----
    preds = {}
    for name in ["rep/relax", "contagion", "logit", "const"]:
        b = bb if name == "logit" else None
        feat = model_features(D, v, beta=b)[name]
        if name == "const":
            preds[name] = np.full(v.sum(), fit_params["const"][0])
        else:
            preds[name] = feat @ fit_params[name]

    # ---- small-n diagnostic ----
    print("\n=== small-n diagnostic (lowest occupancy bin, per richness stratum) ===")
    qb = np.clip(np.digitize(Qv, qs) - 1, 0, NSTRAT - 1)
    nb = np.clip(np.digitize(nv, ns) - 1, 0, NBIN - 1)
    dnv = D["dn"][v]; gapv = (D["nstar"] - D["n"])[v]
    for st in range(NSTRAT):
        sel = (qb == st) & (nb == 0)
        if sel.sum() < 20: continue
        print(f"  stratum {st} (Q~{Qv[qb==st].mean():.1f}), smallest-n bin: n~{nv[sel].mean():.4f}  "
              f"observed n-dot={dnv[sel].mean():+.2e}  gap={gapv[sel].mean():+.2e}  "
              f"(rep predicts n-dot proportional to gap>0; contagion predicts ~0)")

    make_figure(D, v, qs, ns, preds, cv)
    return cv, aic


def make_figure(D, v, qs, ns, preds, cv):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    Qv, nv, dnv = D["Q"][v], D["n"][v], D["dn"][v]
    qb = np.clip(np.digitize(Qv, qs) - 1, 0, NSTRAT - 1)
    mcols = {"rep/relax": "#c0392b", "contagion": "#2c7fb8", "logit": "#8e44ad"}
    mlab = {"rep/relax": "rep/relax $\\lambda(n^*{-}n)$",
            "contagion": "contagion $\\alpha n(cQ{-}n)$",
            "logit": "best-response $\\kappa(\\sigma_\\beta(f){-}n)$"}

    fig = plt.figure(figsize=(13.5, 3.7))
    strata_to_show = [0, NSTRAT // 2, NSTRAT - 1]
    for pi, st in enumerate(strata_to_show):
        ax = fig.add_subplot(1, 4, pi + 1)
        sel = qb == st
        Qbar = Qv[sel].mean()
        ns_edges = np.quantile(nv[sel], np.linspace(0, 1, NBIN + 1)); ns_edges[-1] += 1e-9
        nb = np.clip(np.digitize(nv[sel], ns_edges) - 1, 0, NBIN - 1)
        def binxy(vals):
            xs, ys = [], []
            for b in range(NBIN):
                s2 = nb == b
                if s2.sum() < 10: continue
                xs.append(nv[sel][s2].mean()); ys.append(vals[sel][s2].mean())
            return np.array(xs), np.array(ys)
        xo, yo = binxy(dnv)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.scatter(xo, yo, s=34, color="k", zorder=6, label="observed")
        for name in ["rep/relax", "contagion", "logit"]:
            xp, yp = binxy(preds[name])
            ax.plot(xp, yp, "-o", ms=3, color=mcols[name], lw=1.8, label=mlab[name])
        ax.set_title(f"richness $Q\\approx{Qbar:.0f}$", fontsize=10.5)
        ax.set_xlabel("occupancy $n_i$")
        if pi == 0:
            ax.set_ylabel("mean flow $\\dot n_i$")
            ax.legend(loc="upper right", fontsize=8)
    # held-out R^2 bars
    axb = fig.add_subplot(1, 4, 4)
    order = ["contagion", "rep/relax", "logit", "const"]
    cols = ["#2c7fb8", "#c0392b", "#8e44ad", "#7f8c8d"]
    vals = [cv[k][0] for k in order]
    lo = [cv[k][0] - cv[k][1] for k in order]; hi = [cv[k][2] - cv[k][0] for k in order]
    axb.bar(range(len(order)), vals, yerr=[lo, hi], capsize=3, color=cols)
    axb.set_xticks(range(len(order))); axb.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
    axb.set_ylabel("held-out CV $R^2$ (bin-mean)")
    axb.set_title("leave-16-replicas-out", fontsize=10.5)
    axb.axhline(0, color="k", lw=0.5)
    fig.suptitle("Exp 1 — model competition for the swarm's mean patch-flow (transient, occupied real patches)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_discrim_modelcomp.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    main()
