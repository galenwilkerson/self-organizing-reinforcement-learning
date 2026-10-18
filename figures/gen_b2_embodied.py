#!/usr/bin/env python3
"""
Exp B2 -- embodied learner comparison (EXPERIMENT_DESIGN_B.md B2; referee M3).

Remove the disembodiment confound: put a learner in the SAME spatial foraging environment as the swarm,
with the SAME partial observability and movement primitives. The ONLY difference is fixed-vs-learned control.

Shared environment (torus, N agents, local sensing r, Pareto-clumped depleting+respawning resources, finite
horizon). Action repertoire (identical for both controllers):
    0 = explore  (Levy step)      1 = approach (step toward nearest sensed signal)      2 = stay (ARS)
Local observation (identical sensing): o = (on_food, sees_source_within_S, neighbour-foraging-fraction bin).

Controllers:
  * SWARM  = fixed recruitment+blocking rule expressed in the shared action set:
      on_food -> stay ; else if neighbour-foraging-fraction >= theta -> approach (recruited; blocking = the
      fraction falls when crowded by non-foragers) ; else if sees_source -> approach ; else -> explore.
      Reward-free.
  * LEARNER = each agent an independent online RL policy (tabular REINFORCE, per-agent baseline) over the
      SAME 3 actions conditioned on the SAME local obs; reward = per-capita intake received this step.
      (The reward asymmetry -- swarm needs none -- is the point, stated, not hidden.)

Metrics IN-ENV (both): cumulative intake over the horizon (the real performance number); IFD matching slope
+ Nash-gap on the in-env allocation; intake-rate vs time (time-to-plateau); robustness variant (mid-run
resource RELOCATION). B>=32 replicas, bootstrap CI over replicas. CPU. Data -> /mnt/newssd/.../paper2_embodied/.
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/newssd/flocking_ltm_data/paper2_embodied"
os.makedirs(DATA, exist_ok=True)

# shared environment constants (paper scale)
L = 800.0
N = 200
K = 80
C = 10                      # clumps
CLUMP_BETA = 2.2
CLUMP_RADIUS = 30.0
RHO_R = 2.1
PHI = 0.02
THETA = 0.18               # recruitment / blocking fraction threshold (shared obs bin edge too)
CAPACITY = 40.0
CONSUMPTION = 1.0
STEP = 20.0
EXPLOIT_STEP = 3.0
LEVY_ALPHA = 1.0
R_NEIGH = L * math.sqrt(RHO_R / N)      # neighbour sensing radius
S_SRC = L * math.sqrt(PHI / K)          # resource sensing radius
NFRAC_BINS = (0.0, 0.08, THETA, 1.01)   # neighbour-foraging-fraction bins -> 3 bins
N_OBS = 2 * 2 * (len(NFRAC_BINS) - 1)   # on_food x sees_source x nfrac_bin = 12
N_ACT = 3


class Env:
    """Vectorised spatial foraging environment over (R replicas, N agents). Pure environment -- no policy."""
    def __init__(self, R, seed=0, regime=1):
        self.R = R; self.regime = regime
        self.g = np.random.default_rng(seed)
        self.size = np.array([L, L])
        # Pareto clump richness (same construction as mobile_cascade)
        self.centers = self.g.random((R, C, 2)) * self.size
        b = max(CLUMP_BETA - 1.0, 0.15)
        w = self.g.random((R, C)).clip(1e-6) ** (-1.0 / b)
        w = w / w.sum(1, keepdims=True)
        self.clump_idx = np.array([self.g.choice(C, size=K, p=w[r]) for r in range(R)])   # (R,K)
        self.rich = np.zeros((R, C))
        for r in range(R):
            np.add.at(self.rich[r], self.clump_idx[r], 1.0)
        self.pos = self.g.random((R, N, 2)) * self.size
        self._place_sources()
        self.cap = np.full((R, K), CAPACITY)

    def _place_sources(self):
        ctr = np.take_along_axis(self.centers, self.clump_idx[:, :, None], axis=1)   # (R,K,2)
        self.sources = (ctr + self.g.standard_normal((self.R, K, 2)) * CLUMP_RADIUS) % self.size

    def relocate(self):
        self.centers = self.g.random((self.R, C, 2)) * self.size
        self._place_sources(); self.cap[:] = CAPACITY

    def _mi(self, diff):
        return diff - self.size * np.round(diff / self.size)

    def _pair(self, A, B):
        return np.linalg.norm(self._mi(A[:, :, None, :] - B[:, None, :, :]), axis=-1)

    def observe(self):
        dsrc = self._pair(self.pos, self.sources)                # (R,N,K)
        live = self.cap > 0                                      # (R,K)
        on_food = ((dsrc <= S_SRC) & live[:, None, :]).any(2)    # (R,N)
        sees = ((dsrc <= S_SRC)).any(2)                          # sees any source within S
        dnb = self._pair(self.pos, self.pos)                     # (R,N,N)
        di = np.arange(N)
        dnb[:, di, di] = np.inf                                   # exclude self
        adj = dnb <= R_NEIGH
        deg = adj.sum(2).clip(min=1)
        nfrac = (adj & on_food[:, None, :]).sum(2) / deg          # fraction of neighbours on food
        # nearest attractor direction: nearest sensed live source, else nearest on-food neighbour
        nsrc = np.where(live[:, None, :], dsrc, np.inf).argmin(2)             # (R,N) nearest live source idx
        src_tgt = np.take_along_axis(self.sources, nsrc[:, :, None], 1)       # (R,N,2)
        nb_inf = np.where(on_food[:, None, :], dnb, np.inf)
        nnb = nb_inf.argmin(2)
        nb_tgt = np.take_along_axis(self.pos, nnb[:, :, None], 1)
        has_src = live.any(1)  # unused
        src_avail = np.take_along_axis(np.where(live, 1.0, 0.0)[:, None, :].repeat(N, 1), nsrc[:, :, None], 2)[:, :, 0] > 0
        nb_avail = on_food.any(1)[:, None] & (nb_inf.min(2) < np.inf)
        target = np.where(src_avail[:, :, None], src_tgt, nb_tgt)
        self._target = target; self._has_target = src_avail | nb_avail
        nb_bin = np.digitize(nfrac, NFRAC_BINS[1:-1])            # 0,1,2
        obs = (on_food.astype(int) * 2 + sees.astype(int)) * (len(NFRAC_BINS) - 1) + nb_bin
        return obs, on_food, sees, nfrac

    def _levy_len(self):
        a = LEVY_ALPHA
        num = (math.gamma(1 + a) * math.sin(math.pi * a / 2)) ** (1 / a)
        den = (math.gamma((1 + a) / 2) * a * 2 ** ((a - 1) / 2)) ** (1 / a)
        sig = num / den
        u = self.g.standard_normal((self.R, N)) * sig
        v = np.abs(self.g.standard_normal((self.R, N)))
        return np.abs(u / v.clip(1e-12) ** (1 / a)) * (STEP / 10.0)

    def step(self, actions, on_food):
        # intake: agents on a live source share its consumption (per-capita = congestion)
        dsrc = self._pair(self.pos, self.sources)
        live = self.cap > 0
        on = (dsrc <= S_SRC) & live[:, None, :]                  # (R,N,K)
        n_on = on.sum(1).clip(min=1)                             # (R,K)
        if self.regime == 1:
            consumed = np.minimum(self.cap, CONSUMPTION * on.sum(1))   # (R,K) total removed
            self.cap = (self.cap - consumed).clip(min=0)
            per_src_share = consumed / n_on                      # (R,K) per-agent share
            intake = (on * per_src_share[:, None, :]).sum(2)     # (R,N) sum over sources the agent is on
            # respawn depleted
            dep = self.cap <= 0
            if dep.any():
                self._place_sources_dep(dep)
        else:
            # non-congestion: fixed per-source reward (no depletion, no sharing) -> optimum concentrate
            intake = (on * 1.0).sum(2) * (CAPACITY / 40.0)       # each source gives fixed reward if present
        # movement
        ll = self._levy_len()
        ang = self.g.random((self.R, N)) * 2 * math.pi
        rdir = np.stack([np.cos(ang), np.sin(ang)], -1)
        explore_step = rdir * ll[:, :, None]
        tdir = self._mi(self._target - self.pos)
        tnorm = np.linalg.norm(tdir, axis=-1, keepdims=True).clip(1e-9)
        approach_step = np.where(self._has_target[:, :, None], tdir / tnorm * ll[:, :, None], explore_step)
        stay_step = self.g.standard_normal((self.R, N, 2)) * EXPLOIT_STEP
        step_vec = np.where((actions == 0)[:, :, None], explore_step,
                   np.where((actions == 1)[:, :, None], approach_step, stay_step))
        self.pos = (self.pos + step_vec) % self.size
        return intake

    def _place_sources_dep(self, dep):
        ctr = np.take_along_axis(self.centers, self.clump_idx[:, :, None], axis=1)
        new = (ctr + self.g.standard_normal((self.R, K, 2)) * CLUMP_RADIUS) % self.size
        self.sources = np.where(dep[:, :, None], new, self.sources)
        self.cap = np.where(dep, CAPACITY, self.cap)

    def occupancy(self):
        """committed foragers assigned to nearest clump centre (within 2*clump_radius); explorers in
        transit are excluded, mirroring the active-agent assignment used for the swarm's IFD slope."""
        d = self._pair(self.pos, self.centers)                   # (R,N,C)
        near = d.argmin(2); dmin = d.min(2)
        comm = (dmin <= 2 * CLUMP_RADIUS).astype(float)
        occ = np.zeros((self.R, C))
        for r in range(self.R):
            np.add.at(occ[r], near[r], comm[r])
        return occ


def swarm_action(obs, on_food, sees, nfrac):
    """Fixed recruitment+blocking policy in the shared action set."""
    a = np.zeros_like(on_food, dtype=int)
    a[:] = 0                                   # default explore
    a = np.where(sees, 1, a)                    # sees source -> approach
    a = np.where(nfrac >= THETA, 1, a)          # recruited -> approach (blocking via the fraction)
    a = np.where(on_food, 2, a)                 # on food -> stay (exploit)
    return a


class Learner:
    """Independent per-agent tabular softmax REINFORCE with per-agent EMA baseline."""
    def __init__(self, R, lr=0.3, beta=0.02, seed=1):
        self.theta = np.zeros((R, N, N_OBS, N_ACT))
        self.base = np.zeros((R, N))
        self.lr = lr; self.beta = beta
        self.g = np.random.default_rng(seed)
        self._last = None

    def act(self, obs):
        R = obs.shape[0]
        th = self.theta[np.arange(R)[:, None], np.arange(N)[None, :], obs]    # (R,N,A)
        th = th - th.max(2, keepdims=True)
        p = np.exp(th); p = p / p.sum(2, keepdims=True)
        gum = -np.log(-np.log(self.g.random((R, N, N_ACT)).clip(1e-12, 1)))
        a = (np.log(p.clip(1e-12, 1)) + gum).argmax(2)
        self._last = (obs, a, p)
        return a

    def update(self, reward):
        obs, a, p = self._last
        R = obs.shape[0]
        adv = reward - self.base
        self.base = (1 - self.beta) * self.base + self.beta * reward
        adv_n = adv / (adv.std(1, keepdims=True) + 1e-8)
        oh = np.zeros((R, N, N_ACT)); np.put_along_axis(oh, a[:, :, None], 1.0, 2)
        grad = adv_n[:, :, None] * (oh - p)                                   # (R,N,A)
        ridx = np.arange(R)[:, None]; nidx = np.arange(N)[None, :]
        self.theta[ridx, nidx, obs] += self.lr * grad


def slope_pooled(rich, occ, idx=None):
    r = (rich[idx] if idx is not None else rich).flatten()
    o = (occ[idx] if idx is not None else occ).flatten()
    m = (r >= 1) & (o >= 0.5)
    return np.polyfit(np.log(r[m]), np.log(o[m]), 1)[0] if m.sum() >= 8 else np.nan


def nash_gap(rich, occ):
    q = rich; x = occ
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(x > 0, q / x, 0.0)
    f_dev = q / (x + 1.0)
    best = f_dev.max(1, keepdims=True)
    imp = np.clip(best - f, 0, None)
    n = occ / occ.sum(1, keepdims=True).clip(min=1e-9)
    return (n * imp).sum(1)


def run(controller, R=48, T=600, regime=1, relocate_at=None, seed=0, lr=0.3):
    env = Env(R, seed=seed, regime=regime)
    learner = Learner(R, lr=lr, seed=seed + 100) if controller == "learner" else None
    intake_rate = np.zeros(T)
    cum = np.zeros((R,))
    occ_tail = np.zeros((R, C)); nmeas = 0
    occ_series = np.zeros((T, R, C))
    for t in range(T):
        if relocate_at is not None and t == relocate_at:
            env.relocate()
        obs, on_food, sees, nfrac = env.observe()
        if controller == "swarm":
            a = swarm_action(obs, on_food, sees, nfrac)
        else:
            a = learner.act(obs)
        intake = env.step(a, on_food)
        if controller == "learner":
            learner.update(intake)
        intake_rate[t] = intake.sum(1).mean()      # mean over replicas of total intake this tick
        cum += intake.sum(1)
        o = env.occupancy(); occ_series[t] = o
        if t >= T // 2:
            occ_tail += o; nmeas += 1
    occ_tail /= max(nmeas, 1)
    return dict(rich=env.rich, occ=occ_tail, occ_series=occ_series,
                intake_rate=intake_rate, cum=cum)


def main(R=48, T=600):
    cache = os.path.join(DATA, f"b2_R{R}_T{T}.npz")
    if os.path.exists(cache) and os.environ.get("FORCE", "") != "1":
        print(f"loading {cache}"); z = np.load(cache, allow_pickle=True); res = z["res"].item()
    else:
        t0 = time.time(); res = {}
        print("Regime 1 (congestion):")
        res["sw1"] = run("swarm", R, T, regime=1, seed=0); print(f"  swarm done ({time.time()-t0:.0f}s)")
        res["ln1"] = run("learner", R, T, regime=1, seed=0); print(f"  learner done ({time.time()-t0:.0f}s)")
        print("Regime 2 (non-congestion, fixed reward):")
        res["sw2"] = run("swarm", R, T, regime=2, seed=0)
        res["ln2"] = run("learner", R, T, regime=2, seed=0); print(f"  R2 done ({time.time()-t0:.0f}s)")
        print("Relocation (congestion, relocate@T/2):")
        res["swR"] = run("swarm", R, T, regime=1, relocate_at=T // 2, seed=0)
        res["lnR"] = run("learner", R, T, regime=1, relocate_at=T // 2, seed=0)
        np.savez_compressed(cache, res=res); print(f"wrote {cache} ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(3); draws = rng.integers(0, R, size=(3000, R))
    def report(tag, d):
        s = slope_pooled(d["rich"], d["occ"])
        ng = nash_gap(d["rich"], d["occ"]).mean()
        cm = d["cum"].mean()
        clo, chi = np.percentile(d["cum"][draws].mean(1), [2.5, 97.5])
        print(f"  {tag}: cum-intake={cm:.0f} [{clo:.0f},{chi:.0f}]  IFD-slope={s:.3f}  Nash-gap={ng:.3f}")
        return dict(cum=cm, clo=clo, chi=chi, slope=s, nash=ng)
    print("\n=== B2 Regime 1 (congestion) ===")
    r_sw = report("swarm  ", res["sw1"]); r_ln = report("learner", res["ln1"])
    print("=== B2 Regime 2 (non-congestion, fixed reward: optimum=concentrate) ===")
    r_sw2 = report("swarm  ", res["sw2"]); r_ln2 = report("learner", res["ln2"])
    ratio = r_ln["cum"] / r_sw["cum"]
    print(f"\n  R1 learner/swarm cumulative-intake ratio = {ratio:.3f}")
    make_figure(res, draws, R, T)
    return res


def make_figure(res, draws, R, T):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5})
    t = np.arange(T)
    C_SW, C_LN = "#2ca02c", "#1f77b4"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))

    # (a) cumulative intake vs time (R1)
    def cumtime(d):
        ir = d["intake_rate"]
        return np.cumsum(ir)
    ax[0].plot(t, cumtime(res["sw1"]), color=C_SW, lw=2.2, label="swarm (reward-free)")
    ax[0].plot(t, cumtime(res["ln1"]), color=C_LN, lw=2.2, label="embodied learner (REINFORCE)")
    ax[0].set_xlabel("time (ticks)"); ax[0].set_ylabel("cumulative intake (mean over replicas)")
    ax[0].set_title("(a) in-env performance, congestion", fontsize=10.8); ax[0].legend(loc="upper left")

    # (b) intake RATE vs time (time-to-plateau), smoothed
    def smooth(x, w=15):
        return np.convolve(x, np.ones(w) / w, mode="same")
    ax[1].plot(t, smooth(res["sw1"]["intake_rate"]), color=C_SW, lw=2, label="swarm")
    ax[1].plot(t, smooth(res["ln1"]["intake_rate"]), color=C_LN, lw=2, label="learner")
    ax[1].set_xlabel("time (ticks)"); ax[1].set_ylabel("intake rate / tick (smoothed)")
    ax[1].set_title("(b) time-to-plateau", fontsize=10.8); ax[1].legend()

    # (c) relocation robustness: intake rate with relocation event
    ax[2].plot(t, smooth(res["swR"]["intake_rate"]), color=C_SW, lw=2, label="swarm")
    ax[2].plot(t, smooth(res["lnR"]["intake_rate"]), color=C_LN, lw=2, label="learner")
    ax[2].axvline(T // 2, color="k", ls=":", lw=1.2); ax[2].text(T // 2 + 5, ax[2].get_ylim()[1] * 0.1,
                                                                  "relocation", fontsize=9)
    ax[2].set_xlabel("time (ticks)"); ax[2].set_ylabel("intake rate / tick (smoothed)")
    ax[2].set_title("(c) robustness: mid-run relocation", fontsize=10.8); ax[2].legend()
    fig.suptitle("Exp B2 — embodied like-for-like: fixed reward-free swarm vs online RL learner in the SAME "
                 "spatial environment", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(HERE, f"fig_b2_embodied.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None); print("wrote", p)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "smoke":
        d = run("swarm", R=8, T=120); print("swarm cum", d["cum"].mean(), "slope", slope_pooled(d["rich"], d["occ"]))
        d = run("learner", R=8, T=120); print("learner cum", d["cum"].mean(), "slope", slope_pooled(d["rich"], d["occ"]))
    else:
        main()
