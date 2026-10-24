#!/usr/bin/env python3
"""GPU-batched torch port of the Paper-2 embodied foraging env, F1 sensing-limited (step3_rl_proper/
env_proper.py + gen_b2_embodied.py). Vectorised over W worlds, each with its OWN operating point
(phi->S_SRC, patchiness->clump_radius, theta, gamma) so seeds AND operating points fold into the batch.

Faithfulness is the whole point (this is the comparison env): the physics, constants, per-capita congestion
intake, clumped Pareto resources, deplete/respawn, F1 sensing-limited primitives, multi-hop cascade
(parent-centroid heading) and occupancy/oracle all mirror the numpy env. Validated by `python torch_env.py`
against the numpy reference numbers (swarm ~0.285 / random ~0.19 @ PAPER).

Canonical Code/ and Paper-2 files UNTOUCHED.
"""
from __future__ import annotations
import math, os
import torch

# --- constants (match gen_b2_embodied) ---
# N-scaling copy: N,K,C,L all read from env (defaults == unchanged behaviour). TRUE finite-size scaling grows
# the ARENA L ∝ √N with r (=R_NEIGH below), STEP, EXPLOIT_STEP, clump_radius, S all held FIXED and the counts
# N,K,C scaled with area (∝N). Then R_NEIGH = L√(rho_r/N) is automatically constant, so rho_r AND every length
# ratio (STEP/r, clump_radius/r, S/r) and density stay fixed -- only the absolute system size grows.
L = float(os.environ.get("FLOCKING_L", "800")); N = int(os.environ.get("FLOCKING_N", "200"))
K = int(os.environ.get("FLOCKING_K", "80")); C = int(os.environ.get("FLOCKING_C", "10"))
CLUMP_BETA = 2.2; RHO_R = 2.1
STEP = 20.0; EXPLOIT_STEP = 3.0; LEVY_ALPHA = 1.0
CAPACITY = 40.0; CONSUMPTION = 1.0
F_OBS = 14
R_NEIGH = L * math.sqrt(RHO_R / N)        # fixed by rho_r, N


class TorchForageEnv:
    def __init__(self, W, phi, patchiness, theta, gamma, seed=0, device="cuda", dtype=torch.float32):
        self.W = W; self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.dtype = dtype
        self.g = torch.Generator(device=self.device); self.g.manual_seed(seed)
        self.size = torch.tensor([L, L], device=self.device, dtype=dtype)
        # per-world operating point (each a (W,) tensor)
        tt = lambda x: (x if torch.is_tensor(x) else torch.as_tensor(x)).to(self.device, dtype).expand(W).clone()
        self.phi = tt(phi); self.patch = tt(patchiness); self.theta = tt(theta); self.gamma = tt(gamma)
        self.S = (L * torch.sqrt(self.phi / K)).view(W, 1, 1)             # source-sense radius per world
        self.clump_radius = (self.patch * L).view(W, 1, 1)               # per world
        self.r_neigh = R_NEIGH
        self.reset(seed)

    # ---------- rng helpers ----------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.g, device=self.device, dtype=self.dtype)

    def reset(self, seed=None):
        if seed is not None:
            self.g.manual_seed(seed)
        W = self.W
        self.centers = self._rand(W, C, 2) * self.size
        b = max(CLUMP_BETA - 1.0, 0.15)
        w = self._rand(W, C).clamp(min=1e-6).pow(-1.0 / b); w = w / w.sum(1, keepdim=True)
        self.clump_idx = torch.multinomial(w, K, replacement=True, generator=self.g)      # (W,K)
        self.rich = torch.zeros(W, C, device=self.device, dtype=self.dtype)
        self.rich.scatter_add_(1, self.clump_idx, torch.ones_like(self.clump_idx, dtype=self.dtype))
        self.pos = self._rand(W, N, 2) * self.size
        self._place_sources()
        self.cap = torch.full((W, K), CAPACITY, device=self.device, dtype=self.dtype)
        self._target = self.pos.clone(); self._has_target = torch.zeros(W, N, dtype=torch.bool, device=self.device)

    def _place_sources(self, mask=None):
        ctr = torch.gather(self.centers, 1, self.clump_idx.unsqueeze(-1).expand(-1, -1, 2))   # (W,K,2)
        new = torch.remainder(ctr + self._randn(self.W, K, 2) * self.clump_radius, self.size)
        if mask is None:
            self.sources = new
        else:
            self.sources = torch.where(mask.unsqueeze(-1), new, self.sources)

    def _mi(self, diff):
        return diff - self.size * torch.round(diff / self.size)

    def _pair(self, A, B):
        return self._mi(A.unsqueeze(2) - B.unsqueeze(1)).norm(dim=-1)                          # (W,nA,nB)

    def _levy_len(self):
        a = LEVY_ALPHA
        num = (math.gamma(1 + a) * math.sin(math.pi * a / 2)) ** (1 / a)
        den = (math.gamma((1 + a) / 2) * a * 2 ** ((a - 1) / 2)) ** (1 / a)
        sig = num / den
        u = self._randn(self.W, N) * sig
        v = self._randn(self.W, N).abs()
        return (u / v.clamp(min=1e-12).pow(1 / a)).abs() * (STEP / 10.0)

    def _rand_dir(self):
        ang = self._rand(self.W, N) * (2 * math.pi)
        return torch.stack([ang.cos(), ang.sin()], dim=-1)

    # ---------- observation (F1 sensing-limited; mirrors env_proper.extract_sl) ----------
    def extract(self, intake_ema, onfood_ema):
        S = self.S; W = self.W
        dsrc = self._pair(self.pos, self.sources)                 # (W,N,K)
        live = self.cap > 0                                       # (W,K)
        within = dsrc <= S
        on_food = (within & live.unsqueeze(1)).any(2)             # (W,N)
        sees = within.any(2)
        diag = torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        dnb = self._pair(self.pos, self.pos)
        dnb = dnb.masked_fill(diag, float("inf"))
        adj = dnb <= self.r_neigh
        deg = adj.sum(2).clamp(min=1).to(self.dtype)
        nfrac = (adj & on_food.unsqueeze(1)).sum(2).to(self.dtype) / deg
        BIG = float(L) * 10.0
        dlive = torch.where(live.unsqueeze(1), dsrc, torch.full_like(dsrc, BIG))
        nsrc = dlive.argmin(2); dmin = torch.gather(dlive, 2, nsrc.unsqueeze(-1)).squeeze(-1)
        src_tgt = torch.gather(self.sources, 1, nsrc.unsqueeze(-1).expand(-1, -1, 2))
        src_avail = dmin <= S.squeeze(-1)                         # (W,N): source within S
        nb_d0 = torch.where(on_food.unsqueeze(1) & adj, dnb, torch.full_like(dnb, float("inf")))
        nnb = nb_d0.argmin(2); nb_d = torch.gather(nb_d0, 2, nnb.unsqueeze(-1)).squeeze(-1)
        nb_tgt = torch.gather(self.pos, 1, nnb.unsqueeze(-1).expand(-1, -1, 2))
        nb_avail = torch.isfinite(nb_d)
        self._target = torch.where(src_avail.unsqueeze(-1), src_tgt, nb_tgt)
        self._has_target = src_avail | nb_avail
        Ssq = S.squeeze(-1)
        prox = torch.where(src_avail, torch.exp(-dmin.clamp(0, 5 * Ssq.max()) / Ssq), torch.zeros_like(dmin))
        sdir = self._mi(src_tgt - self.pos); sn = sdir.norm(dim=-1, keepdim=True)
        sudir = torch.where(src_avail.unsqueeze(-1), sdir / sn.clamp(min=1e-9), torch.zeros_like(sdir))
        nprox = torch.where(nb_avail, torch.exp(-nb_d.clamp(0, 5 * self.r_neigh) / self.r_neigh),
                            torch.zeros_like(nb_d))
        ndir = self._mi(nb_tgt - self.pos); nn_ = ndir.norm(dim=-1, keepdim=True)
        nudir = torch.where(nb_avail.unsqueeze(-1), ndir / nn_.clamp(min=1e-9), torch.zeros_like(ndir))
        densS = (within & live.unsqueeze(1)).sum(2).to(self.dtype) / 5.0
        densR = (within & live.unsqueeze(1)).sum(2).to(self.dtype) / 20.0
        degn = deg / 20.0
        feats = torch.stack([on_food.to(self.dtype), sees.to(self.dtype), prox,
                             sudir[..., 0], sudir[..., 1], densS, densR, nfrac, degn,
                             nprox, nudir[..., 0], nudir[..., 1], intake_ema, onfood_ema], dim=-1)
        return on_food, prox, feats

    # ---------- multi-hop cascade (reward-free swarm baseline; parent-centroid heading) ----------
    def cascade(self, on_food):
        dnb = self._pair(self.pos, self.pos)
        diag = torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        dnb = dnb.masked_fill(diag, float("inf"))
        adj = dnb <= self.r_neigh
        deg = adj.sum(2).clamp(min=1).to(self.dtype)
        adj_f = adj.to(self.dtype)
        active = on_food.clone()
        strength = on_food.to(self.dtype)
        gen = torch.where(on_food, torch.zeros(self.W, N, device=self.device, dtype=self.dtype),
                          torch.full((self.W, N), float(N), device=self.device, dtype=self.dtype))
        th = self.theta.view(self.W, 1); gm = self.gamma.view(self.W, 1)
        for rnd in range(1, N + 1):
            a = (adj_f * strength.unsqueeze(1)).sum(2)
            newly = (a / deg >= th) & (~active)
            if not bool(newly.any()):
                break
            active = active | newly
            strength = torch.where(newly, gm.expand(self.W, N) ** float(rnd), strength)
            gen = torch.where(newly, torch.full_like(gen, float(rnd)), gen)
        recruited = active & (~on_food)
        par = adj & active.unsqueeze(1) & (gen.unsqueeze(1) < gen.unsqueeze(2))     # (W,i,j): j parent of i
        diff = self._mi(self.pos.unsqueeze(1) - self.pos.unsqueeze(2))              # (W,i,j,2): i->j
        cnt = par.sum(2).clamp(min=1).unsqueeze(-1).to(self.dtype)
        head = (diff * par.unsqueeze(-1).to(self.dtype)).sum(2) / cnt
        hn = head.norm(dim=-1, keepdim=True)
        head_unit = torch.where(hn > 1e-9, head / hn.clamp(min=1e-9), torch.zeros_like(head))
        return recruited, head_unit

    def swarm_actions(self):
        S = self.S
        dsrc = self._pair(self.pos, self.sources); live = self.cap > 0
        within = dsrc <= S
        on_food = (within & live.unsqueeze(1)).any(2)
        sees = within.any(2)
        dS = torch.where(within & live.unsqueeze(1), dsrc, torch.full_like(dsrc, float("inf")))
        nsrc = dS.argmin(2); sdmin = torch.gather(dS, 2, nsrc.unsqueeze(-1)).squeeze(-1)
        src_tgt = torch.gather(self.sources, 1, nsrc.unsqueeze(-1).expand(-1, -1, 2))
        src_av = torch.isfinite(sdmin)
        recruited, head_unit = self.cascade(on_food)
        a = torch.zeros(self.W, N, dtype=torch.long, device=self.device)
        a = torch.where(sees, torch.ones_like(a), a)
        a = torch.where(recruited, torch.ones_like(a), a)
        a = torch.where(on_food, torch.full_like(a, 2), a)
        tgt = self.pos + head_unit * 10.0
        tgt = torch.where(src_av.unsqueeze(-1), src_tgt, tgt)
        self._target = tgt; self._has_target = src_av | recruited
        return on_food, a, float(recruited.to(self.dtype).mean().item())

    # ---------- step ----------
    def step(self, actions, on_food):
        S = self.S
        dsrc = self._pair(self.pos, self.sources); live = self.cap > 0
        on = (dsrc <= S) & live.unsqueeze(1)                      # (W,N,K) agent on source
        # INTAKE CAP (as in the dimensionless sweep): each agent eats from only its NEAREST live source it is
        # on, so overlapping sources at high phi cannot inflate intake > oracle. 'on_food' detection below is
        # unchanged (any live source within S), so seed/cascade dynamics are intact. At PAPER (low phi,
        # negligible overlap) this leaves the numpy-matched result unchanged.
        dl = torch.where(on, dsrc, torch.full_like(dsrc, float("inf")))
        near = dl.argmin(2)                                       # (W,N) nearest on-source
        oh = torch.zeros_like(on, dtype=self.dtype)
        oh.scatter_(2, near.unsqueeze(-1), 1.0)
        on_cap = oh * on.any(2, keepdim=True).to(self.dtype)      # (W,N,K) one source/agent (0 if on none)
        n_on = on_cap.sum(1).clamp(min=1)                         # (W,K) agents assigned to each source
        consumed = torch.minimum(self.cap, CONSUMPTION * on_cap.sum(1))
        self.cap = (self.cap - consumed).clamp(min=0.0)
        per_src_share = consumed / n_on                           # (W,K)
        intake = (on_cap * per_src_share.unsqueeze(1)).sum(2)     # (W,N)
        dep = self.cap <= 0.0
        if bool(dep.any()):
            self._place_sources(mask=dep)
            self.cap = torch.where(dep, torch.full_like(self.cap, CAPACITY), self.cap)
        # movement
        ll = self._levy_len()
        rdir = self._rand_dir()
        explore = rdir * ll.unsqueeze(-1)
        tdir = self._mi(self._target - self.pos); tn = tdir.norm(dim=-1, keepdim=True)
        approach = torch.where(self._has_target.unsqueeze(-1),
                               tdir / tn.clamp(min=1e-9) * ll.unsqueeze(-1), explore)
        stay = self._randn(self.W, N, 2) * EXPLOIT_STEP
        a = actions.unsqueeze(-1)
        step_vec = torch.where(a == 0, explore, torch.where(a == 1, approach, stay))
        self.pos = torch.remainder(self.pos + step_vec, self.size)
        return intake

    def occupancy(self):
        d = self._pair(self.pos, self.centers)                    # (W,N,C)
        near = d.argmin(2); dmin = d.min(2).values
        comm = (dmin <= 2.0 * self.clump_radius.squeeze(-1)).to(self.dtype)
        occ = torch.zeros(self.W, C, device=self.device, dtype=self.dtype)
        occ.scatter_add_(1, near, comm)
        return occ


def slope_pooled(rich, occ):
    """Pooled IFD slope over a set of worlds (rich,occ: (n,C) numpy)."""
    import numpy as np
    r = rich.flatten(); o = occ.flatten(); m = (r >= 1) & (o >= 0.5)
    if m.sum() < 8:
        return float("nan")
    return float(np.polyfit(np.log(r[m]), np.log(o[m]), 1)[0])


# =================== faithfulness validation (G1, G2) ===================
def _eval(controller, W, ticks, window, op, seed):
    import numpy as np
    env = TorchForageEnv(W, op["phi"], op["patchiness"], op["theta"], op["gamma"], seed=seed)
    ie = torch.zeros(W, N, device=env.device); oe = torch.zeros(W, N, device=env.device)
    gen = torch.Generator(device=env.device); gen.manual_seed(seed + 777)
    hr = torch.zeros(W, device=env.device); hc = torch.zeros(W, device=env.device)
    occ = torch.zeros(W, C, device=env.device); nmeas = 0
    for t in range(ticks):
        capb = env.cap.sum(1)
        if controller == "swarm":
            on, a, _ = env.swarm_actions()
        elif controller == "random":
            on, prox, feats = env.extract(ie, oe)
            a = torch.randint(0, 3, (W, N), generator=gen, device=env.device)
        intake = env.step(a, on)
        ie = 0.9 * ie + 0.1 * intake; oe = 0.95 * oe + 0.05 * on.to(env.dtype)
        if t >= ticks - window:
            hr += intake.sum(1); hc += torch.minimum(torch.tensor(float(N), device=env.device), capb)
            occ += env.occupancy(); nmeas += 1
    frac = (hr / hc.clamp(min=1e-9)).cpu().numpy()
    slope = slope_pooled(env.rich.cpu().numpy(), (occ / max(nmeas, 1)).cpu().numpy())
    return frac, slope


if __name__ == "__main__":
    import time, numpy as np
    PAPER = dict(phi=0.02, patchiness=0.0375, theta=0.18, gamma=1.0)
    FAV = dict(phi=0.08, patchiness=0.02, theta=0.18, gamma=0.8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} R_NEIGH={R_NEIGH:.2f}")
    for name, op in [("PAPER", PAPER), ("FAVORABLE", FAV)]:
        t0 = time.time()
        fs, ss = _eval("swarm", 256, 2000, 1500, op, 0)
        fr, _ = _eval("random", 256, 2000, 1500, op, 0)
        print(f"[{name}] swarm frac={fs.mean():.3f}±{fs.std():.3f} slope={ss:.2f} | "
              f"random frac={fr.mean():.3f} | ({time.time()-t0:.1f}s)  "
              f"[numpy ref PAPER: swarm~0.285 slope~0.96, random~0.19]")
