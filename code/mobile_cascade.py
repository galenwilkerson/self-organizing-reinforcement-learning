"""
Minimal model: a seed-driven mobile threshold cascade on a torus.

The simplest instrument for the fundamental questions --- no oil, no consumption, no
depletion, no foraging task. Just:

  * N agents on a torus; a random geometric graph by proximity (connectivity rho_r).
  * A steady drive: K fixed, NON-DEPLETING sources; agents within detection radius are seeds
    (so seeding is sustained -- no starvation).
  * A linear-threshold cascade to its fixed point (threshold theta).
  * Motion (the one novel ingredient / the topology-geometry coupling): cascade-recruited
    agents step toward their recruiters (densify); everyone else random-walks (disperse).
  * Blocking is emergent: as density rises, degree k rises, the active fraction a/k falls
    below theta, and recruitment self-limits.

Dimensionless controls: rho_r = N (r/L)^2, phi = K (S/L)^2, theta. (No Levy: exploration is a
plain random walk; alpha is not a control.)

Ablation (for Q3): converge=False removes the recruitment->motion coupling (activation without
convergence); frozen_sources has no effect (sources are always fixed).

Observables per tick (B,): sigma (branching ratio), z_local (mean degree of active agents),
z_global, active_frac, cascade_size, exploring_fraction.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import Tensor

from batched_swarm import select_device  # reuse device picker


@dataclass
class MobileCascadeConfig:
    B: int = 256
    N: int = 100
    K: int = 30                 # number of fixed sources (drive)
    L: float = 800.0
    rho_r: float = 2.0          # connectivity density -> radius r
    phi: float = 0.05           # source coverage -> detection radius S
    theta: float = 0.18         # LTM threshold (mean, or the constant value)
    theta_dist: str = "const"   # per-agent threshold law: 'const' | 'gaussian' | 'uniform'
    theta_std: float = 0.0      # gaussian: std of the threshold distribution (mean = theta)
    theta_low: Optional[float] = None   # uniform: low; defaults to theta - theta_std if None
    theta_high: Optional[float] = None  # uniform: high; defaults to theta + theta_std if None
    p: float = 1.0              # edge probability within radius
    step: float = 20.0          # random-walk / Levy exploration step scale (dispersal)
    pull: float = 0.1           # convergence fraction toward recruiters (densify)
    levy_alpha: Optional[float] = None  # if set: exploring (inactive) agents move by Levy(alpha),
                                        # active agents by small Brownian -> the recruit-state mode switch
    exploit_step: float = 3.0   # small Brownian step for active agents (local exploitation)
    consumption: float = 1.0    # source charge consumed per agent-on-source per tick
    capacity: float = 40.0      # source charge; depletes then respawns -> drives the SOC cycle
    clump_count: Optional[int] = None   # if set: sources live in this many spatial clumps (patchy resource,
                                        # so a find predicts more resource nearby -> recruitment informative)
    clump_radius: float = 40.0          # spread of sources within a clump (world units)
    clump_beta: Optional[float] = None  # if set: SCALE-FREE resource field -- clump sizes follow a power law
                                        # with exponent ~clump_beta (Pareto weights over clumps)
    # --- ping economy: how OFTEN each channel is emitted (defaults reproduce every-tick behaviour) ---
    beacon_interval: int = 1        # emit "I am here" every k ticks; graph uses LAST-BEACONED positions
    beacon_move_frac: float = 0.0   # >0: event-driven instead -- beacon after moving this fraction of r
                                    # (self-adapting: the graph decorrelates on ~(r/step)^alpha ticks,
                                    #  so slow/near-Brownian motion can beacon far less often)
    activation_event_only: bool = False  # True: "I found a target" only on the inactive->active
                                    # TRANSITION, not re-broadcast every tick while active
    converge: bool = True       # ablation: False = no recruitment->motion coupling
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None


class MobileCascade:
    def __init__(self, cfg: MobileCascadeConfig):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.dtype = cfg.dtype
        self.B, self.N, self.K = cfg.B, cfg.N, cfg.K
        self.size = torch.tensor([cfg.L, cfg.L], device=self.device, dtype=self.dtype)
        self.r = cfg.L * math.sqrt(cfg.rho_r / cfg.N)        # rho_r = N (r/L)^2
        self.S = cfg.L * math.sqrt(cfg.phi / cfg.K)          # phi   = K (S/L)^2
        self.theta = cfg.theta
        self.theta_dist = cfg.theta_dist
        self.theta_std = cfg.theta_std
        self.theta_low = cfg.theta_low
        self.theta_high = cfg.theta_high
        self.p = cfg.p
        self.step_scale = cfg.step
        self.pull = cfg.pull
        self.consumption = cfg.consumption
        self.capacity = cfg.capacity
        self.converge = cfg.converge
        self.beacon_interval = cfg.beacon_interval
        self.beacon_move_frac = cfg.beacon_move_frac
        self.activation_event_only = cfg.activation_event_only
        self.levy_alpha = cfg.levy_alpha
        self.exploit_step = cfg.exploit_step
        self.clump_count = cfg.clump_count
        self.clump_radius = cfg.clump_radius
        self.clump_beta = cfg.clump_beta
        self.gen = torch.Generator(device=self.device) if self.device.type in ("cpu", "cuda") else None
        self.reset(cfg.seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None and self.gen is not None:
            self.gen.manual_seed(seed)
        elif seed is not None:
            torch.manual_seed(seed)
        self.pos = self._rand(self.B, self.N, 2) * self.size
        # per-agent LTM thresholds (heterogeneity: const / gaussian(theta,std) / uniform[low,high])
        if self.theta_dist == "gaussian" and self.theta_std > 0:
            self.theta_vec = (self.theta + self.theta_std * self._randn(self.B, self.N)).clamp(0.01, 0.999)
        elif self.theta_dist == "uniform":
            lo = self.theta - self.theta_std if self.theta_low is None else self.theta_low
            hi = self.theta + self.theta_std if self.theta_high is None else self.theta_high
            self.theta_vec = (lo + (hi - lo) * self._rand(self.B, self.N)).clamp(0.01, 0.999)
        else:
            self.theta_vec = torch.full((self.B, self.N), float(self.theta),
                                        device=self.device, dtype=self.dtype)
        self._n_clumps = self.clump_count if self.clump_count is not None else 40
        if self.clump_count is not None or self.clump_beta is not None:
            self.clump_centers = self._rand(self.B, self._n_clumps, 2) * self.size
        if self.clump_beta is not None:
            # Pareto clump weights -> scale-free clump SIZES with exponent ~ clump_beta; persistent per replica
            b = max(self.clump_beta - 1.0, 0.15)
            w = self._rand(self.B, self._n_clumps).clamp(min=1e-6).pow(-1.0 / b)
            w = w / w.sum(dim=1, keepdim=True)
            self._clump_idx = (torch.multinomial(w, self.K, replacement=True, generator=self.gen)
                               if self.gen is not None else torch.multinomial(w, self.K, replacement=True))
        self.sources = self._place_sources()
        self.source_cap = torch.full((self.B, self.K), self.capacity,
                                     device=self.device, dtype=self.dtype)
        self._prev_active = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        self.beacon_pos = self.pos.clone()          # last position each agent broadcast
        self._beacon_age = torch.zeros(self.B, self.N, device=self.device, dtype=self.dtype)
        self.t = 0

    def _rand(self, *shape):
        if self.gen is not None:
            return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)
        return torch.rand(*shape, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        if self.gen is not None:
            return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)
        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    def _min_image(self, diff):
        return diff - self.size * torch.round(diff / self.size)

    def _place_sources(self):
        """(B,K,2) source positions: scale-free clumps (clump_beta), uniform clumps (clump_count), or uniform."""
        if self.clump_beta is not None:
            ctr = torch.gather(self.clump_centers, 1, self._clump_idx.unsqueeze(-1).expand(-1, -1, 2))
            return torch.remainder(ctr + self._randn(self.B, self.K, 2) * self.clump_radius, self.size)
        if self.clump_count is None:
            return self._rand(self.B, self.K, 2) * self.size
        if self.gen is not None:
            idx = torch.randint(0, self.clump_count, (self.B, self.K), generator=self.gen, device=self.device)
        else:
            idx = torch.randint(0, self.clump_count, (self.B, self.K), device=self.device)
        ctr = torch.gather(self.clump_centers, 1, idx.unsqueeze(-1).expand(-1, -1, 2))  # (B,K,2)
        return torch.remainder(ctr + self._randn(self.B, self.K, 2) * self.clump_radius, self.size)

    def _levy_length(self):
        """Heavy-tailed step LENGTH ~ |alpha-stable| (Mantegna), (B,N). Direction is applied
        separately so the SAME length law is used for random (exploring) and cascade-directed
        (recruited) motion -- recruitment then changes only the heading, never the step size, so it
        cannot 'win' an efficiency comparison by making agents move less."""
        a = max(0.3, min(1.99, float(self.levy_alpha)))
        num = (math.gamma(1 + a) * math.sin(math.pi * a / 2)) ** (1.0 / a)
        den = (math.gamma((1 + a) / 2) * a * 2 ** ((a - 1) / 2)) ** (1.0 / a)
        sigma = num / den
        u = self._randn(self.B, self.N) * sigma
        w = self._randn(self.B, self.N).abs()
        raw = (u / w.clamp(min=1e-12).pow(1.0 / a)).abs()
        return raw * (self.step_scale / 10.0)

    def _rand_dir(self):
        """Uniform-angle unit vectors, (B,N,2). Isotropic -- unlike the old per-component alpha-stable
        draw, which biased long jumps along the coordinate axes."""
        ang = self._rand(self.B, self.N) * (2.0 * math.pi)
        return torch.stack([ang.cos(), ang.sin()], dim=-1)

    def _pair_dist(self, A, Bp):
        diff = self._min_image(A.unsqueeze(2) - Bp.unsqueeze(1))
        return diff.norm(dim=-1)

    @torch.no_grad()
    def step(self):
        B, N = self.B, self.N
        # --- decide who re-beacons this tick, then build the graph from BEACONED positions ---
        # The graph is only as fresh as the last broadcast. Stale positions corrupt the DEGREE k, and
        # because the rule is a/k >= theta that shifts the effective threshold -- missing neighbours
        # mean smaller k, hence EASIER activation. Staleness therefore biases toward over-recruitment,
        # the failure mode blocking exists to prevent. Watch active_frac when tuning these.
        self._beacon_age = self._beacon_age + 1.0
        if self.beacon_move_frac > 0.0:                      # event-driven: moved far enough?
            moved = self._min_image(self.pos - self.beacon_pos).norm(dim=-1)
            send = moved >= self.beacon_move_frac * self.r
        elif self.beacon_interval > 1:                       # fixed interval
            send = self._beacon_age >= float(self.beacon_interval)
        else:
            send = torch.ones(B, N, dtype=torch.bool, device=self.device)
        self.beacon_pos = torch.where(send.unsqueeze(-1), self.pos, self.beacon_pos)
        self._beacon_age = torch.where(send, torch.zeros_like(self._beacon_age), self._beacon_age)
        self._n_beacon = send.sum(dim=1).to(self.dtype)      # beacons ACTUALLY emitted this tick

        # --- interaction graph (RGG) ---
        rr = self._pair_dist(self.beacon_pos, self.beacon_pos)
        within = rr <= self.r
        rp = self._rand(B, N, N)
        rs = torch.triu(rp, diagonal=1); rs = rs + rs.transpose(1, 2)
        eye = torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        adj = within & (rs < self.p) & (~eye)
        degree = adj.sum(dim=2).to(self.dtype)
        inv_deg = torch.where(degree > 0, 1.0 / degree.clamp(min=1.0), torch.zeros_like(degree))

        # --- seeds: agents within S of a NON-depleted source ---
        d_src = self._pair_dist(self.pos, self.sources)          # (B,N,K)
        on_src = d_src <= self.S                                 # (B,N,K)
        live = self.source_cap > 0.0                             # (B,K)
        seed = (on_src & live.unsqueeze(1)).any(dim=2)           # (B,N)

        # --- LTM cascade to fixed point + branching instrumentation ---
        active = seed.clone()
        parents = torch.zeros(B, N, N, dtype=self.dtype, device=self.device)
        branch_num = torch.zeros(B, device=self.device, dtype=self.dtype)
        branch_den = torch.zeros(B, device=self.device, dtype=self.dtype)
        # OPTIONAL, behaviour-neutral: record per-round activation masks (generation sizes) so an
        # external harness can measure the per-cluster branching ratio. Off unless _record_generations.
        _rec = getattr(self, "_record_generations", False)
        if _rec:
            self._generations = [seed.clone()]                  # generation 0 = seeds
        for _ in range(N):
            a = (adj & active.unsqueeze(1)).sum(dim=2).to(self.dtype)
            newly = (a * inv_deg >= self.theta_vec) & (~active)
            if not bool(newly.any()):
                break
            branch_num = branch_num + newly.sum(dim=1).to(self.dtype)
            branch_den = branch_den + active.sum(dim=1).to(self.dtype)
            recruiters = adj & active.unsqueeze(1)
            parents = parents + (recruiters & newly.unsqueeze(-1)).to(self.dtype)
            active = active | newly
            if _rec:
                self._generations.append(newly.clone())         # generation r = recruited at round r

        recruited = active & (~seed)
        # --- BLOCKING observables: separate true Watts blocking from "never encountered" ---
        # after the fixed point, any INACTIVE agent with >=1 active neighbour necessarily has
        # a/k < theta_i -> it is suppressed by the fraction rule (blocked). Inactive with zero
        # active neighbours = genuinely exploring (no recruitment signal available).
        a_final = (adj & active.unsqueeze(1)).sum(dim=2).to(self.dtype)
        inactive = ~active
        self._blocked = (inactive & (a_final > 0)).sum(dim=1).to(self.dtype)
        self._isolated = (inactive & (a_final == 0)).sum(dim=1).to(self.dtype)
        _rev = self._prev_active & inactive                                        # released this tick
        self._reverted = _rev.sum(dim=1).to(self.dtype)
        # released-by-BLOCKING (still has active neighbours, a/k<theta) vs released-by-ISOLATION
        # (no active neighbours left -> its local patch/cluster dissolved). Distinguishes "squeezed
        # out by crowding" from "patch worked out and everyone left".
        self._reverted_blocked = (_rev & (a_final > 0)).sum(dim=1).to(self.dtype)
        self._reverted_isolated = (_rev & (a_final == 0)).sum(dim=1).to(self.dtype)
        # capture the inactive->active TRANSITIONS before _prev_active is overwritten -- the
        # event-driven activation ledger below needs the previous tick's state, not this one's
        self._newly_active = (active & ~self._prev_active).sum(dim=1).to(self.dtype)
        self._prev_active = active.clone()
        # --- observables ---
        n_active = active.sum(dim=1).to(self.dtype)
        self._sigma = torch.where(branch_den > 0, branch_num / branch_den.clamp(min=1.0),
                                  torch.zeros_like(branch_den))
        self._z_local = torch.where(n_active > 0, (degree * active).sum(dim=1) / n_active.clamp(min=1.0),
                                    torch.zeros_like(n_active))
        self._z_global = degree.mean(dim=1)
        self._active_count = n_active
        self._cascade_size = recruited.sum(dim=1).to(self.dtype)
        self._msgs_edge = parents.sum(dim=(1, 2))               # recruiting signals (per-edge; legacy)
        self._msgs_node = n_active                              # agents broadcasting active status (legacy alias)
        self._msgs_beacon = getattr(self, "_n_beacon", torch.full_like(n_active, float(N)))
        # "I found a target" is naturally an EVENT. Re-announcing it every tick while still active is
        # redundant, so event mode charges only the inactive->active transitions.
        self._msgs_activation = (self._newly_active if self.activation_event_only else n_active)
        self._adj = adj                                         # for local connected-component analysis
        self._active_state = active

        # --- consume sources; respawn depleted ones (this drives the SOC cycle) ---
        n_on = on_src.sum(dim=1).to(self.dtype)                  # (B,K) agents on each source
        before = self.source_cap
        after = (before - self.consumption * n_on).clamp(min=0.0)
        self._consumed = (before - after).sum(dim=1)            # (B,) resource removed = throughput
        self.source_cap = after
        depleted = self.source_cap <= 0.0
        if bool(depleted.any()):
            new_src = self._place_sources()                      # clumped (patchy) or uniform respawn
            self.sources = torch.where(depleted.unsqueeze(-1), new_src, self.sources)
            self.source_cap = torch.where(depleted,
                                          torch.full_like(self.source_cap, self.capacity),
                                          self.source_cap)

        # --- motion: three mutually exclusive modes (Galen's decided semantics 2026-07-18) ---
        #   ON FOOD (seed)          -> area-restricted search: small isotropic Brownian step
        #   RECRUITED, off food     -> Levy flight with heading SET BY THE CASCADE (toward parents,
        #                              which lie transitively toward the seed) -- same length law as
        #                              exploration, only the direction is informed
        #   EXPLORING (inactive)    -> Levy flight in a RANDOM direction
        # Recruitment thus changes the direction of search, never its step size, so the movement-cost
        # comparison against a non-recruiting swarm is honest (the old code drifted recruited agents a
        # tiny 'pull'-fraction toward one-hop recruiters, which read as a near-freeze).
        if self.levy_alpha is not None:
            length = self._levy_length()                         # (B,N) shared Levy length
            rdir = self._rand_dir()                              # (B,N,2) random headings
            # cascade heading: unit vector toward the parent centroid (min-image on the torus)
            disp = -self._min_image(self.pos.unsqueeze(2) - self.pos.unsqueeze(1))   # i->j (B,N,N,2)
            pmask = parents.unsqueeze(-1)
            count = parents.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
            mean_disp = (disp * pmask).sum(dim=2) / count         # (B,N,2) toward recruiters
            mnorm = mean_disp.norm(dim=-1, keepdim=True)
            cascade_dir = torch.where(mnorm > 1e-9, mean_disp / mnorm.clamp(min=1e-9), rdir)
            has_head = parents.sum(dim=2) > 0
            # coupling ablation: converge=False -> recruited agents get NO cascade heading (move
            # randomly like explorers), isolating the recruitment->motion coupling
            directed = (recruited & has_head) if self.converge else torch.zeros_like(recruited)
            heading = torch.where(directed.unsqueeze(-1), cascade_dir, rdir)
            levy_vec = heading * length.unsqueeze(-1)
            brownian = self._randn(B, N, 2) * self.exploit_step  # isotropic ARS step on food
            step_vec = torch.where(seed.unsqueeze(-1), brownian, levy_vec)
        else:
            step_vec = self._randn(B, N, 2) * self.step_scale    # backward-compatible random walk
            directed = torch.zeros_like(recruited)
        # --- movement ledgers, decomposed by motion PHASE (partition all N; sum to total) ---
        #   explore  = inactive (random Levy) ; converge = recruited & directed (cascade Levy) ;
        #   exploit  = the rest of the active set (seeds doing ARS, + recruited without a heading)
        norms = step_vec.norm(dim=-1)                            # (B,N) per-agent step length
        self._dist = norms.sum(dim=1)                            # (B,) total distance this tick
        self._dist_converge = (norms * directed).sum(dim=1)
        self._dist_exploit = (norms * (active & ~directed)).sum(dim=1)
        self._dist_explore = (norms * (~active)).sum(dim=1)
        self.pos = torch.remainder(self.pos + step_vec, self.size)
        self.t += 1

    def _snapshot(self) -> Dict[str, Tensor]:
        z = torch.zeros(self.B, device=self.device, dtype=self.dtype)
        return {
            "sigma": getattr(self, "_sigma", z),
            "z_local": getattr(self, "_z_local", z),
            "z_global": getattr(self, "_z_global", z),
            "active_count": getattr(self, "_active_count", z),
            "cascade_size": getattr(self, "_cascade_size", z),
            "exploring_fraction": (self.N - getattr(self, "_active_count", z)) / self.N,
            "resource": self.source_cap.sum(dim=1),
            # --- raw cost ledgers (Q4): keep separate, derive efficiencies downstream ---
            "distance": getattr(self, "_dist", z),                 # (1) movement: total step-length
            # movement split by phase (sums to "distance"); enables exploration/recruitment-travel
            # efficiencies later. Aggregate to the total unless a phase-specific question is asked.
            "dist_explore": getattr(self, "_dist_explore", z),     # Levy search (inactive)
            "dist_exploit": getattr(self, "_dist_exploit", z),     # local Brownian (active)
            "dist_converge": getattr(self, "_dist_converge", z),   # recruitment travel to parents
            "msgs_beacon": getattr(self, "_msgs_beacon", z),       # (2) presence pings (network upkeep, ~N/tick)
            "msgs_activation": getattr(self, "_msgs_activation", z),# (3) cascade pings (recruitment, n_active/tick)
            "consumed": getattr(self, "_consumed", z),             # (4) resource harvested
            # --- blocking decomposition (is self-limitation from the fraction rule or depletion?) ---
            "blocked": getattr(self, "_blocked", z),                # inactive WITH active neighbours
            "isolated": getattr(self, "_isolated", z),              # inactive with NO active neighbours
            "reverted": getattr(self, "_reverted", z),              # active last tick, inactive now
            "msgs_node": getattr(self, "_msgs_node", z),           # legacy alias of activation
            "msgs_edge": getattr(self, "_msgs_edge", z),           # legacy per-edge count
        }

    @torch.no_grad()
    def run(self, T: int) -> Dict[str, Tensor]:
        keys = list(self._snapshot().keys())
        buf = {k: torch.empty(self.B, T, device=self.device, dtype=self.dtype) for k in keys}
        for t in range(T):
            self.step()
            snap = self._snapshot()
            for k in keys:
                buf[k][:, t] = snap[k]
        return buf


if __name__ == "__main__":
    cfg = MobileCascadeConfig(B=128, N=100, K=30, rho_r=2.0, phi=0.05, theta=0.1, seed=0)
    sim = MobileCascade(cfg)
    print("device:", sim.device, " r=%.1f S=%.1f" % (sim.r, sim.S))
    out = sim.run(300)
    t0 = 120
    for k, v in out.items():
        print(f"{k:18s} steady-state mean {v[:, t0:].mean().item():.4f}")
