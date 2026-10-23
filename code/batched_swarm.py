"""
Batched toroidal Lévy-flight foraging swarm with linear-threshold cascade.

A PyTorch port of the D3 "robots and oil blobs" simulation, designed for
Monte Carlo experiments. The key idea is that the batch dimension B *is* the
parallelism: B independent simulations advance in lockstep, every operation
vectorized over B, with no Python loops over robots or blobs. One sim is far
too small to use a GPU; ten thousand at once is exactly the dense-tensor
workload a GPU is built for.

State tensors all carry a leading batch axis B:
    robot_pos       (B, N, 2)   float
    robot_state     (B, N)      bool   (False=inactive, True=active)
    robot_thresh    (B, N)      float
    blob_pos        (B, M, 2)   float
    blob_radius     (B, M)      float  (clamped >= 0)

N and M are fixed within a batch (rectangular tensors). To sweep N or M, run
separate batches.

Per-tick pipeline preserves the original JS ordering:
    1. move      Lévy flight if inactive, Brownian if active; wrap on torus
    2. consume   reduce every blob a robot touches; robot active if touching any
    3. cascade   one synchronous linear-threshold round on the undirected graph
    4. attract   cascade-only activations drift toward active-neighbour centroid
    5. commit    write state, record metrics

Geometry is toroidal throughout via the minimum-image convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union, Dict

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# Device selection: CUDA (e.g. RTX 3090) -> Apple Metal (MPS) -> CPU
# --------------------------------------------------------------------------
def select_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available device.

    Order: explicit `prefer` if given and available, else CUDA, else MPS
    (Apple Metal), else CPU. This lets the model be developed on an M-series
    Mac and run for real on a Linux/CUDA box with no code change.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# A parameter may be a single scalar (shared across the whole batch) or a
# per-run tensor/sequence of shape (B,). Stored as given; broadcast at use.
ScalarOrBatch = Union[float, int, Tensor]


@dataclass
class SwarmConfig:
    """All simulation parameters. Each numeric field may be a scalar (shared
    across the batch) or a (B,) tensor (one value per run). N, M, B, and the
    world size are structural and must be plain ints/floats.
    """

    # ---- structural (fixed within a batch) ----
    B: int = 1024            # number of independent simulations
    N: int = 50              # robots per simulation
    M: int = 20              # oil blobs per simulation
    width: float = 800.0
    height: float = 800.0

    # ---- foraging / motion ----
    levy_alpha: ScalarOrBatch = 1.0      # Lévy stability exponent, (0,2)
    step_scale: ScalarOrBatch = 20.0     # "v": Lévy scale (matches JS v)
    exploit_step: ScalarOrBatch = 3.0    # Brownian step sigma when active

    # ---- blobs ----
    blob_size: ScalarOrBatch = 15.0      # mean blob radius S
    blob_cv: ScalarOrBatch = 0.5         # coeff. of variation of blob radius
    consumption_rate: ScalarOrBatch = 1.0  # per robot, per touched blob, per tick
    replenish: bool = False              # respawn depleted blobs -> steady-state task (for stationary SOC tests)

    # ---- interaction graph / cascade ----
    radius: ScalarOrBatch = 100.0        # neighbour / connection radius
    p: ScalarOrBatch = 1.0               # edge probability given within radius
    min_threshold: ScalarOrBatch = 0.18  # LTM threshold lower bound
    max_threshold: ScalarOrBatch = 0.18  # LTM threshold upper bound
    centroid_pull: ScalarOrBatch = 0.1   # fraction of displacement per tick

    # ---- runtime ----
    device: Optional[str] = None         # None -> auto (cuda/mps/cpu)
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None


def config_from_dimensionless(
    rho_r: ScalarOrBatch,
    phi: ScalarOrBatch,
    alpha: ScalarOrBatch,
    theta: ScalarOrBatch,
    *,
    N: int = 50,
    M: int = 20,
    B: int = 1024,
    L: float = 800.0,
    **kwargs,
) -> SwarmConfig:
    """Build a SwarmConfig from the four dimensionless control parameters.

    The model's genuine degrees of freedom (see writeup) are:
        rho_r = N (r/L)^2   robot connectivity density   -> sets radius r
        phi   = M (S/L)^2   oil coverage fraction         -> sets blob_size S
        alpha               Levy exponent                 -> levy_alpha
        theta               cascade threshold (single)    -> min=max=theta

    Inverting the first two:
        r = L * sqrt(rho_r / N)
        S = L * sqrt(phi   / M)

    rho_r, phi, alpha, theta may each be a scalar (shared across the batch) or
    a length-B tensor (one value per run), so a single batch can sweep any of
    them. N, M, B, L are structural and fixed within a batch. Extra kwargs
    (consumption_rate, exploit_step, blob_cv, p, centroid_pull, device, seed,
    ...) pass straight through to SwarmConfig as fixed constants.
    """
    def _sqrt(x):
        return x.sqrt() if torch.is_tensor(x) else math.sqrt(x)

    # invert rho_r = N(r/L)^2 and phi = M(S/L)^2; works for scalar or (B,) tensor
    radius = L * _sqrt(rho_r / N)
    blob_size = L * _sqrt(phi / M)

    return SwarmConfig(
        B=B, N=N, M=M, width=L, height=L,
        radius=radius,
        blob_size=blob_size,
        levy_alpha=alpha,
        min_threshold=theta,
        max_threshold=theta,
        **kwargs,
    )


def _as_b1(x: ScalarOrBatch, B: int, device, dtype) -> Tensor:
    """Coerce a scalar-or-(B,) parameter to a (B, 1) tensor for broadcasting
    against (B, N) or (B, M) state. A scalar expands to all runs; a (B,)
    tensor is reshaped; a (B,1) passes through.
    """
    if not torch.is_tensor(x):
        return torch.full((B, 1), float(x), device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    if x.ndim == 0:
        return x.expand(B).reshape(B, 1).clone()
    if x.ndim == 1:
        if x.shape[0] != B:
            raise ValueError(f"per-run parameter has length {x.shape[0]}, expected B={B}")
        return x.reshape(B, 1)
    if x.ndim == 2 and x.shape == (B, 1):
        return x
    raise ValueError(f"parameter must be scalar or shape (B,), got {tuple(x.shape)}")


def _levy_sigma(alpha: Tensor) -> Tensor:
    """Mantegna (1994) scale for the numerator normal in a symmetric stable
    draw. Depends only on alpha, so it is computed once per step, not per
    robot. alpha is a (B,1) tensor.
    """
    a = alpha
    # gammas via lgamma for stability; sin term uses the torch tensor directly
    num = (torch.lgamma(1 + a).exp() * torch.sin(math.pi * a / 2)).pow(1.0 / a)
    den = (torch.lgamma((1 + a) / 2).exp() * a * torch.pow(torch.as_tensor(2.0, device=a.device, dtype=a.dtype), (a - 1) / 2)).pow(1.0 / a)
    return num / den


class BatchedSwarm:
    """Holds the state of B simulations and advances them in lockstep.

    Typical use:
        cfg = SwarmConfig(B=4096, N=50, M=20, levy_alpha=1.0)
        sim = BatchedSwarm(cfg)
        metrics = sim.run(T=2000)   # dict of (B, T) tensors
    """

    def __init__(self, cfg: SwarmConfig):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.dtype = cfg.dtype
        self.B, self.N, self.M = cfg.B, cfg.N, cfg.M
        self.replenish = bool(cfg.replenish)
        self.size = torch.tensor([cfg.width, cfg.height], device=self.device, dtype=self.dtype)

        # one generator so reset(seed) is reproducible across devices that
        # support it (CPU/CUDA). MPS falls back to the default RNG stream.
        self.gen = torch.Generator(device=self.device) if self.device.type in ("cpu", "cuda") else None

        # pre-coerce broadcastable parameters to (B,1) tensors
        B = self.B
        self.alpha = _as_b1(cfg.levy_alpha, B, self.device, self.dtype)
        self.step_scale = _as_b1(cfg.step_scale, B, self.device, self.dtype)
        self.exploit_step = _as_b1(cfg.exploit_step, B, self.device, self.dtype)
        self.blob_size = _as_b1(cfg.blob_size, B, self.device, self.dtype)
        self.blob_cv = _as_b1(cfg.blob_cv, B, self.device, self.dtype)
        self.consumption_rate = _as_b1(cfg.consumption_rate, B, self.device, self.dtype)
        self.radius = _as_b1(cfg.radius, B, self.device, self.dtype)
        self.p = _as_b1(cfg.p, B, self.device, self.dtype)
        self.min_threshold = _as_b1(cfg.min_threshold, B, self.device, self.dtype)
        self.max_threshold = _as_b1(cfg.max_threshold, B, self.device, self.dtype)
        self.centroid_pull = _as_b1(cfg.centroid_pull, B, self.device, self.dtype)

        self.reset(cfg.seed)

    # ------------------------------------------------------------------
    # RNG helpers (route through self.gen when available)
    # ------------------------------------------------------------------
    def _rand(self, *shape) -> Tensor:
        if self.gen is not None:
            return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)
        return torch.rand(*shape, device=self.device, dtype=self.dtype)

    def _randn(self, *shape) -> Tensor:
        if self.gen is not None:
            return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)
        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Randomise initial positions, blob sizes, and thresholds."""
        if seed is not None and self.gen is not None:
            self.gen.manual_seed(seed)
        elif seed is not None:
            torch.manual_seed(seed)

        B, N, M = self.B, self.N, self.M

        self.robot_pos = self._rand(B, N, 2) * self.size            # (B,N,2)
        self.robot_state = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        # per-robot threshold uniform in [min, max]
        u = self._rand(B, N)
        self.robot_thresh = self.min_threshold + u * (self.max_threshold - self.min_threshold)

        self.blob_pos = self._rand(B, M, 2) * self.size             # (B,M,2)
        # blob radius ~ Normal(S, cv*S), floored at 1 (matches JS)
        blob_r = self.blob_size + self._randn(B, M) * (self.blob_cv * self.blob_size)
        self.blob_radius = blob_r.clamp(min=1.0)                    # (B,M)

        self.t = 0

    # ------------------------------------------------------------------
    # geometry: minimum-image (toroidal) displacement
    # ------------------------------------------------------------------
    def _min_image(self, diff: Tensor) -> Tensor:
        """Wrap a coordinate difference into [-size/2, size/2] per axis.
        `diff` has trailing dim 2; self.size broadcasts over it.
        """
        return diff - self.size * torch.round(diff / self.size)

    def _pair_dist(self, A: Tensor, Bp: Tensor) -> Tensor:
        """Toroidal distances between point sets A (B,n,2) and Bp (B,m,2).
        Returns (B,n,m).
        """
        diff = A.unsqueeze(2) - Bp.unsqueeze(1)          # (B,n,m,2)
        diff = self._min_image(diff)
        return diff.norm(dim=-1)                          # (B,n,m)

    # ------------------------------------------------------------------
    # Lévy step: symmetric stable via Mantegna, drawn per (B,N,2)
    # ------------------------------------------------------------------
    def _levy_step(self) -> Tensor:
        a = self.alpha.clamp(0.3, 1.99)                  # (B,1) domain guard
        sigma = _levy_sigma(a)                           # (B,1)
        # x and y are independent draws -> shape (B,N,2)
        u = sigma.unsqueeze(-1) * self._randn(self.B, self.N, 2)
        w = self._randn(self.B, self.N, 2).abs()
        inv_a = (1.0 / a).unsqueeze(-1)                  # (B,1,1)
        raw = u / w.pow(inv_a)
        return raw * (self.step_scale / 10.0).unsqueeze(-1)   # scale by v/10

    # ------------------------------------------------------------------
    # one tick
    # ------------------------------------------------------------------
    def step(self) -> None:
        B, N, M = self.B, self.N, self.M

        # --- 1. move: Lévy if inactive, Brownian if active ---
        levy = self._levy_step()                                   # (B,N,2)
        brownian = self._randn(B, N, 2) * self.exploit_step.unsqueeze(-1)
        active = self.robot_state.unsqueeze(-1)                    # (B,N,1) bool
        step = torch.where(active, brownian, levy)
        self.robot_pos = torch.remainder(self.robot_pos + step, self.size)
        # per-tick locomotion distance (excludes the cascade centroid drift below)
        self._dist_step = step.norm(dim=-1).sum(dim=1)             # (B,)

        # --- 2. consume from every touched blob ---
        rb = self._pair_dist(self.robot_pos, self.blob_pos)        # (B,N,M)
        overlap = rb <= self.blob_radius.unsqueeze(1)              # (B,N,M) bool
        touching_any = overlap.any(dim=2)                          # (B,N)
        # per-blob consumption = rate * (#robots touching it)
        n_touch = overlap.sum(dim=1).to(self.dtype)                # (B,M)
        consumed = self.consumption_rate * n_touch                 # (B,M)
        self.blob_radius = (self.blob_radius - consumed).clamp(min=0.0)

        # optional replenishment: respawn fully-depleted blobs at fresh random
        # positions -> a steady-state task, giving a stationary setting for the
        # SOC self-tuning test (Q2). Off by default.
        if self.replenish:
            depleted = self.blob_radius <= 0.0                     # (B,M)
            if bool(depleted.any()):
                new_r = (self.blob_size + self._randn(B, M) * (self.blob_cv * self.blob_size)).clamp(min=1.0)
                new_pos = self._rand(B, M, 2) * self.size
                self.blob_radius = torch.where(depleted, new_r, self.blob_radius)
                self.blob_pos = torch.where(depleted.unsqueeze(-1), new_pos, self.blob_pos)

        # robots touching a blob are seed-active this tick
        collision_active = touching_any                            # (B,N)

        # --- 3. cascade: run the LTM to its fixed point within this tick ---
        # Information propagates across the network much faster than robots
        # move, so the cascade fully resolves each tick. Activation is monotone
        # (robots only ever turn on during the cascade), so the iteration is
        # guaranteed to converge in <= N rounds.
        rr = self._pair_dist(self.robot_pos, self.robot_pos)       # (B,N,N)
        within = rr <= self.radius.unsqueeze(-1)                   # (B,N,N) bool

        # symmetric edge draw: upper triangle mirrored, compare < p
        rand_pair = self._rand(B, N, N)
        rand_sym = torch.triu(rand_pair, diagonal=1)
        rand_sym = rand_sym + rand_sym.transpose(1, 2)            # symmetric, 0 on diag
        edge_keep = rand_sym < self.p.unsqueeze(-1)               # (B,N,N)
        eye = torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        adj = within & edge_keep & (~eye)                         # undirected, no self
        degree = adj.sum(dim=2).to(self.dtype)                    # (B,N)
        inv_deg = torch.where(degree > 0, 1.0 / degree.clamp(min=1.0),
                              torch.zeros_like(degree))            # (B,N)

        # state recomputed fresh from geometry + oil-seeds; no activation memory.
        # Dense active clusters are self-sustaining under the fixed-point LTM,
        # so persistence emerges from geometry rather than being stored.
        active = collision_active.clone()                         # (B,N) seeds on
        # parents[b,i,j] = 1 if j was active at the round i crossed threshold
        parents = torch.zeros(B, N, N, dtype=self.dtype, device=self.device)

        # --- SOC instrumentation (additive; does NOT affect the dynamics) ---
        # branching ratio sigma = offspring per active site, accumulated per run
        # over the productive rounds of the fixed-point iteration. At the self-
        # organized (homeostatic) state sigma -> 1: cascades amplify (sigma>1)
        # until blocking chokes them (sigma<1). casc_rounds is a per-run
        # correlation-time proxy that should diverge near the critical ridge.
        casc_rounds = torch.zeros(B, device=self.device, dtype=self.dtype)   # productive rounds / run
        casc_gen1 = torch.zeros(B, device=self.device, dtype=self.dtype)     # 1st-generation recruits / run
        branch_num = torch.zeros(B, device=self.device, dtype=self.dtype)    # sum_r newly[r]
        branch_den = torch.zeros(B, device=self.device, dtype=self.dtype)    # sum_r active[r-1]  (parent generation)

        for r in range(N):                                        # <= N rounds to converge
            active_nbrs = (adj & active.unsqueeze(1)).sum(dim=2).to(self.dtype)
            ratio = active_nbrs * inv_deg                         # (B,N)
            newly = (ratio >= self.robot_thresh) & (~active)      # (B,N) just crossed
            if not bool(newly.any()):
                break
            newly_cnt = newly.sum(dim=1).to(self.dtype)           # (B,) offspring this round
            active_cnt = active.sum(dim=1).to(self.dtype)         # (B,) parent generation size
            casc_rounds = casc_rounds + (newly_cnt > 0).to(self.dtype)
            branch_num = branch_num + newly_cnt
            branch_den = branch_den + active_cnt
            if r == 0:
                casc_gen1 = newly_cnt                             # seeds' direct offspring
            # record the active neighbours that recruited each newly-active robot
            recruiters = adj & active.unsqueeze(1)                # (B,N,N) bool
            parents = parents + (recruiters & newly.unsqueeze(-1)).to(self.dtype)
            active = active | newly

        new_state = active                                        # (B,N)

        # cascade recruits = active but not oil-seeds (seeds hold the oil)
        recruited = new_state & (~collision_active)               # (B,N)

        # --- SOC / order observables (additive; recorded per tick, for Q2) ---
        # sigma -> 1 at the marginal (homeostatic) regime; z_local is the mean RGG
        # degree experienced by active agents (should self-tune to z_c if SOC holds);
        # z_global is the imposed average connectivity.
        self._cascade_size = recruited.sum(dim=1).to(self.dtype)          # (B,)
        self._sigma = torch.where(branch_den > 0, branch_num / branch_den.clamp(min=1.0),
                                  torch.zeros_like(branch_den))           # (B,) branching ratio
        active_f = new_state                                             # (B,N) bool
        n_active = active_f.sum(dim=1).to(self.dtype)                    # (B,)
        self._z_local = torch.where(n_active > 0,
                                    (degree * active_f).sum(dim=1) / n_active.clamp(min=1.0),
                                    torch.zeros_like(n_active))          # (B,) mean deg among active
        self._z_global = degree.mean(dim=1)                             # (B,) mean deg over all

        # --- communication metrics for this tick ---
        # per-node: robots that changed state (must broadcast their new status)
        self._msgs_node = (new_state != self.robot_state).sum(dim=1).to(self.dtype)  # (B,)
        # per-edge: every recruiting message received, i.e. all parent edges
        self._msgs_edge = parents.sum(dim=(1, 2))                 # (B,)

        # --- 4. recruited robots move toward their actual recruiting parents ---
        # This induced convergence raises local density, which feeds back on the
        # LTM (active *fraction*) and is the source of the explore/exploit
        # blocking. Seeds touching oil do not drift.
        has_parents = parents.sum(dim=2) > 0                      # (B,N)
        move_mask = recruited & has_parents                       # (B,N)
        if bool(move_mask.any()):
            # toroidal displacement i -> j for each neighbour j
            diff = self.robot_pos.unsqueeze(2) - self.robot_pos.unsqueeze(1)  # (B,N,N,2)
            disp = -self._min_image(diff)                        # i -> j
            pmask = parents.unsqueeze(-1)                        # (B,N,N,1)
            count = pmask.sum(dim=2).clamp(min=1.0)              # (B,N,1)
            mean_disp = (disp * pmask).sum(dim=2) / count        # (B,N,2) centroid of parents
            move = mean_disp * self.centroid_pull.unsqueeze(-1)  # (B,N,2)
            move = move * move_mask.unsqueeze(-1)                # only recruited-with-parents
            self.robot_pos = torch.remainder(self.robot_pos + move, self.size)

        # --- 5. commit ---
        self.robot_state = new_state
        self.t += 1

    # ------------------------------------------------------------------
    # metrics  (five trajectories; see writeup section "Observables")
    # ------------------------------------------------------------------
    def _metrics_snapshot(self) -> Dict[str, Tensor]:
        """Per-tick observables, one scalar per run. Returns dict of (B,)
        tensors recorded into (B, T) buffers by run().

            oil_remaining  total oil left (the cleanup spine)
            active_count   instantaneous cascade size
            distance       locomotion distance this tick (cumulate downstream)
            msgs_node      robots that changed state (per-node broadcasts)
            msgs_edge      recruiting messages received (per-edge, all parents)
        """
        z = torch.zeros(self.B, device=self.device, dtype=self.dtype)
        active_count = self.robot_state.sum(dim=1).to(self.dtype)
        return {
            "oil_remaining": self.blob_radius.sum(dim=1),
            "active_count": active_count,
            "distance": getattr(self, "_dist_step", z),
            "msgs_node": getattr(self, "_msgs_node", z),
            "msgs_edge": getattr(self, "_msgs_edge", z),
            # --- SOC / order observables (Q2) ---
            "exploring_fraction": (self.N - active_count) / self.N,   # phi(t) -> phi*?
            "cascade_size": getattr(self, "_cascade_size", z),
            "sigma": getattr(self, "_sigma", z),                      # branching ratio -> 1?
            "z_local": getattr(self, "_z_local", z),                  # local degree -> z_c?
            "z_global": getattr(self, "_z_global", z),                # imposed connectivity
        }

    # ------------------------------------------------------------------
    # run T ticks, collecting metrics into (B, T) tensors
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, T: int) -> Dict[str, Tensor]:
        keys = list(self._metrics_snapshot().keys())
        buffers = {k: torch.empty(self.B, T, device=self.device, dtype=self.dtype) for k in keys}
        for t in range(T):
            self.step()
            snap = self._metrics_snapshot()
            for k in keys:
                buffers[k][:, t] = snap[k]
        return buffers


if __name__ == "__main__":
    # build from the four dimensionless control parameters
    cfg = config_from_dimensionless(
        rho_r=4.0, phi=0.05, alpha=1.0, theta=0.18,
        N=50, M=20, B=256, seed=0,
    )
    sim = BatchedSwarm(cfg)
    print("device:", sim.device)
    print(f"derived radius={sim.radius[0,0]:.1f}  blob_size={sim.blob_size[0,0]:.1f}")
    out = sim.run(T=200)
    for k, v in out.items():
        print(f"{k:14s}: shape {tuple(v.shape)}  final/total mean {v[:, -1].mean().item():.3f}")
