# Self-organizing reinforcement learning: a swarm approaches the explore–exploit optimum

Code and manuscript for the paper *"Self-organizing reinforcement learning: a swarm approaches the
explore–exploit optimum."*

A minimal foraging swarm — Lévy exploration + a threshold recruitment cascade on the proximity graph +
density-dependent dispersal — self-organizes to the ideal-free / Nash-equilibrium allocation of a multi-agent
habitat-selection (congestion) explore/exploit problem, with **no learner, no reward signal, and no training**.
The swarm reaches the *shape* of the optimum (not the exact equilibrium), does so by reaching rather than
learning, and — measured against a validated oracle ceiling in a shared embodied environment — is competitive
with an online decentralized RL learner at none of its overhead.

## The idea, in one picture

![The physical model: Lévy exploration, a recruitment cascade, and blocking](figures/fig_mechanism_tikz.png)

The swarm is solving a **multi-armed bandit** — a spatial, embodied one. Each resource patch is an *arm* whose
**state must be discovered**: unlike a textbook bandit with fixed, labelled arms, here an arm's existence, its
**location, and its size (richness) are unknown** and have to be found by searching, and its payoff *per
forager* **drops as more foragers crowd onto it** (a congestion bandit). So the swarm must both *explore* —
Lévy flights, to find the patches — and *exploit* — recruit neighbours to a discovery, then disperse as it
crowds.

The surprise: it reaches this bandit's optimum with **none of reinforcement learning's machinery** — no
learner, no value function, no training, no reward signal in the loop. The self-organizing
explore → recruit → disperse cycle settles at the **ideal-free / Nash allocation** (foragers in proportion to
patch richness), which *is* the bandit's optimal allocation — reached *approximately*, and by *reaching* rather
than *learning* (see the paper for the precise, calibrated claims).

## Interactive demo

A self-contained, browser-based demo — this paper's reward-free / learner-free version — is in
[`demos/collective_foraging_demo.html`](demos/collective_foraging_demo.html). No install: just open the file
in a browser. It runs the whole story live — Lévy exploration, recruitment cascades, blocking, and the swarm
settling at the ideal-free allocation. (When this repository is made public it can be served directly via
GitHub Pages.)

## Contents

- `main.tex`, `refs.bib` — manuscript source (TMLR format; anonymized for double-blind review).
- `figures/` — the figures used in the paper (`*.pdf`/`*.png`) and their **self-contained generators**
  (`gen_*.py`):
  - `gen_ifd_fig_b64.py`, `gen_convergence_fig.py` — ideal-free matching + dynamic convergence.
  - `gen_bridge_figs.py` — recruitment-vs-payoff (reward-free illustration).
  - `gen_b1_selforg.py` — self-organizing operating regime (cascade window / σ-invariance sweep).
  - `gen_discrim_exp1.py`, `gen_discrim_exp2.py`, `gen_discrim_exp3.py` — dynamics model competition,
    blocking ablation, information plateau.
  - `gen_marl_figs.py` — abstract congestion-game trained-RL comparison.
  - `gen_b2_embodied.py` — embodied learner comparison (shared spatial environment).
  - `gen_crossover.py` — oracle-normalized horizon × nonstationarity sweep.
- `main.pdf` — compiled manuscript.

## Reproducing

```
pip install -r requirements.txt
cd figures
python3 gen_<name>.py        # regenerates the corresponding figure
```

Simulations run on CPU. Large intermediate data (sweep caches) are written to a local scratch path and are
**not** committed; each generator recomputes or reloads its own cache. See each script's header for details.

## Companion paper

The underlying self-organized-critical mechanism (the cascade window; movement-neutral recruitment) is
developed in a companion paper (Part I).

## License

MIT — see `LICENSE`.
