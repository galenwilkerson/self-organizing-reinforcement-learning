# Self-organizing reinforcement learning: a swarm approaches the explore–exploit optimum

Code and manuscript for the paper *"Self-organizing reinforcement learning: a swarm approaches the
explore–exploit optimum."*

A minimal foraging swarm — Lévy exploration + a threshold recruitment cascade on the proximity graph +
density-dependent dispersal — self-organizes to the ideal-free / Nash-equilibrium allocation of a multi-agent
habitat-selection (congestion) explore/exploit problem, with **no learner, no reward signal, and no training**.
The swarm reaches the *shape* of the optimum (not the exact equilibrium), does so by reaching rather than
learning, and — measured against a validated oracle ceiling in a shared embodied environment — is competitive
with an online decentralized RL learner at none of its overhead.

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
