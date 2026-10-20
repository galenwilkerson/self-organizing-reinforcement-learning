# arXiv submission manifest — de-anonymized version

This folder is now **arXiv-only**: `main.tex` is the de-anonymized build (forces `\anonfalse`) and
`refs.bib` is the de-anonymized bibliography. The anon/TMLR double-blind twin was removed 2026-10-20.

Upload the bundle `arxiv_submission.tar.gz` (preserves the `figures/` subdir). Or upload the files below
individually, keeping the `figures/` subdirectory.

**This goes up as an arXiv v2 REPLACEMENT of arXiv:2610.25113** (keeps the ID). Reason for v2: new title
("... a learner-free swarm approaches the explore--exploit optimum").

## Files to upload (in the tarball)
- **Main TeX:** `main.tex`  — standalone, de-anonymized; arXiv compiles this directly.
- **Bibliography:** `main.bbl`  — include so arXiv need not rerun BibTeX.
- **Style files:** `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`
- **Figures (10, in `figures/`):**
  `fig_mechanism_tikz.pdf`, `fig_cycle_schematic_tikz.pdf`, `fig_ifd.pdf`, `fig_convergence.pdf`,
  `fig_b1_selforg.pdf`, `fig_recruit_vs_payoff.pdf`, `fig_discrim_plateau.pdf`, `fig_marl_compare.pdf`,
  `fig_marl_regimes.pdf`, `fig_stronger.pdf`

## Do NOT upload
- `refs.bib` (not needed — the `.bbl` is bundled), this file, `README.md`, `CITATION.cff`, `LICENSE`
- `_working_notes/`, `figures/_unused/`, all `gen_*.py`, all `.png`, `demos/`

## Metadata to set on the v2 replacement
- **Title:** Self-organizing reinforcement learning: a learner-free swarm approaches the explore--exploit optimum
- Author/affiliation/email and the real competing-interests + companion reference appear (de-anonymized).
- Companion Part I is cited as "Companion paper (Part I), under review" (no arXiv ID in the entry yet; Part I
  is arXiv:2610.24865 — add it to `refs.bib` if you want the ID to appear, then rebuild the `.bbl` + tarball).

## Build notes
- Verified: standalone compile with the bundled `.bbl` (no bibtex) = 16 pages, 0 undefined. Matches arXiv AutoTeX.
- `arxiv_preview.pdf` is the compiled output of this exact bundle.
