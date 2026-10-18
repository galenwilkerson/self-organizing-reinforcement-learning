# arXiv submission manifest — de-anonymized version

Upload the **de-anonymized** build. Easiest: upload `arxiv_submission.tar.gz` (preserves the `figures/`
subdir). Or upload these files individually, keeping the `figures/` subdirectory:

## Files to upload
- **Main TeX:** `main_arxiv.tex`  — standalone, de-anonymized (forces `\anonfalse`); arXiv compiles this directly.
- **Bibliography:** `main_arxiv.bbl`  — include so arXiv need not rerun BibTeX. (Optionally also `refs_arxiv.bib`.)
- **Style files:** `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`
- **Figures (10, in `figures/`):**
  `fig_mechanism_tikz.pdf`, `fig_cycle_schematic_tikz.pdf`, `fig_ifd.pdf`, `fig_convergence.pdf`,
  `fig_b1_selforg.pdf`, `fig_recruit_vs_payoff.pdf`, `fig_discrim_plateau.pdf`, `fig_marl_compare.pdf`,
  `fig_marl_regimes.pdf`, `fig_stronger.pdf`
  (NOTE: `fig_b2_embodied.pdf` and `fig_crossover.pdf` were REMOVED 2026-10-18 — the weak-baseline embodied
  comparison they showed was superseded by the properly-trained PPO/MAPPO result now in `fig_stronger.pdf`.)

## Do NOT upload
- `main.tex` (the **anonymized** journal version — arXiv could compile it instead)
- `refs.bib` (anonymized bib), this file, `ARXIV_SUBMISSION.md`
- `_working_notes/`, `figures/_unused/`, all `gen_*.py`, all `.png`, `demos/`

## Notes
- Main file `main_arxiv.tex` basename → arXiv jobname `main_arxiv` → it uses `main_arxiv.bbl`. Good.
- Author/affiliation/email and the real competing-interests + companion reference appear (de-anonymized).
- Competing-interests statement is generic (no patent number) — edit `main_arxiv.tex` if you want the exact
  provisional-application number stated.
- Part I is cited as "Companion paper (Part I), under review" (no arXiv ID yet; add it once Part I is announced).
- The anonymized journal version is `main.tex` (default `pdflatex main.tex`).
