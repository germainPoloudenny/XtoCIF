Submission quick guide (CVPR and generic)

Overview
- Main (CVPR review): `main.tex` → anonymized CVPR‑style 8‑page paper with references. Supplementary is excluded from this PDF.
- Supplementary (CVPR review): `cvpr_supp.tex` → separate anonymized PDF from `sec/X_suppl.tex`.
- Optional duplicate: `cvpr_main.tex` builds the same CVPR main if needed.
- Generic (non‑CVPR): use `generic_supp.tex` for a standalone supplementary PDF; a generic main is no longer the default.

Build commands
- CVPR main (default): `latexmk -pdf main.tex`
- CVPR supplementary: `latexmk -pdf cvpr_supp.tex`
- Optional CVPR main duplicate: `latexmk -pdf cvpr_main.tex`
- Generic supplementary: `latexmk -pdf generic_supp.tex`

Anonymization toggle (generic build)
- File: `main.tex`
- Default: anonymized (`\anonymizedtrue`).
- After review/camera‑ready, set `\anonymizedfalse` to reveal authors and affiliations embedded in the `\else` branch.

Supplementary material
- Content lives in `sec/X_suppl.tex`.
- For CVPR upload, submit `cvpr_supp.pdf` as the separate supplementary file.
- For generic builds, `generic_supp.tex` produces a standalone supplementary PDF. Alternatively, set `\includesupptrue` in `main.tex` if you explicitly want appendix in the same PDF (not for CVPR).
- Do not add URLs, code repos, or acknowledgments that could deanonymize during review.

Notes
- The CVPR build (`cvpr_main.tex`) already uses `\usepackage[review]{cvpr}` and sets authors to `Anonymous CVPR submission`.
- Keep the supplementary excluded from `cvpr_main.tex` (the include is intentionally not present).
