# Paper sources

LaTeX sources for *"What Metric Scale Can a Frozen Foundation-Model Depth Prior
Give Monocular SLAM?"* (RA-L submission). Three build variants share the body
text:

| entry point | format | pages | notes |
|---|---|---|---|
| `main_ral_8p.tex` | IEEE conference (`ieeeconf.cls`) | 8 | RA-L initial submission, expanded body |
| `main_ral.tex` | IEEE conference (`ieeeconf.cls`) | 6 | RA-L initial submission, compact body |
| `main_final.tex` | IEEE journal (`IEEEtran.cls`) | 6 | camera-ready |

`main_ral*.tex` include `body_8p.tex` / `body.tex`; `main_final.tex` includes
`body.tex`. Bibliography is `caliprior.bib`; figures are in `figures/`
(the same PNGs as the top-level `figures/`). `IEEEtran.bst` is bundled so the
numeric IEEE style resolves without a system install of the IEEEtran package.

## Build

```bash
pdflatex main_ral_8p
bibtex   main_ral_8p
pdflatex main_ral_8p
pdflatex main_ral_8p
```

Replace the target with `main_ral` or `main_final` as needed. `main_final.tex`
requires `IEEEtran.cls` (TeX Live: `texlive-publishers`); `ieeeconf.cls` is
bundled.

Prebuilt PDFs are included: `main_ral_8p.pdf`, `main_ral.pdf`, `main_final.pdf`.
