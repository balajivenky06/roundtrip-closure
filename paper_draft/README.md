# paper_draft/

Assembly directory for the Chapter 3 manuscript LaTeX source.

## Layout

```
paper_draft/
  main.tex                        # top-level document; \input's the sections
  references.bib                  # bibliography (fill placeholders before submit)
  Makefile                        # build targets: all / quick / watch / clean / regen / wc
  README.md                       # this file
  build/
    md_to_tex.py                  # Markdown -> LaTeX converter

  # Body sections (edit .md; regenerate .tex via `make regen`)
  section_1_introduction.md/.tex
  section_2_related_work.md/.tex
  section_3_methods.md/.tex
  section_4_results.md/.tex
  section_5_discussion.md/.tex
  section_6_threats.md/.tex
  section_7_conclusion.md/.tex
  section_author_matter.md/.tex   # acknowledgements / funding / ethics / data avail.
  equations_and_algorithms.tex    # 20+ equations + Algorithms 1/2/3 (drop-in)
```

## Build

```
make          # full build: pdflatex + bibtex + pdflatex + pdflatex
make quick    # single pdflatex pass (drafting)
make clean    # remove build artefacts
```

Requires TeX Live (or MacTeX) with the following packages, all bundled by default:
`amsmath, amssymb, dsfont, algorithm, algpseudocode, booktabs, graphicx,
hyperref, cleveref, natbib`.

## Editing workflow

Edit the `.md` files (they are easier to read on GitHub and in code review).
Then regenerate the `.tex` files:

```
make regen    # converts *.md -> *.tex via build/md_to_tex.py
make          # rebuild PDF
```

Or edit the `.tex` files directly — but be aware that `make regen` will
overwrite them from the `.md` source.

## Placeholders

Before final submission, fill every `\PLACEHOLDER{...}` field. Grep them:

```
grep -n PLACEHOLDER *.tex
```

Common placeholders:
- Advisor name and co-author name (in `main.tex` title block and
  `section_author_matter.tex` CRediT block)
- Funding grant identifiers (`section_author_matter.tex` Funding block)
- IRB institution name (`section_author_matter.tex` Ethics block)
- arXiv preprint ID (`section_author_matter.tex` Preprint disclosure)
- CO2 estimate (`section_author_matter.tex` Environmental impact)

## Target venues

- **Primary:** Empirical Software Engineering (Springer Nature)
- **Secondary:** IEEE Transactions on Software Engineering
- **Conference derivative:** ICSE 2028 short paper or ICSE-SEIP if an industrial
  replication partner is found

## Related files outside this directory

- `../paper_equations.md` — original 21-equation working document (superseded
  by `equations_and_algorithms.tex` in this directory).
- `../chapter3_concept_note.md` — the pre-registered concept note. Deviations
  from pre-registration are documented in §6.5 of `section_6_threats.md`.
- `../tables/*.tex` — 11 paper-ready LaTeX tables to be `\input`'d from
  `section_4_results.tex`. Currently referenced by label but not yet
  `\input`'d; the intended pattern is:

  ```latex
  \input{../tables/tab_closure_rate_matrix.tex}
  ```

  Adjust to your build's include-path or copy the `tables/` directory
  into `paper_draft/` before final compilation.

- `../plots/output/*.png` — 10 paper-ready PNG figures. Referenced by label
  (`\ref{fig:closure_heatmap}` etc.) in the section drafts. Insert the actual
  `\begin{figure}\includegraphics{...}\end{figure}` blocks in the appropriate
  places before final compilation.
