# Makefile — build the Chapter 3 manuscript PDF.
#
# Usage:
#     make            # produce main.pdf via pdflatex + bibtex + pdflatex + pdflatex
#     make quick      # single pdflatex pass (no bibliography updates)
#     make watch      # continuous rebuild via latexmk -pvc (must have latexmk)
#     make clean      # remove *.aux, *.log, *.out, *.bbl, *.blg, *.toc, *.pdf
#     make regen      # re-run md_to_tex.py to regenerate .tex files from .md
#     make wc         # word count via texcount
#
# Requires: pdflatex, bibtex (both TeX Live / MacTeX). latexmk / texcount optional.

MAIN     = main
SECTIONS = section_1_introduction.tex \
           section_2_related_work.tex \
           section_3_framework.tex \
           section_4_experiment_setup.tex \
           section_5_results.tex \
           section_6_discussion.tex \
           section_7_conclusion.tex \
           section_end_matter.tex
MD_SRC   = section_1_introduction.md \
           section_2_related_work.md \
           section_3_methods.md \
           section_4_results.md \
           section_5_discussion.md \
           section_6_threats.md \
           section_7_conclusion.md
BIB      = references.bib

.PHONY: all quick watch clean regen polish wc help

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(BIB)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	@echo ""
	@echo "  Built $(MAIN).pdf"

quick:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -view=none $(MAIN).tex

regen:
	@echo "WARNING: LaTeX files are now the source of truth. Running 'regen' may overwrite"
	@echo "manually polished LaTeX-only additions (such as Section 5.10 human-evaluation results)."
	@echo "If you are sure, run build/md_to_tex.py manually."
	@exit 1

polish:
	@python3 build/polish_tex.py \
	    section_1_introduction.tex section_2_related_work.tex \
	    section_3_framework.tex section_4_experiment_setup.tex \
	    section_5_results.tex section_6_discussion.tex \
	    section_7_conclusion.tex section_end_matter.tex

wc:
	@if command -v texcount > /dev/null 2>&1; then \
	    texcount -inc -total $(MAIN).tex; \
	else \
	    echo "texcount not installed; falling back to markdown wc"; \
	    wc -w $(MD_SRC) *.tex; \
	fi

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).out \
	      $(MAIN).bbl $(MAIN).blg $(MAIN).toc $(MAIN).fls \
	      $(MAIN).fdb_latexmk $(MAIN).synctex.gz \
	      *.aux
	@echo "Cleaned build artefacts"

help:
	@echo "Targets:"
	@echo "  all       — full pdflatex + bibtex + pdflatex + pdflatex build (default)"
	@echo "  quick     — single pdflatex pass (drafting cycle)"
	@echo "  watch     — continuous rebuild via latexmk -pvc"
	@echo "  regen     — regenerate .tex files from .md sources"
	@echo "  wc        — word count via texcount (or wc fallback)"
	@echo "  clean     — remove build artefacts"
