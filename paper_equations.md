# Equations to add to the Chapter 3 paper

Working document. Each entry shows where the equation goes (§), gives LaTeX-ready source, surrounding motivation, and any derivation. Numbered globally so cross-references stay stable as we move text around.

---

## 0. Notation & symbol glossary

Goes in §3.1 (right after the introduction of the docstring-test-code triangle). Establish all symbols once; every later equation references this.

| Symbol | Meaning |
|---|---|
| $C$ | function code (ground-truth body) |
| $D$ | docstring describing $C$ |
| $T$ | reference test suite for $C$ |
| $C', D', T'$ | reconstructed artefacts produced by the pipeline |
| $L_{\text{spec}}, L_{\text{test}}, L_{\text{code}}$ | the three pipeline SLMs assigned to the three stages |
| $L_J$ | the external judge SLM (DeepSeek-R1, frozen, family-disjoint from $L_{\text{spec}}, L_{\text{test}}, L_{\text{code}}$) |
| $\mathcal{M} = \{m_1, \ldots, m_6\}$ | the set of pipeline SLMs (6-model lineup) |
| $\mathcal{S} = \{\textsf{spec}, \textsf{test}, \textsf{code}\}$ | the three pipeline stages |
| $a : \mathcal{S} \to \mathcal{M}$ | a *cell assignment*: which model owns which stage |
| $K(T', C)$ | mutation kill rate of test suite $T'$ against $C$ (Eq. 5) |
| $R(C', T)$ | reference test pass rate of $C'$ against $T$ (Eq. 6) |
| $B(D_1, D_2)$ | BERTScore-F1 between two docstrings (Eq. 7) |
| $J(x, y) \in \{-1, 0, 1, 2, 3, 4\}$ | external judge rating; $-1$ means parse failure |
| $\tau$ | mutation kill-rate validity threshold ($\tau = 0$ in our experiments — any kill counts) |
| $\rho$ | judge-rating validity threshold ($\rho = 3$: "equivalent or better") |
| $\mathds{1}[\cdot]$ | indicator function (1 if true, 0 otherwise) |
| $N$ | per-cell sample size (150 mono, 75 hetero/null) |

---

# §3 Methods — formal definitions

## 1. The round-trip closure operator

> The docstring-test-code triangle defines three closure paths. We model each path as a composition of two pipeline stages followed by an evaluation step.

**LaTeX:**

```latex
\textbf{Round-trip closure paths.} Let $a$ be a cell assignment.
The three closure paths are defined as

\begin{align}
  \text{Path 1 (}C \to D \to T\text{):} \quad
  T' &\;=\; L_{\text{test}}\bigl(L_{\text{spec}}(C)\bigr), \label{eq:path1}\\
  \text{Path 2 (}D \to T \to C\text{):} \quad
  C' &\;=\; L_{\text{code}}\bigl(D,\; L_{\text{test}}(D)\bigr), \label{eq:path2}\\
  \text{Path 3 (}C \to T \to D\text{):} \quad
  D' &\;=\; L_{\text{spec}}\bigl(L_{\text{test}}(C)\bigr). \label{eq:path3}
\end{align}
```

**Goes in:** §3.2 (Closure paths). Right after introducing the triangle.

**Note:** Path 2 explicitly conditions $L_{\text{code}}$ on both the original docstring $D$ *and* the freshly-generated tests $T'$ — this matches our `_call_code_from_doc_tests` implementation and is the strongest synthesis context.

---

## 2. Closure validity predicate

> A round-trip closes iff the closure metric exceeds threshold AND the external judge agrees.

**LaTeX:**

```latex
\textbf{Closure validity.} For Path 1, given mutation kill rate
$K(T', C)$ and judge rating $J(D, D')$:
\begin{equation}
  \text{valid}_1(C, D, D', T')
  \;=\;
  \mathds{1}\!\bigl[\, K(T', C) > \tau \,\bigr]
  \;\wedge\;
  \mathds{1}\!\bigl[\, J(D, D') \geq \rho \,\bigr].
  \label{eq:valid1}
\end{equation}
Analogous definitions hold for Path 2 (with reference pass rate $R$) and
Path 3 (with BERTScore $B$), see Eqs.~\ref{eq:valid2} and~\ref{eq:valid3}.
```

**Goes in:** §3.3 (Validity criterion).

**Add in body:**

```latex
\begin{equation}
  \text{valid}_2 = \mathds{1}[R(C', T) > 0] \wedge \mathds{1}[J(C, C') \geq \rho],
  \label{eq:valid2}
\end{equation}
\begin{equation}
  \text{valid}_3 = \mathds{1}[B(D, D') > 0] \wedge \mathds{1}[J(D, D') \geq \rho].
  \label{eq:valid3}
\end{equation}
```

---

## 3. Test-suite filter (a clarifying defn)

> Before computing mutation kill rate, we discard generated tests that fail on the original (ground-truth) code; this avoids penalising mutants for surviving spuriously-wrong tests.

**LaTeX:**

```latex
\textbf{Test filter.} Let $T' = \{t_1, t_2, \ldots\}$ be the candidate test
set. The filtered suite is
\begin{equation}
  T'_{\text{filt}}(C)
  \;=\;
  \bigl\{\, t \in T' \;:\; \text{exec}(t, C) = \text{pass} \,\bigr\}.
  \label{eq:filter}
\end{equation}
```

**Goes in:** §3.4 (Mutation testing protocol). Introduces $T'_{\text{filt}}$ used in Eq. 5.

---

## 4. Cell assignment formalism (the DOE backbone)

> A cell is a stage-to-model assignment; cells are the units of analysis.

**LaTeX:**

```latex
\textbf{Cell assignment.} A cell is a function
$a: \mathcal{S} \to \mathcal{M} \cup \{\bot\}$
assigning each pipeline stage to an SLM (or to $\bot$, denoting an
ablated stage). The DOE specifies 20 such cells:
\begin{equation}
  \mathcal{C}_{\text{DOE}} \;=\;
  \underbrace{\bigl\{a : a(\textsf{spec}) = a(\textsf{test}) = a(\textsf{code})\bigr\}}_{\text{Mono stratum (6 cells)}}
  \;\cup\;
  \underbrace{\bigl\{a : |\{a(s) : s \in \mathcal{S}\}| = 3\bigr\}}_{\text{Hetero stratum (11 cells)}}
  \;\cup\;
  \underbrace{\mathcal{C}_{\text{null}}}_{\text{3 cells}}.
  \label{eq:doe}
\end{equation}
```

**Goes in:** §3.5 (DOE table introduction).

---

## 5. Mutation kill rate (Path 1 metric)

> Path 1 success is measured by how many code mutations the LLM-generated tests detect.

**LaTeX:**

```latex
\textbf{Mutation kill rate.} Given the filtered test suite
$T'_{\text{filt}}$ and the set of mutants
$\mathcal{M}_C = \{m_1, \ldots, m_n\}$ generated from $C$:
\begin{equation}
  K(T', C)
  \;=\;
  \frac{\bigl|\{\, m \in \mathcal{M}_C \;:\; \exists t \in T'_{\text{filt}}(C),\; \text{exec}(t, m) = \text{fail} \,\}\bigr|}{\bigl|\mathcal{M}_C\bigr| \;-\; \bigl|\mathcal{M}_C^{\text{equiv}}\bigr|}.
  \label{eq:killrate}
\end{equation}
A mutant is *killed* if at least one filtered test fails on it. Equivalent
mutants $\mathcal{M}_C^{\text{equiv}}$ are excluded from the denominator
following Chapter 2's protocol.
```

**Goes in:** §3.6 (Closure metrics → Path 1).

---

## 6. Reference test pass rate (Path 2 metric)

> Path 2 success is whether the reconstructed code passes the original-suite test cases.

**LaTeX:**

```latex
\textbf{Reference test pass rate.} Let $T = \{t_1, \ldots, t_k\}$ be the
ground-truth reference suite for $C$. The pass rate is
\begin{equation}
  R(C', T)
  \;=\;
  \frac{1}{|T|}\sum_{i=1}^{|T|} \mathds{1}\!\bigl[\,\text{exec}(t_i, C') = \text{pass}\,\bigr].
  \label{eq:passrate}
\end{equation}
In this experiment we use a binary indicator ($R \in \{0, 1\}$ — the suite
either fully passes or does not) for §4 statistical analysis; per-test
fractions are reported in Appendix~\ref{app:per_test}.
```

**Goes in:** §3.6 (Path 2 subsection).

---

## 7. BERTScore F1 (Path 3 metric)

> Path 3 success is the semantic similarity between the original and reconstructed docstring.

**LaTeX:**

```latex
\textbf{Docstring BERTScore.} For two docstrings $D_1$ and $D_2$, let
$\phi(D)$ denote their RoBERTa-large contextual embeddings (token-level).
Precision and recall under cosine similarity are
\begin{align}
  P(D_1, D_2) &\;=\; \frac{1}{|\phi(D_2)|} \sum_{x \in \phi(D_2)} \max_{y \in \phi(D_1)} \cos(x, y), \label{eq:bert_p}\\
  R(D_1, D_2) &\;=\; \frac{1}{|\phi(D_1)|} \sum_{y \in \phi(D_1)} \max_{x \in \phi(D_2)} \cos(x, y), \label{eq:bert_r}
\end{align}
and the BERTScore F1 metric we report is
\begin{equation}
  B(D_1, D_2)
  \;=\;
  \frac{2 \cdot P(D_1, D_2) \cdot R(D_1, D_2)}{P(D_1, D_2) + R(D_1, D_2)}.
  \label{eq:bertscore}
\end{equation}
We use baseline-rescaled scores so $B \in (-\infty, 1]$ with $B = 0$
denoting "no better than random pairing".
```

**Goes in:** §3.6 (Path 3 subsection).

---

# §4 Results — statistical decomposition

## 8. Type-III ANOVA model

> To attribute variance in closure metrics to cell choice vs sample variability, we fit a Type-III ANOVA.

**LaTeX:**

```latex
\textbf{Type-III ANOVA.} For each closure metric $Y$ (kill rate, pass rate, BERTScore), we fit
\begin{equation}
  Y_{ij} \;=\; \mu \;+\; \alpha_i \;+\; \beta_j \;+\; \varepsilon_{ij},
  \quad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2),
  \label{eq:anova}
\end{equation}
where $\alpha_i$ is the fixed effect of cell $i \in \mathcal{C}_{\text{DOE}}$,
$\beta_j$ is the fixed effect of sample (function) $j$, and $\varepsilon_{ij}$
is residual error. Type-III sums of squares are used so that
$\alpha_i$ and $\beta_j$ are tested after adjusting for each other.
```

**Goes in:** §4.2 (Statistical analysis).

---

## 9. F-statistic and significance test

> The headline ANOVA F-test asks: do cells differ in mean closure performance after sample variance is removed?

**LaTeX:**

```latex
\textbf{F-statistic for cell effect.} The F-statistic
\begin{equation}
  F_{\text{cell}}
  \;=\;
  \frac{MS_{\text{cell}}}{MS_{\text{within}}}
  \;\sim\; F(d_{\text{cell}}, d_{\text{err}})
  \label{eq:fstat}
\end{equation}
under $H_0: \alpha_1 = \cdots = \alpha_{|\mathcal{C}_{\text{DOE}}|}$.
Degrees of freedom: $d_{\text{cell}} = |\mathcal{C}_{\text{DOE}}| - 1 = 19$,
$d_{\text{err}} = N \cdot |\mathcal{C}_{\text{DOE}}| - |\mathcal{C}_{\text{DOE}}| - N + 1$.
```

**Goes in:** §4.2 (right after Eq. 8).

---

## 10. Tukey HSD for pairwise cell comparisons

> Post-ANOVA, we compare each pair of cells while controlling family-wise error.

**LaTeX:**

```latex
\textbf{Tukey honest significant difference.} Pairwise mean differences
between cells $i$ and $k$ are tested via
\begin{equation}
  q_{ik}
  \;=\;
  \frac{\bar{Y}_i - \bar{Y}_k}{\sqrt{MS_{\text{within}} / n}},
  \label{eq:tukey}
\end{equation}
where $n$ is the harmonic mean of per-cell sample sizes (we use the
unbalanced-design extension since mono cells have $n = 150$ and
hetero/null cells have $n = 75$ or $n = 20$). A pair is significant at
family-wise $\alpha = 0.05$ when $|q_{ik}|$ exceeds the studentised range
critical value $q_{0.05, |\mathcal{C}_{\text{DOE}}|, d_{\text{err}}}$.
```

**Goes in:** §4.3 (Pairwise comparisons).

---

## 11. Mixed-effects logistic regression (per-stage decomposition)

> To isolate which stage assignment drives closure, we treat the cell as the product of three stage choices and fit a generalised linear mixed model.

**LaTeX:**

```latex
\textbf{Mixed-effects logit for stage attribution.} Let
$\text{closure}_{ij} \in \{0, 1\}$ denote whether function $j$ closed
under cell $i$. We model
\begin{equation}
  \text{logit}\bigl(\Pr[\text{closure}_{ij} = 1]\bigr)
  \;=\;
  \beta_0
  + \beta_1 \cdot a_i(\textsf{spec})
  + \beta_2 \cdot a_i(\textsf{test})
  + \beta_3 \cdot a_i(\textsf{code})
  + \gamma_j
  \label{eq:mixedlogit}
\end{equation}
where $\gamma_j \sim \mathcal{N}(0, \sigma_{\text{sample}}^2)$ is a random
intercept for sample, and $a_i(\textsf{stage})$ is a categorical
indicator over $\mathcal{M}$. Estimates $\hat\beta_1, \hat\beta_2, \hat\beta_3$
quantify each stage's contribution to closure odds.
```

**Goes in:** §4.4 (Per-stage decomposition).

---

## 12. Cohen's κ for judge–human agreement

> The external judge's reliability is validated against a 60-pair human-annotated subset.

**LaTeX:**

```latex
\textbf{Judge--human agreement.} For each binarised rating
($J \geq \rho$ vs $J < \rho$), Cohen's $\kappa$ is
\begin{equation}
  \kappa
  \;=\;
  \frac{p_o - p_e}{1 - p_e}
  \label{eq:kappa}
\end{equation}
where $p_o$ is the observed agreement proportion and $p_e$ is the
chance-expected proportion under independence.
$\kappa \in [-1, 1]$; we follow Landis and Koch's bands ($> 0.6$:
substantial; $> 0.8$: almost perfect).
```

**Goes in:** §4.5 (Judge validation).

---

## 13. Krippendorff's α for multi-rater reliability

> With three judges (DeepSeek-R1 + 2 humans), we report Krippendorff's α for ordinal ratings.

**LaTeX:**

```latex
\textbf{Inter-rater reliability.} For ordinal ratings $r_{ij}$ from
rater $i$ on item $j$, Krippendorff's
\begin{equation}
  \alpha
  \;=\;
  1 \;-\; \frac{D_o}{D_e},
  \quad
  D_o \;=\; \tfrac{1}{n}\sum_{j} \tfrac{1}{\binom{m_j}{2}}
            \sum_{i_1 < i_2} \delta(r_{i_1 j}, r_{i_2 j})^2,
  \label{eq:alpha}
\end{equation}
where $\delta$ is the ordinal difference metric and $D_e$ is the
expected disagreement under random pairing. Values $\alpha > 0.667$
are considered acceptable for inferential use; $\alpha > 0.8$ is
high.
```

**Goes in:** §4.5 (right after Eq. 12).

---

# §5 Discussion — novel formalisms (your contribution beyond the experiment)

## 14. Stage-strength score

> The DOE makes per-stage strengths estimable: for each (model, stage) pair, we can ask "how often does $m$ at stage $s$ lead to closure?"

**LaTeX:**

```latex
\textbf{Stage-strength score.} For model $m \in \mathcal{M}$ and stage
$s \in \mathcal{S}$, define
\begin{equation}
  S(m, s)
  \;=\;
  \frac{1}{|\mathcal{C}_{m,s}|}
  \sum_{a \in \mathcal{C}_{m,s}}
  \;
  \mathbb{E}_{j}\!\bigl[\text{valid}(a, j) \;\big|\; a(s) = m\bigr],
  \label{eq:stagestrength}
\end{equation}
where $\mathcal{C}_{m, s} = \{a \in \mathcal{C}_{\text{DOE}} : a(s) = m\}$
is the set of cells assigning $m$ to stage $s$, and the expectation is
over the 150 sampled functions. $S(m, s) \in [0, 1]$ where 1 means
"$m$ at stage $s$ always closes". The marginal effect of moving $m$
from stage $s$ to stage $s'$ is $S(m, s) - S(m, s')$.
```

**Goes in:** §5.2 (Per-stage strengths — *new* formalism). Cite from main results table.

---

## 15. Heterogeneity gain

> The heterogeneous-SLM thesis claims hetero ≥ best mono; we quantify this directly.

**LaTeX:**

```latex
\textbf{Heterogeneity gain.} The heterogeneity gain is
\begin{equation}
  G
  \;=\;
  \max_{a \in \mathcal{C}_{\text{hetero}}}
  \;
  \mathbb{E}_{j}\!\bigl[\text{valid}(a, j)\bigr]
  \;-\;
  \max_{a \in \mathcal{C}_{\text{mono}}}
  \;
  \mathbb{E}_{j}\!\bigl[\text{valid}(a, j)\bigr].
  \label{eq:heterogain}
\end{equation}
$G > 0$ confirms the thesis; $G \approx 0$ indicates the best hetero
assignment merely matches the best mono baseline. The 95\% confidence
interval is bootstrapped over functions.
```

**Goes in:** §5.1 (Headline result).

---

## 16. Cheap-drafter asymmetry

> The H4–H5 contrast (cheap model at synthesis vs at spec) is best captured as a stage-position effect on a fixed small model.

**LaTeX:**

```latex
\textbf{Cheap-drafter asymmetry.} For a small model $m_{\text{small}}$
(here, \texttt{llama3.2:3b}), the cheap-drafter position effect is
\begin{equation}
  \Delta_{\text{cheap}}(m_{\text{small}})
  \;=\;
  S(m_{\text{small}}, \textsf{code})
  \;-\;
  S(m_{\text{small}}, \textsf{spec}).
  \label{eq:cheap}
\end{equation}
$\Delta > 0$ confirms our finding that small drafters help more at
synthesis than at specification.
```

**Goes in:** §5.3 (Cheap-drafter asymmetry).

---

## 17. Per-stage variance decomposition

> The mixed-logit (Eq. 11) gives stage-contribution coefficients; we decompose total variance over the three stages.

**LaTeX:**

```latex
\textbf{Stage variance decomposition.} The proportion of explained
variance attributable to stage $s$ is
\begin{equation}
  V_s
  \;=\;
  \frac{\text{Var}(\hat\beta_s \cdot a(s))}{\sum_{s' \in \mathcal{S}} \text{Var}(\hat\beta_{s'} \cdot a(s'))}.
  \label{eq:stagevar}
\end{equation}
We expect $V_{\text{test}} > V_{\text{code}} > V_{\text{spec}}$
(test generation is the bottleneck, code synthesis is robust, spec
generation is relatively model-invariant) — this is testable against the
fitted coefficients in §4.4.
```

**Goes in:** §5.4 (Where in the pipeline does model choice matter most?).

---

# §6 Limitations — null cell expected behaviour

## 18. Null cell expected closure under corruption

> The N1 prompt-shuffled control's expected closure rate should be near zero; we formalise the expectation as a bound.

**LaTeX:**

```latex
\textbf{Null cell upper bound.} For the prompt-shuffled control N1,
where each first-stage input is word-shuffled before being passed to
the pipeline, the expected closure rate satisfies
\begin{equation}
  \mathbb{E}_{j}\!\bigl[\text{valid}_p(\text{N1}, j)\bigr]
  \;\leq\;
  \Pr\bigl[J(\text{orig}, \text{shuffled}) \geq \rho\bigr]
  \;+\;
  \epsilon_{\text{spurious}},
  \label{eq:nullbound}
\end{equation}
for each path $p \in \{1, 2, 3\}$, where $\epsilon_{\text{spurious}}$
accounts for trivial agreement (e.g., very short artefacts where
ordering is irrelevant). A measured closure rate substantially above
this bound would falsify the closure metric as a meaningful signal.
```

**Goes in:** §6.1 (Validity threats and null controls).

---

## 19. Stage-ablation expected behaviour

> N2 and N3 quantify what happens when L_spec or L_test is removed entirely.

**LaTeX:**

```latex
\textbf{Stage ablation invariants.} For N2 ($L_{\text{spec}} = \bot$):
\begin{align}
  \text{valid}_1(\text{N2}, j) &\;=\; 0 \quad \forall j, \label{eq:n2_p1}\\
  \text{valid}_3(\text{N2}, j) &\;=\; 0 \quad \forall j, \label{eq:n2_p3}
\end{align}
because both Path 1 and Path 3 require $L_{\text{spec}}$ in their
first or last stage; only Path 2 (which conditions $L_{\text{code}}$ on the
ground-truth $D$) can succeed. Analogously for N3 ($L_{\text{test}} = \bot$),
all three paths fail by construction since every path requires
$L_{\text{test}}$.
```

**Goes in:** §6.1 (after Eq. 18). Provides the formal justification for
the gate logic that excludes ablated paths from cell-validity counts.

---

# Appendix — derivations + accounting

## 20. Cache hit rate and amortised cost

> A key reproducibility lever: the disk-keyed LLM cache amortises repeated calls across reruns and across cells.

**LaTeX:**

```latex
\textbf{Cache hit rate.} Over the full sweep, the LLM cache hit rate is
\begin{equation}
  h
  \;=\;
  \frac{n_{\text{hits}}}{n_{\text{hits}} + n_{\text{misses}}},
  \label{eq:hitrate}
\end{equation}
and the amortised wall-clock cost per cell, given mean miss-latency
$\tau_{\text{miss}}$, is
\begin{equation}
  \mathbb{E}[\text{cell time}]
  \;=\;
  N \cdot |\text{paths}| \cdot |\text{stages}| \cdot (1 - h) \cdot \tau_{\text{miss}}
  \;+\;
  o(N).
  \label{eq:cellcost}
\end{equation}
With our observed $h \approx 0.6$ across hetero cells, the cost drops
to roughly 40\% of the cold-run cost — a critical enabler for the
20-cell DOE on free-tier compute.
```

**Goes in:** Appendix C (Reproducibility + cost analysis).

---

## 21. Per-test pass rate (refinement of Eq. 6)

> Reviewers will ask: why binary pass-rate not fractional? The fractional version is in Appendix.

**LaTeX:**

```latex
\textbf{Per-test pass rate (Appendix variant).} The per-test pass rate
splits the suite into individual tests:
\begin{equation}
  R_{\text{per-test}}(C', T)
  \;=\;
  \frac{1}{|T|}
  \sum_{t \in T}
  \mathds{1}[\text{exec}(t, C') = \text{pass}].
  \label{eq:passrate_fine}
\end{equation}
We use the binary form (Eq.~\ref{eq:passrate}) in the main analysis to
stabilise the ANOVA against per-test correlation; Appendix~\ref{app:per_test}
reports both formulations.
```

**Goes in:** Appendix B.

---

# Cross-reference map

| Equation | Cited from | Key insight |
|---|---|---|
| Eq. 1–3 (paths) | §3.2, §4.1, §5 | the experimental skeleton |
| Eq. 4 (validity) | §3.3, §4.5 | binary outcome for logit |
| Eq. 5 (kill rate) | §3.6, §4.1 | Path 1 success |
| Eq. 6, 21 (pass rates) | §3.6, §4.1, App B | Path 2 success |
| Eq. 7 (BERTScore) | §3.6, §4.1 | Path 3 success |
| Eq. 8 (ANOVA) | §4.2 | overall test of cell effect |
| Eq. 9 (F-stat) | §4.2 | significance |
| Eq. 10 (Tukey) | §4.3 | pairwise comparisons |
| Eq. 11 (mixed logit) | §4.4 | per-stage decomposition |
| Eq. 12 (κ) | §4.5 | judge–human agreement |
| Eq. 13 (α) | §4.5 | multi-rater reliability |
| Eq. 14 (stage-strength) | §5.2 | *novel* — per-(model, stage) |
| Eq. 15 (heterogeneity gain) | §5.1 | *novel* — quantifies thesis |
| Eq. 16 (cheap-drafter Δ) | §5.3 | *novel* — H4 vs H5 |
| Eq. 17 (stage variance) | §5.4 | *novel* — which stage matters most |
| Eq. 18 (null bound) | §6.1 | validates the metric |
| Eq. 19 (ablation invariants) | §6.1 | justifies gate logic |
| Eq. 20 (cache) | App C | reproducibility cost |

---

# Implementation notes

- Use `align` environment for multi-line related equations; use `equation` for standalone numbered ones (so they're individually citable).
- Eq. labels start with `eq:` for clarity in the LaTeX source.
- For Eqs. 14–17 (the *novel* formalisms), consider boxing in the paper LaTeX with `\boxed{\cdots}` to mark them as the paper's mathematical contributions.
- The Krippendorff-α formulation in Eq. 13 simplifies the general case to ordinal ratings; the full formula (with arbitrary metrics) is in the appendix.

# Next steps after the experiments finish

1. Plug the final numbers (mean kill_rate, ANOVA F, Tukey p-values, mixed-logit $\hat\beta$s) into Eqs. 8–17.
2. Compute and report the actual values of $G$ (Eq. 15) and $\Delta_{\text{cheap}}$ (Eq. 16) — these are the paper's headline numbers.
3. For Eq. 14, produce a heatmap of $S(m, s)$ over all (model, stage) pairs — that becomes a key figure.
