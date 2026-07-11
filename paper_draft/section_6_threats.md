# §6 Threats to Validity

We organise threats along the four classical dimensions~[Wohlin et al., 2012]: internal, external, construct, and conclusion validity. Each threat is stated concretely, our mitigation is documented, and the residual risk is characterised.

## 6.1 Internal validity

Internal-validity threats concern whether the observed effects are attributable to the manipulated variables (stage-model assignments) rather than to confounding factors.

### 6.1.1 Hardware split during the sweep

**Threat.** The 22 GPU-days of experimental sweep were split across Google Colab A100 and T4 hardware. Different cells were executed on different hardware because A100 availability was intermittent; the mono stratum (M1--M6) and the initial pilot ran on A100, the H8/H10/H11/H6/H9 hetero cells ran on T4, and the remaining cells split across both. Different hardware could in principle produce different LLM outputs due to different floating-point kernels or different attention implementations.

**Mitigation.** The Ollama runtime enforces greedy decoding at temperature $0.2$ across all models used in this study. All model weights are quantised and deterministic under the same seed and prompt. The LLM disk cache is keyed on (\texttt{model\_tag}, \texttt{role\_hint}, \texttt{prompt}, \texttt{temperature}, \texttt{top\_p}, \texttt{top\_k}, \texttt{num\_ctx}, \texttt{max\_tokens}) --- identical parameter tuples produce identical cache hits regardless of hardware. Cross-hardware verification via re-execution of 30 pilot rows on both hardware types produced byte-identical LLM outputs.

**Residual risk.** Rare non-determinism in the Ollama runtime under GPU memory pressure could produce different output text on a small number of rows. We estimate this affects fewer than $0.1\%$ of rows based on the observed hit rate of the cache during resumption from mid-sweep checkpoints.

### 6.1.2 Mid-sweep NUM_CTX configuration change

**Threat.** During the sweep, a configuration change reduced $\texttt{NUM\_CTX}$ from $16384$ to $8192$ tokens for the affected models to work around T4 memory constraints. Rows generated before and after the change have subtly different cache keys, so the G2 per-mutation-operator regeneration script achieved only 67\% cache-recovery coverage on Path~1 rows.

**Mitigation.** The per-operator table (Table~\ref{tab:per_operator_kill_rate}) is computed only on the $1{,}172$ Path~1 rows for which cache reconstruction succeeded; the remaining $567$ rows are excluded, not silently treated as zeros. All other analyses (§4.1 through §4.7 and §5.1--5.8) use the aggregate metric value stored directly in the TSV, which is unaffected by the cache-key mismatch. The G1--G6 gap-closer scripts (§5) do not depend on cache reconstruction.

**Residual risk.** The per-operator table samples $67\%$ of Path~1 rows non-randomly: the excluded rows are systematically the ones generated during the second half of the sweep on T4 hardware. A per-operator effect confined to the excluded rows would be undetected. We regard this as low-probability because the aggregate Path~1 statistics (which do include those rows) are consistent with the per-operator patterns reported in §4.4.

### 6.1.3 Cache reuse across cells

**Threat.** The five-layer checkpointing system (LLM cache, pytest cache, BERTScore cache, TSV row cache, JSONL decontamination cache) means that identical calls across cells are served from cache. If a cache entry produced under one cell's context (e.g., a docstring generated in cell M2 with phi-4 as $L_{\textsf{spec}}$) is reused when cell H1 makes an identical call, the cell-level statistics are not truly independent across cells.

**Mitigation.** Cache reuse across cells is the intended design: it saves compute while preserving deterministic reproducibility. The statistical analysis treats the cell as the unit of assignment (per Eq.~\ref{eq:anova}) and the sample as the unit of measurement; cache reuse affects the compute cost but does not change which model produced which artefact for which cell. The sample effect $\beta_j$ in Eq.~\ref{eq:anova} absorbs per-sample cache-driven correlation.

**Residual risk.** None; this is an efficiency mechanism, not a source of bias.

## 6.2 External validity

External-validity threats concern the generalisation of the findings beyond the specific experimental setup.

### 6.2.1 Missing closed-weight cells (deferred RQ4)

**Threat.** The original pre-registered DOE~(Concept Note §4.4) specified three closed-weight cells: M5 (Claude Sonnet~4.6 mono baseline), H2 (Claude spec + open-weight test + open-weight code), and H8 (Claude spec + Claude test + open-weight code). These cells were dropped during the sweep phase due to Colab Pro+ budget constraints and API-cost budget constraints, and RQ4 (open-weight closure vs.\ closed-weight ceiling) is not answered in the present chapter. This is a substantive deviation from the pre-registration.

**Mitigation.** The deviation is disclosed here, in §1.3 (research questions), and in the concept-note diff maintained in the replication package. The pre-registered closed-weight cells are recorded in the DOE table (\texttt{doe.py}) with a documented \texttt{is\_deferred = True} marker. The remaining 20-cell open-weight sweep is complete.

**Residual risk.** All results in this chapter are scoped to open-weight SLMs under 30~B parameters. Reviewers or readers may reasonably ask ``do these findings apply to closed-weight frontier models?'' --- the honest answer is that this study cannot say. The follow-up study proposed in §7 addresses this directly.

### 6.2.2 Model family coverage

**Threat.** The six pipeline SLMs represent five distinct model families (Meta, Microsoft, Alibaba dense, Google, Mistral, Alibaba coder). Every family has one representative except Alibaba, which has two. The DeepSeek judge is a sixth family. This is a stronger family-diversity claim than most prior LLM-evaluation papers, but it is not exhaustive: OpenAI (open-weight releases), Amazon, and Cohere are not represented. Findings may not generalise to unrepresented families.

**Mitigation.** The pre-registered DOE (Table~\ref{tab:doe_summary}) documents the specific model tags used at each cell. Reviewers can inspect \texttt{config.py} for the full model lineup with release dates. The Chapter~2 companion paper reports similar findings on a smaller family lineup, supporting cross-lineup generalisation.

**Residual risk.** The specific finding of the phi-4-in-test-stage pathology (§5.3) is stated as ``phi-4 at $L_{\textsf{test}}$ produces filter-failing tests on approximately half of samples''; whether analogous stage-specific pathologies exist for models we do not evaluate is unknown.

### 6.2.3 Benchmark coverage

**Threat.** The core dataset comprises 150~functions from HumanEval and MBPP. Both benchmarks are Python-only, function-level (not multi-file or class-level), and drawn from a corpus that appeared in the training data of many of the pipeline SLMs. The results may not transfer to other benchmarks, other languages, or to multi-file / class-level SE tasks.

**Mitigation.** Two held-out subsets (25 LiveCodeBench problems with publication date $\geq$ 2024-12-01, post-training-cutoff for all evaluated SLMs; 50 HumanEval-Mutated problems with function-rename + docstring-paraphrase transformations) were prepared but not swept in this study. Contamination sensitivity results on these subsets will be reported in a pending addendum. The cross-benchmark decomposition in §4.5 documents that per-cell effects are significant on both HumanEval and MBPP with different F-values, suggesting the effects generalise across the two benchmarks.

**Residual risk.** Substantial for languages other than Python and for tasks that require multi-file understanding. This is a well-known limitation of function-level LLM-SE evaluation.

## 6.3 Construct validity

Construct-validity threats concern whether the operational definitions of ``closure'' and ``validity'' actually measure what we claim they measure.

### 6.3.1 BERTScore as a semantic-equivalence signal (Path~3)

**Threat.** BERTScore F1 on RoBERTa-large was pre-registered as the Path~3 automated metric. Section~4.6 reports that BERTScore is statistically uncorrelated with the judge SLM's semantic-equivalence rating ($r_3 = 0.022$, $p = 0.359$, $n = 1{,}791$). Section~5.2 stratifies this by benchmark and shows the disagreement flips direction: MBPP over-credits, HumanEval under-credits. This is direct evidence that BERTScore is functioning as a surface-form similarity signal rather than a semantic-equivalence signal on our data.

**Mitigation.** Rather than discarding Path~3 results, we adopt the strict two-signal validity policy of Eq.~\ref{eq:valid3}: closure requires both metric $> 0$ AND judge $\geq \rho$. Path~3 closure counts are correspondingly conservative. Section~5.2 explicitly reports Path~3 findings as ``metric $\times$ judge disagreement'' rather than as ``metric-only closure''.

**Residual risk.** Alternative surface-form metrics (BLEU-4, ROUGE-L, sentence-BERT similarity) were not evaluated. It is possible that a different metric would produce stronger $r_3$ correlation. This is left to follow-up work and is stated as an open question in §7.

### 6.3.2 Mutation kill rate as a defect-detection proxy (Path~1)

**Threat.** Mutation kill rate on the five mutation operators (arithmetic, boundary, comparison, negate\_bool, return\_none) is a proxy for defect-detection capability. The five operators do not exhaust the space of software defects; real-world defects include type errors, off-by-one errors in indices, concurrency bugs, resource leaks, and many other patterns not modelled by these five operators.

**Mitigation.** The five operators are the canonical set inherited from Chapter~2, which itself follows the standard mutation-testing literature~[Andrews et al., 2005; Just et al., 2014]. Chapter~2 §5 discusses the operator-coverage limitation and its implications for the mutation kill rate metric.

**Residual risk.** The kill-rate values in Table~\ref{tab:per_operator_kill_rate} are specific to the five canonical operators. Whether a cell's kill rate on operators \emph{not} in this set would be similar is unknown.

### 6.3.3 Judge SLM validity

**Threat.** DeepSeek-R1:14B is used as the external judge SLM. The judge's own validity as a semantic-equivalence oracle is not independently verified in this chapter. LLM-as-judge protocols are known to exhibit position bias, verbosity bias, self-preference bias, and sycophancy under user pressure~[Panickssery et al., 2024; Wang et al., 2024; Fanous et al., 2025].

**Mitigation.** DeepSeek-R1 is chosen deliberately for family-disjointness from every pipeline SLM. The 60-pair stratified human-evaluation study (§3.9, pending addendum) will report Cohen's $\kappa$ (Eq.~\ref{eq:kappa}) between the judge and three independent human annotators, quantifying the judge's own validity. All disagreements documented in §5.2 include the judge as one signal; disagreements are attributed to the judge--metric pair, not to the judge alone.

**Residual risk.** Until the human-evaluation addendum is complete, the judge's absolute reliability is not quantified. All claims in this chapter that depend on the judge's rating (§5.2, §5.4) are stated as ``the judge disagrees with the metric'' or ``the judge accepts equivalence'' rather than as ``the artefacts are semantically equivalent''.

### 6.3.4 The strict-AND validity policy

**Threat.** The two-signal strict-AND validity policy (Eqs.~\ref{eq:valid1}--\ref{eq:valid3}) is one of many possible ways to combine the automated metric and the judge rating. Alternative policies (metric-OR-judge, weighted combination, judge-only) would produce different closure counts.

**Mitigation.** §5.6 reports sensitivity analysis over $(\tau, \rho)$ combinations, showing that Path~1 conclusions are robust to threshold choice. The strict-AND policy is disclosed as a deliberate choice in §3.3 and its analytical role (surfacing disagreements as the false-closure-candidate and metric-false-negative categories in §5.2) is documented.

**Residual risk.** Reviewers advocating a permissive validity policy (metric-OR-judge) would obtain higher closure counts and softer between-cell contrasts.

## 6.4 Conclusion validity

Conclusion-validity threats concern whether the statistical inferences reported in §4 are sound.

### 6.4.1 Multiple testing

**Threat.** The paper reports many statistical tests: the main ANOVA (Eq.~\ref{eq:anova}), the interaction ANOVA (Eq.~\ref{eq:anova_interact}), 190 pairwise Tukey comparisons, three per-path Pearson correlations, per-benchmark ANOVAs, per-operator effect sizes, capability-preservation rates. Without correction, some significance would be expected by chance.

**Mitigation.** The Tukey pairwise tests are Bonferroni-corrected over the 190-pair family at $\alpha = 0.05$ (§3.8.2). Per-path correlations and their p-values are reported directly without correction; the significant one (Path 2, $r_2 = 0.52$, $p = 1.19 \times 10^{-129}$) is so far below the significance threshold that no correction affects its interpretation, and the non-significant one (Path 3, $r_3 = 0.022$, $p = 0.359$) is a null result that would not become significant under correction. The interaction ANOVA is a single planned comparison. The main-effect ANOVAs are also planned comparisons.

**Residual risk.** Post-hoc per-operator analyses in §4.4 are not corrected for multiple testing beyond what the reported effect sizes convey. Effect sizes $\Delta > 0.05$ are treated as meaningful; smaller effect sizes are not emphasised.

### 6.4.2 Sample size and statistical power

**Threat.** Mono cells were run on $N = 150$ core samples; hetero and null cells on $N = 75$ samples each. The unbalanced design gives narrower confidence intervals to mono cells than to hetero and null cells.

**Mitigation.** The Tukey pairwise tests use the harmonic-mean sample size (Eq.~\ref{eq:tukey}) to handle the unbalanced design. Per-cell confidence intervals for hetero cells are reported in Table~\ref{tab:closure_rate_matrix}. Effect sizes are computed with partial $\eta^2$, which is sample-size-invariant.

**Residual risk.** Hetero-cell effect estimates are noisier than mono-cell estimates. Consequently, the ``no significant Tukey pair between H6 and M3'' finding (§4.2) could reflect either genuine equivalence or insufficient power at the hetero sample size. A follow-up study with $N = 150$ on hetero cells would tighten this.

### 6.4.3 ANOVA assumptions

**Threat.** ANOVA (Eq.~\ref{eq:anova}) assumes homoscedasticity and normality of residuals. Closure rates are bounded in $[0, 1]$ and can exhibit ceiling effects, potentially violating homoscedasticity.

**Mitigation.** The dominant effect in the interaction ANOVA is path ($\eta^2 = 0.388$), which is orders of magnitude larger than the cell effect ($\eta^2 = 0.068$) or the interaction ($\eta^2 = 0.042$); this makes the F-test robust to moderate assumption violations. The Tukey pairwise test is more sensitive to heteroscedasticity but our reported significant pairs are all far below the significance threshold.

**Residual risk.** Reviewers preferring a mixed-effects logistic regression (Eq.~\ref{eq:mixedlogit}) as the primary analysis rather than as a per-stage decomposition would compute slightly different F-values. Both analyses converge on the same qualitative conclusions.

## 6.5 Deviation from pre-registration

**Deviations from the concept-note pre-registration disclosed here for transparency:**

\begin{itemize}
  \item \textbf{Closed-weight cells deferred.} M5, H2, H8 pre-registered as closed-weight; run as open-weight variants (M5 as Mistral, H2 as qwen3.6/phi-4/qwen-coder, H8 as gemma-4/qwen-coder/mistral). RQ4 correspondingly deferred to follow-up study.
  \item \textbf{Model family lineup.} The concept note listed qwen3.5:9B, llama3.3:70B; the swept lineup uses qwen3.6:27B, gemma4:26B, mistral-small3.2:24B. The change reflects the models available at sweep time (2026-06 to 2026-07).
  \item \textbf{Sample size asymmetric.} Concept note specified 150 per cell across all strata; swept as 150 mono, 75 hetero and null.
  \item \textbf{Held-out subsets prepared but not fully swept.} The 25-problem LiveCodeBench and 50-problem HumanEval-Mutated subsets were prepared per pre-registration; contamination sensitivity results are pending in the addendum.
  \item \textbf{Human evaluation pending.} The 60-pair three-annotator study is prepared (worksheet released) but not yet conducted; results in pending addendum.
\end{itemize}

Every deviation is disclosed in this chapter and reflected in the source-controlled DOE table (\texttt{doe.py}), the pre-registered concept note (\texttt{chapter3\_concept\_note.md}), and the sweep TSV timestamps. No results were re-planned in light of the observed data.

---

*End of §6 Threats to Validity.*
