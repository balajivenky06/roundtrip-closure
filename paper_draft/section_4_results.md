# §4 Results

We report closure rates across the 20-cell DOE on the 150-function core dataset, comprising 5{,}890 measurement rows (150 mono samples $\times$ 3 paths $\times$ 6 mono cells $+$ 75 hetero samples $\times$ 3 paths $\times$ 11 hetero cells $+$ 75 null samples $\times$ 3 paths $\times$ 3 null cells). Results are organised by research question. Every closure rate reported here uses the strict two-signal validity policy of Eq.~\ref{eq:valid1}--\ref{eq:valid3} with default thresholds $\tau_1 = \tau_2 = \tau_3 = 0$ and $\rho = 3$; §5.6 confirms every headline finding survives alternative threshold choices.

## 4.1 Overall closure rates per cell and path

Table~\ref{tab:closure_rate_matrix} and Fig.~\ref{fig:closure_heatmap} present the mean closure rate for every cell $\times$ path combination. Three high-level patterns emerge.

First, Path~1 (mutation kill rate) closure rates cluster in the $[0.7, 0.9]$ range for all non-ablated cells, with a mean of $0.831$. The strongest mono baseline is M3 (qwen3.6:27B, all stages) at $0.900$; the strongest hetero cell is H6 (phi-4/qwen3.6/phi-4) at $0.882$. The single mono failure is M1 (llama3.2:3B mono) at $0.577$.

Second, Path~2 (reference test pass rate) closure rates are systematically lower than Path~1, averaging $0.630$ across non-ablated cells. This reflects the harder nature of the code-synthesis task: passing the reference tests requires reconstructing the function's behaviour exactly rather than merely covering it. Notably, the spec-stage-ablated null cell N2 (Path~2 closure rate $= 0.667$) ties or exceeds every mono baseline (M1 $= 0.367$, M2 $= 0.547$, others $\in [0.533, 0.647]$). We return to the implications of this in §5.3.

Third, Path~3 (BERTScore) closure rates are uniformly low, averaging $0.211$ across non-ablated cells, and vary in a compressed range $[0.17, 0.28]$. This is an artefact of BERTScore's rescaled value distribution on our data --- the natural distribution of BERTScore F1 values between original and reconstructed docstrings clusters in $[-0.1, 0.5]$. The compressed range does not indicate poor pipeline performance; §5.2 shows that judge equivalence ratings on the same rows tell an entirely different story.

## 4.2 Mono versus heterogeneous cells

Fig.~\ref{fig:mono_vs_hetero} presents the per-strand distribution of closure rates on Path~1. The distributions overlap substantially: the median hetero cell ($0.862$) is competitive with the median mono cell ($0.807$), but the ceiling of each strand differs. The best mono cell (M3, kill rate $= 0.900$) exceeds the median hetero cell but not the best hetero cell (H6, $0.882$).

To test whether cell-mean differences are statistically significant, we fit the ANOVA model of Eq.~\ref{eq:anova}. Cell is a significant predictor of closure rate ($F(13, 3918) = 17.45$, $p < 0.001$, partial $\eta^2 = 0.068$) after adjusting for per-sample difficulty. Tukey's HSD (Eq.~\ref{eq:tukey}) with Bonferroni correction over 190 pairs identifies 35 pairs significant at $p_{\textsf{adj}} < 0.05$ and 33 significant at $p_{\textsf{adj}} < 0.01$ (Table~\ref{tab:tukey_significant}). The strongest single contrasts are:

\begin{itemize}
  \item H1 vs.\ M1: $\Delta = -0.182$ ($p_{\textsf{adj}} < 0.001$) --- phi-4 in the spec stage plus qwen-coder in test/code stages substantially outperforms llama3.2 mono.
  \item H4 vs.\ N1: $\Delta = -0.262$ ($p_{\textsf{adj}} < 0.001$) --- every hetero cell beats the word-shuffled control by $> 0.25$, confirming that hetero closure signal is not a metric artefact.
  \item M1 vs.\ N2: $\Delta = +0.282$ ($p_{\textsf{adj}} < 0.001$) --- the spec-stage-ablated null cell N2 beats M1 mono; skipping the docstring stage and reusing the ground-truth docstring works better than generating a weak docstring in the first place.
\end{itemize}

The strongest non-directional finding is that the M3 mono ceiling and the best hetero cells cluster within $0.05$ of each other on Path~1. §5.1 develops the interpretation that hetero pipelines *match* rather than *dominate* the strongest mono baseline on aggregate closure rate, and the more interesting stage-attribution structure emerges in the per-operator decomposition (§4.4).

## 4.3 Per-stage bottleneck attribution

Each hetero cell composes three models $(m_{\textsf{spec}}, m_{\textsf{test}}, m_{\textsf{code}})$. Table~\ref{tab:per_stage_bottleneck} attributes each hetero cell's closure rate to the mono cell whose model appears at the same slot: for hetero cell $a$ and stage $s$, the *stage baseline* $\hat{Y}_{a,s}$ is the closure rate of the mono cell using $a(s)$ at all stages. The *bottleneck stage* is the stage whose baseline is the lowest, and $\Delta_a = \bar{Y}_a - \min_s \hat{Y}_{a,s}$ is the improvement from specialising the other two stages.

Across the 11 hetero cells, spec-stage attribution accounts for 15 of the 33 (cell $\times$ path) triples; test-stage 10; code-stage 8. The most striking single-stage attributions are:

\begin{itemize}
  \item \textbf{H5 spec attribution, $\Delta = 0.267$} --- llama3.2 at $L_{\textsf{spec}}$, qwen-coder at $L_{\textsf{test}}$ and $L_{\textsf{code}}$. Substituting stronger models at test and code stages recovers $27$ percentage points of closure rate that llama3.2 mono cannot achieve alone.
  \item \textbf{H4 code attribution, $\Delta = 0.186$} --- qwen-coder at $L_{\textsf{spec}}$ and $L_{\textsf{test}}$, llama3.2 at $L_{\textsf{code}}$. Even the ``cheap drafter at the code stage'' composition recovers $\Delta = 0.186$ over the llama3.2 mono baseline, suggesting that spec and test quality can compensate substantially for weak synthesis.
\end{itemize}

Fig.~\ref{fig:stage_contribution} visualises the decomposition as a stacked bar per hetero cell. Cells H4 and H11 exceed the strongest mono ceiling (M4 = $0.773$); every other hetero cell lies between its worst-stage mono baseline and that ceiling; only H10 falls *below* its worst-stage mono baseline, indicating a heterogeneity *loss* rather than gain. §5.3 develops this into a design guideline.

## 4.4 Per-mutation-operator kill rate

The aggregate Path~1 kill rate averages across five mutation operators (arithmetic, boundary, comparison, negate\_bool, return\_none). Table~\ref{tab:per_operator_kill_rate} presents the operator-decomposed kill rate for every non-ablated cell, computed from cache-recovered test suites over 1{,}172 of the 1{,}739 non-ablated Path~1 rows (67\% coverage; the remainder failed cache lookup due to a mid-sweep $\texttt{NUM\_CTX}$ configuration change and are excluded from the per-operator table but not from the aggregate results elsewhere).

Two structural findings emerge. First, mono baselines exhibit different operator strengths: M3 (qwen3.6) dominates 4 of 5 operators; M6 (qwen-coder) wins negate\_bool. Second, hetero cells inherit the operator-specific ceilings:

\begin{itemize}
  \item Arithmetic: mono ceiling M3 $= 0.806$; hetero winner \textbf{H1} $= 0.939$, $\Delta = +0.133$.
  \item Boundary: mono ceiling M3 $= 0.748$; hetero winner \textbf{H1} $= 0.844$, $\Delta = +0.096$.
  \item Comparison: mono ceiling M3 $= 0.912$; hetero winners \textbf{H1} and \textbf{H4} $= 1.000$, $\Delta = +0.088$.
  \item Negate\_bool: mono ceiling M6 $= 0.944$; hetero winners \textbf{H1}, \textbf{H4}, \textbf{H8} $= 1.000$, $\Delta = +0.056$.
  \item Return\_none: mono ceiling M3 $= 0.966$; hetero winner \textbf{H6} $= 0.976$, $\Delta = +0.010$.
\end{itemize}

Table~\ref{tab:per_operator_headline} summarises these findings. \textbf{H1 (phi-4 at $L_{\textsf{spec}}$, qwen-coder at $L_{\textsf{test}}$ and $L_{\textsf{code}}$) is the single strongest cell on 4 of 5 mutation operators}, beating the best mono baseline by $\Delta \geq 0.056$ on all four and by $\Delta = 0.133$ on arithmetic. This finding directly validates H1's pre-registered hypothesis in the DOE table: H1 was assigned this composition based on Chapter~2's finding that phi-4 dominates predicate reasoning (Ch.~2 §4.4) and qwen-coder dominates comparison operators (Ch.~2 Table~13). The per-operator data confirms the hypothesis at the resolution of individual defect families.

Two outliers merit specific attention. H10 (mistral spec + phi-4 test + qwen-coder code) achieves arithmetic $= 0.351$ and boundary $= 0.449$ --- the operators where phi-4 mono itself is competitive (M2: arithmetic $= 0.580$, boundary $= 0.653$). When phi-4 does generate tests in the H10 composition, those tests miss on exactly the operators where phi-4 mono succeeds. §5.3 develops this as the phi-4-in-test-stage pathology at operator resolution.

## 4.5 Per-benchmark decomposition

Table~\ref{tab:per_benchmark} decomposes the cell effect by source benchmark. On HumanEval ($n = 1{,}652$), $F(19, 1632) = 5.06$, $p < 0.001$; on MBPP ($n = 4{,}238$), $F(19, 4218) = 8.91$, $p < 0.001$. Cell assignment is a significant predictor on both benchmarks, with MBPP showing a stronger effect --- consistent with MBPP's shorter and more uniform problem structure making per-cell differences more measurable.

The most striking benchmark-specific effect surfaces not in the aggregate closure rate but in the Path~3 judge--metric decomposition (§5.2), which flips direction between HumanEval and MBPP.

## 4.6 Judge--metric correlation (RQ2)

Fig.~\ref{fig:judge_corr} plots the automated closure metric against the judge rating for each path, with per-path Pearson correlations reported in Table~\ref{tab:judge_correlation}. The three paths yield markedly different agreement:

\begin{itemize}
  \item Path~1 (mutation kill rate vs.\ judge): $r_1 = 0.192$, $p = 2.12 \times 10^{-13}$, $n = 1{,}441$. Statistically significant but weak.
  \item Path~2 (pass rate vs.\ judge): $r_2 = 0.523$, $p = 1.19 \times 10^{-129}$, $n = 1{,}839$. Strong and highly significant.
  \item Path~3 (BERTScore vs.\ judge): $r_3 = 0.022$, $p = 0.359$, $n = 1{,}791$. \textbf{Not significant}: the null hypothesis that BERTScore and the judge rating are independent cannot be rejected.
\end{itemize}

Path~3's non-significant correlation at $n = 1{,}791$ is the paper's strongest single validity finding. Standard practice in NLP evaluation (Zhang et al., 2020) treats BERTScore F1 as a semantic-equivalence proxy; our result shows it is uncorrelated with an LLM-based semantic-equivalence judgement at scale. §5.2 stratifies this finding by benchmark to characterise the structural nature of the disagreement.

## 4.7 Path $\times$ cell interaction

To test whether the three closure paths carry non-redundant information, we fit the interaction ANOVA of Eq.~\ref{eq:anova_interact}. Table~\ref{tab:anova_interaction} presents the results for $n = 5{,}068$ rows after excluding structural NaN entries and cells present only on a subset of paths.

The cell $\times$ path interaction is significant ($F(34, 4865) = 6.21$, $p < 0.001$, partial $\eta^2 = 0.042$). This validates that the three-path framework is not redundant: cells respond \emph{differently} to different closure paths rather than paths adding a constant offset. Concretely, the ranking of cells is not preserved across paths --- Table~\ref{tab:closure_rate_matrix} shows M3 (best Path~1) is not the best Path~3 performer; N2 (spec-ablated) is competitive with any mono cell on Path~2 but structural NA on Paths~1 and~3.

The largest main effect is path itself (partial $\eta^2 = 0.388$), reflecting that Path~1, Path~2, and Path~3 metrics inhabit different value scales; the cell main effect ($\eta^2 = 0.068$) captures aggregate composition quality; and sample difficulty ($\eta^2 = 0.227$) is a substantial random-intercept-equivalent term. All four effects are significant at $p < 0.001$.

## 4.8 Cache efficiency and computational cost

Table~\ref{tab:cache_efficiency} reports the closure-call cache hit rate per cell. Because the LLM cache is keyed on (model, role, prompt, generation-parameters), cells that reuse the same model at overlapping stages achieve high hit rates through cross-cell reuse. Mono cells run first and see near-zero hit rates ($\leq 0.6\%$ for M1, M2, M4, M5, M6); M3 achieves 21.5\% because it was re-run after a mid-sweep configuration change. Hetero cells run subsequently and inherit substantial hit rates: H4 achieves 89.2\% because its qwen-coder stages reuse M6 cache entries directly. N2 and N3 achieve 100\% because their surviving stages are identical to earlier cells.

The full 20-cell sweep consumed 16{,}302 SLM calls, of which 3{,}706 were served from cache ($22.7\%$ hit rate). The total wall-clock time on Google Colab hardware was approximately 22 GPU-days accumulated over 23~calendar days, spanning both A100 and T4 hardware.

---

*End of §4 Results.* Section~5 interprets these findings against RQ1--RQ5 and discusses design implications and threats to validity.
