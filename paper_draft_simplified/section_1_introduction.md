# §1 Introduction

Automated software-engineering pipelines built on top of Small Language Models (SLMs) are increasingly deployed as multi-stage systems: a model generates a specification (docstring) from code, a second model generates unit tests from the specification, and a third model reconstructs code from the specification and tests. Practitioners assembling such pipelines face a concrete question with no empirically-grounded answer: *should I use the same SLM for documentation, test generation, and code synthesis, or should I specialise by stage?* Existing evaluations of LLM-based software-engineering tools either report on individual stages in isolation (LLM-for-docstring-generation~[Liu et al., 2024]; LLM-for-test-generation~[Schäfer et al., 2023; Tufano et al., 2022]; LLM-for-code-synthesis~[Chen et al., 2021]) or evaluate self-consistency within a single LLM across multiple stages~[Allamanis et al., 2024; Zhang et al., 2025]. \emph{No published study has measured cross-stage semantic preservation when stages are owned by different LLM families, under a software-engineering-validated defect-detection metric.} The present chapter addresses this gap.

## 1.1 The docstring--test--code closure question

For each Python function $f$ in a benchmark, three artefacts are canonically associated: the code $C$ (the function body), the docstring $D$ (a natural-language specification), and the reference test suite $T$ (a set of pytest cases exercising~$C$). Together these artefacts form a triangle, and each edge of the triangle can be traversed by a generative pipeline. The pipeline is said to \emph{close} the triangle on a traversal when the reconstructed artefact is semantically equivalent to the original.

We define three closure paths:
\begin{itemize}
  \item \textbf{Path 1 (}$C \to D \to T$\textbf{):} an SLM generates a docstring from the code; a second SLM generates tests from that docstring; we measure whether those tests catch defects introduced in the original code.
  \item \textbf{Path 2 (}$D \to T \to C$\textbf{):} an SLM generates tests from the original docstring; a second SLM reconstructs code from the docstring and generated tests; we measure whether the reconstructed code passes the original reference tests.
  \item \textbf{Path 3 (}$C \to T \to D$\textbf{):} an SLM generates tests from the code; a second SLM recovers a docstring from those tests; we measure whether the recovered docstring is semantically equivalent to the original.
\end{itemize}

Each path is measured by an SE-validated automated metric (mutation kill rate on Path~1, reference-test pass rate on Path~2, docstring BERTScore F1 on Path~3) paired with an external judge SLM's semantic-equivalence rating.

The central research question is whether the mapping of specific SLMs to specific stages of the pipeline matters --- and if so, in what direction, at what magnitude, and with what per-stage attribution.

## 1.2 Position in the thesis

This chapter is the third of a three-paper thesis on the empirical evaluation of open-weight Small Language Models for software engineering. The through-line of the thesis is that \emph{heterogeneous multi-SLM pipelines under SE-relevant metrics reveal interaction effects between method choice and underlying-model capability that single-LLM studies cannot surface}.

Chapter~1 established this claim on the first single-stage artefact of the SE pipeline: retrieval-augmented docstring generation. Chapter~1 showed that LLM choice matters materially for documentation quality under identical retrieval and prompting conditions, and that per-model strengths at different documentation styles do not aggregate into a single ``best'' model. Chapter~2 established the claim on the second single-stage artefact: unit-test generation. Chapter~2 introduced mutation kill rate as the SE-validated defect-detection metric, ran a $4 \times 4$ cross-LLM $\times$ cross-method matrix on 300 HumanEval and MBPP functions, and reported the headline finding that \emph{method rankings do not generalise across LLMs} --- LLM capability dominates RAG-method choice in explained variance. Chapter~2 also produced the per-mutation-operator capability map (phi-4 dominates predicate reasoning; qwen-coder dominates comparison operators) that Chapter~3 takes as the DOE oracle for per-stage strengths.

Chapter~3 generalises the interaction-effect claim from the single-artefact setting (Chapters~1 and~2) to the cross-stage setting: LLM choice interacts with \emph{stage assignment} across the docstring--test--code triangle. The three-chapter arc closes with Chapter~3's empirical demonstration that cross-stage specialisation is a first-class experimental variable, and that its effects are attributable to specific model-stage pairings visible only in multi-stage evaluations.

## 1.3 Research questions

This chapter addresses four research questions. A fifth research question originally in the pre-registered concept note (RQ4: open-weight versus closed-weight ceiling comparison) was deferred to a follow-up study due to resource constraints during the sweep phase --- see §6 (Threats to Validity) for a detailed discussion of this deviation.

\textbf{RQ1.} When the docstring--test--code triangle is traversed by heterogeneous SLMs (different SLM families assigned to the three stages), does the closure rate differ significantly from same-family closure, after controlling for per-sample difficulty?

\textbf{RQ2.} Does mutation kill rate --- measured against the original code's reference-suite mutations --- agree with an external judge SLM's semantic-equivalence rating on each closure path? Where does the two-signal validity policy of Eq.~\ref{eq:valid1}--\ref{eq:valid3} produce disagreement, and does the disagreement structure suggest that the automated metric is capturing the intended signal?

\textbf{RQ3.} Across the $|\mathcal{M}| \times |\mathcal{S}| = 6 \times 3 = 18$ (model, stage) pairings, which stage is the closure bottleneck? Does the bottleneck attribution shift across source benchmark (HumanEval versus MBPP) or across mutation-operator family?

\textbf{RQ5.} What is the false-closure rate --- the proportion of pipelines where the automated metric threshold is satisfied but the judge SLM disagrees --- and is this rate bounded below the empirical noise floor established by the word-shuffled null cell N1?

## 1.4 Contributions

This chapter makes five contributions to the empirical literature on multi-SLM software-engineering pipelines:

\begin{enumerate}
  \item \textbf{First empirical study of heterogeneous multi-SLM round-trip closure} across the docstring--test--code triangle, with a pre-registered 20-cell design of experiments spanning mono (6~cells), hetero (11~cells), and null (3~cells) strata. All cells are committed to source prior to any experimental results being observed.

  \item \textbf{Pre-registered hypothesis validation at per-operator resolution.} The H1 composition (phi-4 spec + qwen-coder test/code) was pre-registered based on Chapter~2's per-operator capability map. H1's empirical per-operator kill rate matches the pre-registered prediction: H1 dominates 4 of 5 mutation operators, with the largest gain ($\Delta = +0.133$) on arithmetic operators. To our knowledge this is the first cross-stage pre-registered LLM composition validated by data at operator resolution.

  \item \textbf{The phi-4-in-test-stage pathology.} We identify a specific stage-model pairing (phi-4 as $L_{\textsf{test}}$) as producing systematically defective test suites at approximately $50\%$ of samples, replicated across three independent cells (M2 mono $= 44\%$, H2 $= 45\%$, H10 $= 55\%$). At operator resolution the pathology localises to the exact mutation families phi-4 mono itself is competitive on --- a cross-stage interaction effect invisible in single-stage evaluations.

  \item \textbf{Judge--metric disagreement decomposition.} On Path~3 (docstring--docstring closure), BERTScore F1 is statistically uncorrelated with the judge SLM's semantic-equivalence rating ($r_3 = 0.02$, $p = 0.359$, $n = 1{,}791$). Stratifying by source benchmark reveals that the disagreement flips direction: on MBPP, BERTScore over-credits equivalence in $29.2\%$ of rows; on HumanEval, BERTScore under-credits in $30.0\%$. This structural directional signature supports the specific claim that BERTScore functions as a surface-form similarity signal rather than a semantic-equivalence signal on our data.

  \item \textbf{Open replication package} including the pre-registered DOE, the full 5{,}890-row sweep TSV, the closure validity decision as a tested Python function (\texttt{closure\_decision.decide\_validity} with 43 unit tests covering all five decision reasons), the test-filter validity gate as a tested Python function (\texttt{closure\_metrics.filter\_tests\_with\_reason} with 8 unit tests covering all four filter reasons), all G1--G6 gap-closer scripts, 11 LaTeX tables, and 10 PNG figures. The replication package is released under the MIT licence.
\end{enumerate}

A sixth methodological contribution --- shared with Chapters~1 and~2 --- is the establishment of open-weight SLM evaluation with pre-registered statistical rigour as a viable methodology for empirical software-engineering research. The six pipeline SLMs used here are all under 30~B parameters and all released with open weights; the entire 5{,}890-row sweep fit within a single-researcher Colab Pro+ monthly allowance.

## 1.5 Roadmap

The remainder of this chapter is organised as follows. Section~2 reviews related work on LLM-based test and docstring generation, round-trip and consistency evaluation, LLM-as-judge protocols, and multi-agent architectural specialisation. Section~3 presents the methods, including the notation (§3.1), the formal definition of the three closure paths (§3.2, Eqs.~\ref{eq:path1}--\ref{eq:path3}), the closure validity decision (§3.3, Eq.~\ref{eq:valid1} and Algorithm~\ref{alg:validity}), the test-filter validity gate (§3.4, Eq.~\ref{eq:filter} and Algorithm~\ref{alg:test_filter}), the closure metrics (§3.5), the 20-cell design of experiments (§3.6, Eq.~\ref{eq:doe}), the datasets (§3.7), and the statistical methodology (§3.8). Section~4 reports the empirical results across all four research questions with anchor tables and figures. Section~5 interprets the findings against RQ1--RQ5, characterises the phi-4-in-test-stage pathology and the BERTScore surface-form finding, and derives three design guidelines for practitioners deploying multi-SLM SE pipelines. Section~6 details the threats to validity. Section~7 concludes with future research directions, including a follow-up study to address the deferred RQ4 on closed-weight comparison.

---

*End of §1 Introduction.*
