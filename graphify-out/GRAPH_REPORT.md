# Graph Report - Financial-NRF  (2026-10-30)

## Corpus Check
- 209 files · ~354,572 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 193 file(s) not represented in the graph (top: .csv 147, .jsonl 25, .tex 12)

## Summary
- 887 nodes · 1727 edges · 49 communities (45 shown, 4 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Experimental Chapter Findings
- Study Guide and Metrics
- Evidence Check Analysis
- Workflow Noise Inputs
- NRF Core Framework
- Paper and Release Notes
- Data Loading and Experiment 2
- Arithmetic Verifier
- Erratum Claims
- CI Workflow
- Confidence Audit
- Paper Figures
- Release Docs Workflow
- Scale Report
- Experiment 4 Corruption
- Paper Build Workflow
- Experiment 1 Baseline
- Results Explorer App
- Judge Workflow Inputs
- FinancialRAG Generation
- Chunking and Indexing
- Corruption Chart
- HF Space Deploy
- Results Files
- Question Issue Template
- Environment Variables
- Test Report Template
- Document Index
- Generation Cache
- Workflow Steps
- Evidence Support Scoring
- Realistic Error Findings
- Experiment 5 Baselines
- LLM Judge Baseline
- Generate Shards Job
- Judge Workflow
- Replay Job Steps
- Judge Shard Job
- TAT-QA Loader
- Cleanup and Replay Jobs
- Replay Environment
- Collect Job
- Plan Shards Job
- Plan Shards Job B
- Issue Config
- Demo and Landing Page
- GGUF Input
- arXiv Build Script

## God Nodes (most connected - your core abstractions)
1. `NumericalReliabilityFramework` - 44 edges
2. `FinancialRAG` - 30 edges
3. `f()` - 24 edges
4. `main()` - 20 edges
5. `main()` - 19 edges
6. `inputs` - 17 edges
7. `load_dataset()` - 16 edges
8. `main()` - 15 edges
9. `main()` - 15 edges
10. `exp5_audit()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `sweep()` --semantically_similar_to--> `Weight sensitivity sweep (1,771 weightings)`  [INFERRED] [semantically similar]
  experiments/scale_analysis.py → README.md
- `Corruption chart (accuracy collapses, support flat)` --semantically_similar_to--> `LinkedIn chart: wrong numbers collapse accuracy, scores needing no right answer move far less`  [INFERRED] [semantically similar]
  README.md → paper/linkedin/chart.png
- `components()` --implements--> `Composite ONRS formula and DEFAULT_W weights`  [INFERRED]
  experiments/scale_analysis.py → docs/STUDY_GUIDE.md
- `Paired item-level effects (last level minus first)` --rationale_for--> `analyse()`  [EXTRACTED]
  docs/STUDY_GUIDE.md → experiments/scale_analysis.py
- `Composite ONRS formula and DEFAULT_W weights` --semantically_similar_to--> `Composite reliability score (ONRS)`  [INFERRED] [semantically similar]
  docs/STUDY_GUIDE.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **NRF experiment suite (Exp 1 to 5)** — nrf_experimental_chapter_exp1_baseline, nrf_experimental_chapter_exp2_hallucination_sensitivity, nrf_experimental_chapter_exp3_calibration, nrf_experimental_chapter_exp4_noise_levels, nrf_experimental_chapter_exp5_oracle_evidence [EXTRACTED 1.00]
- **ONRS composite components (accuracy, NESS, hallucination, calibration)** — nrf_experimental_chapter_onrs, nrf_experimental_chapter_ness_strict, nrf_experimental_chapter_hallucination_detection, nrf_experimental_chapter_ece_self_reported_confidence [EXTRACTED 1.00]
- **Defects found in the methodology audit** — nrf_experimental_chapter_fabricated_faithfulness_constant, nrf_experimental_chapter_percentage_scaling_defects, nrf_experimental_chapter_chunking_shredding_defect, nrf_experimental_chapter_inconsistent_gold_defect, nrf_experimental_chapter_confidence_counted_unsupported [EXTRACTED 1.00]
- **Four checks composing the NRF composite score** — readme_execution_accuracy, readme_ness_evidence_support, readme_arithmetic_verification, readme_calibration_ece, readme_onrs_composite [EXTRACTED 1.00]
- **Gold-free measures blind to corrupted evidence** — readme_ness_evidence_support, readme_entailment_nli, readme_hallucination_flag, readme_ragas_faithfulness_judge, readme_corrupted_evidence_experiment [EXTRACTED 1.00]
- **scale_analysis statistical pipeline (load, analyse, sweep, held-out)** — experiments_scale_analysis_load_config, experiments_scale_analysis_analyse, experiments_scale_analysis_sweep, experiments_scale_analysis_held_out, experiments_scale_analysis_main [EXTRACTED 1.00]
- **Reference-free reliability scores that fail to detect corrupted evidence** — results_paper_analysis_numeric_overlap_support, results_paper_analysis_entailment_support, results_paper_analysis_lexical_groundedness, results_paper_analysis_llm_judge_faithfulness, results_paper_analysis_no_hallucination_flag, results_paper_analysis_self_reported_confidence [EXTRACTED 1.00]
- **CPU re-run pipeline: sharded generation, cache, replay scoring** — results_cpu_fp32_readme_cpu_rerun_workflow, results_cpu_fp32_readme_replay_engine, results_cpu_fp32_readme_corrected_support_scorer, results_cpu_fp32_readme_weight_sensitivity [EXTRACTED 1.00]
- **Corrupted-evidence finding across analysis, scale summary, paper and post** — results_paper_analysis_finding_corrupted_evidence_undetected, results_scale_summary_corrupted_evidence_experiment, paper_arxiv_paper, paper_linkedin_post_headline_claim [INFERRED 0.85]
- **Reference-free scores stay flat while accuracy collapses under evidence corruption** — docs_corruption_chart_evidence_support_score, docs_corruption_chart_accuracy_metric, paper_linkedin_chart_model_confidence, paper_linkedin_chart_numbers_appear_in_text, paper_linkedin_chart_text_supports_claims, paper_linkedin_chart_accuracy_metric [INFERRED 0.85]

## Communities (49 total, 4 thin omitted)

### Community 0 - "Experimental Chapter Findings"
Cohesion: 0.06
Nodes (62): NRF Experimental Evaluation Chapter, AMT/2005 retrieval recall miss case, Answer instability under prompt perturbation (~28%), Anthropic API backend (--engine anthropic), Character-shredding chunking defect in Exp 2 and 4, Chunking strategy (one chunk per sentence / table row), Confidence value counted as unsupported evidence defect, Appendix A consolidated results table (+54 more)

### Community 1 - "Study Guide and Metrics"
Cohesion: 0.07
Nodes (30): AUROC, Item-level bootstrap, Make-the-next-change exercises, Natural language inference (premise/hypothesis entailment), cross-encoder/nli-deberta-v3-small, evidence_windows(), extract_claims(), load_predictor() (+22 more)

### Community 2 - "Evidence Check Analysis"
Cohesion: 0.09
Nodes (31): analyse_run(), main(), run_files(), auroc(), boot_auroc(), corruption_detection(), error_detection(), fmt_auc() (+23 more)

### Community 3 - "Workflow Noise Inputs"
Cohesion: 0.04
Nodes (49): default, description, default, description, default, description, default, description (+41 more)

### Community 4 - "NRF Core Framework"
Cohesion: 0.10
Nodes (6): evaluate_example(), run_experiment(), main(), NumericalReliabilityFramework, ParsedNumber, _run_tests()

### Community 5 - "Paper and Release Notes"
Cohesion: 0.09
Nodes (36): arXiv submission notes, arxiv.zip build package and workflow, Paper: Reference-Free Reliability Scores Cannot See Corrupted Evidence, Pre-submission checklist (endorsement, bibliography, AI disclosure), LinkedIn post draft, Post claim: accuracy fell 53% to 8% while support score barely moved, CPU fp32 re-run README, Corrected evidence-support scorer (+28 more)

### Community 6 - "Data Loading and Experiment 2"
Cohesion: 0.14
Nodes (13): load_dataset(), load_finqa(), apply_adversarial_noise(), backup_existing_results(), evaluate_condition(), main(), summarise(), backup_existing_results() (+5 more)

### Community 7 - "Arithmetic Verifier"
Cohesion: 0.11
Nodes (9): _clean_expr(), find_last_expression(), PostHocArithmeticVerifier, _run_tests(), safe_eval_arithmetic(), _safe_eval_node(), UnsafeExpressionError, VerificationResult (+1 more)

### Community 8 - "Erratum Claims"
Cohesion: 0.10
Nodes (23): Accuracy 0.340 not stable across numerics, Claims in submitted text and their status, Claim withdrawn: hallucination rose 0.10 to 0.20, Claim not established: no weighting passes both tests, What was not done (4-bit GPU rerun, larger models, real retrieval corruption), Claim not reproducible: oracle scores lower on composite, Re-analysis runs in repository, Submitted dissertation documents (NRF_Report_IEEE_v3, NRF_Supporting_Evidence_v3) (+15 more)

### Community 9 - "CI Workflow"
Cohesion: 0.11
Nodes (24): name, runs-on, steps, name, runs-on, steps, jobs, framework-tests (+16 more)

### Community 10 - "Confidence Audit"
Cohesion: 0.17
Nodes (15): candidates(), corrected_options(), exact_audit(), block(), exp4_audit(), exp5_audit(), grid_fraction(), onrs() (+7 more)

### Community 11 - "Paper Figures"
Cohesion: 0.17
Nodes (14): auc(), f2(), fig_auroc(), fig_corruption(), groups(), main(), numbers(), table_corruption() (+6 more)

### Community 12 - "Release Docs Workflow"
Cohesion: 0.10
Nodes (19): GH_TOKEN, TAG, tag, jobs, release, name, on, workflow_dispatch (+11 more)

### Community 13 - "Scale Report"
Cohesion: 0.18
Nodes (10): ci(), f(), main(), pc(), render(), ms(), component_scores(), load_exp4() (+2 more)

### Community 14 - "Experiment 4 Corruption"
Cohesion: 0.15
Nodes (13): Paired item-level effects (last level minus first), Experimental-design flaw in Experiment 4, Claims not established: hallucination tracks noise; composite does not track noise, apply_noise(), corrupt_evidence(), h(), repl(), edit_evidence() (+5 more)

### Community 15 - "Paper Build Workflow"
Cohesion: 0.11
Nodes (18): runs-on, steps, jobs, build, name, on, push, workflow_dispatch (+10 more)

### Community 16 - "Experiment 1 Baseline"
Cohesion: 0.16
Nodes (6): backup_existing_results(), main(), print_summary(), write_results_csv(), resolve_gold_answer(), _run_tests()

### Community 17 - "Results Explorer App"
Cohesion: 0.17
Nodes (10): comparison_table(), _get_rag(), headline_table(), inspect_row(), _load_csv(), _load_json(), noise_curve(), per_example() (+2 more)

### Community 18 - "Judge Workflow Inputs"
Cohesion: 0.12
Nodes (17): default, description, cleanup, limit, shards, split, tag, default (+9 more)

### Community 20 - "Chunking and Indexing"
Cohesion: 0.17
Nodes (3): chunk_document(), _document_key(), _format_table_row()

### Community 21 - "Corruption Chart"
Cohesion: 0.18
Nodes (16): Corruption chart (accuracy collapses, support flat), Accuracy (0.22 to 0.03 under full corruption), Share of numbers in retrieved evidence replaced with wrong ones (0-100%), Data source: Qwen2.5-3B on FinQA, 100 dev questions (results/scale/qwen3bcorrupt_dev/), Evidence-support score (needs no right answer, 0.74 to 0.75), Insight: reference-free support score is blind to corrupted evidence while accuracy collapses, Same support score checked against clean original evidence (0.74 to 0.38), LinkedIn chart: wrong numbers collapse accuracy, scores needing no right answer move far less (+8 more)

### Community 22 - "HF Space Deploy"
Cohesion: 0.20
Nodes (15): deploy/README.md (Hugging Face Space card), Hugging Face Space (Gradio sdk 4.44.0), Live pipeline mode (NRF_ENABLE_LIVE=1), Expected calibration error (ECE), Composite ONRS formula and DEFAULT_W weights, rag_pipeline/ package, all-MiniLM-L6-v2 embeddings, Calibration (ECE) (+7 more)

### Community 23 - "Results Files"
Cohesion: 0.14
Nodes (13): data_loader/ package, Results explorer mode (default, no GPU), FinQA dataset, confidence_line_audit_dev.json, experiment1_results_dev_summary.json, experiment3_results_dev_summary.json, experiment4_results_dev.csv, experiment5_results_dev_summary.json (+5 more)

### Community 24 - "Question Issue Template"
Cohesion: 0.14
Nodes (13): description, label, value, body, attributes, id, type, validations (+5 more)

### Community 25 - "Environment Variables"
Cohesion: 0.20
Nodes (14): HF_HUB_DISABLE_PROGRESS_BARS, NRF_CPU_DTYPE, NRF_DATASET, NRF_GEN_CACHE, NRF_GEN_CACHE_READ, NRF_GEN_SEED, NRF_GGUF, NRF_LEVELS (+6 more)

### Community 26 - "Test Report Template"
Cohesion: 0.15
Nodes (12): description, label, body, attributes, id, type, validations, description (+4 more)

### Community 27 - "Document Index"
Cohesion: 0.26
Nodes (3): Chunk, DocumentIndex, _NumpyFlatIndex

### Community 29 - "Workflow Steps"
Cohesion: 0.26
Nodes (12): steps, steps, steps, continue-on-error, if, name, run, uses (+4 more)

### Community 30 - "Evidence Support Scoring"
Cohesion: 0.25
Nodes (6): exclude_confidence_from_support option, Scoring defect: confidence number counted as unsupported, EvidenceSupportResult, SampleResult, Confidence-line scoring defect, Recorded runs predate scoring fix

### Community 31 - "Realistic Error Findings"
Cohesion: 0.24
Nodes (10): Structural argument stands: evidence support cannot detect corrupted evidence, Support against clean evidence, Corrupted-evidence experiment, Evidence-side consistency checks, Llama-3.1-8B, llama.cpp CPU runner, Misplaced-number errors (swapped periods/rows), Qwen2.5-Instruct models (1.5B-14B) (+2 more)

### Community 32 - "Experiment 5 Baselines"
Cohesion: 0.29
Nodes (6): backup_existing_results(), build_oracle_evidence(), evaluate(), _fmt(), main(), summarise()

### Community 33 - "LLM Judge Baseline"
Cohesion: 0.35
Nodes (6): build_messages(), faithfulness(), load_rows(), main(), parse_verdicts(), self_test()

### Community 34 - "Generate Shards Job"
Cohesion: 0.18
Nodes (11): if, name, needs, runs-on, strategy, timeout-minutes, generate, shard (+3 more)

### Community 35 - "Judge Workflow"
Cohesion: 0.18
Nodes (10): env, GGUF, LIMIT, SHARDS, SPLIT, TAG, name, on (+2 more)

### Community 36 - "Replay Job Steps"
Cohesion: 0.25
Nodes (11): steps, steps, RUN_ID, SHARD, steps, env, name, run (+3 more)

### Community 37 - "Judge Shard Job"
Cohesion: 0.18
Nodes (11): judge, if, name, needs, runs-on, strategy, timeout-minutes, shard (+3 more)

### Community 38 - "TAT-QA Loader"
Cohesion: 0.36
Nodes (6): _clean(), _convert(), _format_row(), load_tatqa(), _split_header(), _table_rows()

### Community 39 - "Cleanup and Replay Jobs"
Cohesion: 0.22
Nodes (9): if, runs-on, jobs, cleanup, replay, if, needs, runs-on (+1 more)

### Community 40 - "Replay Environment"
Cohesion: 0.22
Nodes (9): INJ, N_MAIN, RUN_ID, SEED_FROM, SHARD, SHARDS, SPLIT, TAG (+1 more)

### Community 41 - "Collect Job"
Cohesion: 0.25
Nodes (8): if, runs-on, if, needs, runs-on, jobs, cleanup, collect

### Community 42 - "Plan Shards Job"
Cohesion: 0.29
Nodes (7): plan, shards, if, outputs, runs-on, steps, id

### Community 43 - "Plan Shards Job B"
Cohesion: 0.29
Nodes (7): plan, shards, if, outputs, runs-on, steps, id

### Community 44 - "Issue Config"
Cohesion: 0.33
Nodes (5): blank_issues_enabled, contact_links, about, name, url

### Community 45 - "Demo and Landing Page"
Cohesion: 0.40
Nodes (5): index.html static demo, Research paper draft (paper/main.pdf), README.md (Financial-NRF), Financial-NRF research project, Browser live demo (index.html, GitHub Pages)

### Community 46 - "GGUF Input"
Cohesion: 0.67
Nodes (3): default, description, gguf

## Ambiguous Edges - Review These
- `Corrected evidence-support scorer` → `Post claim: accuracy fell 53% to 8% while support score barely moved`  [AMBIGUOUS]
  paper/linkedin/post.txt · relation: references

## Knowledge Gaps
- **31 isolated node(s):** `ERRATUM.md`, `SmolLM2-1.7B-Instruct`, `all-MiniLM-L6-v2 embedding model`, `FAISS IndexFlatIP exact retrieval`, `Tolerance and percentage-mismatch policy` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 341 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Re-analysis runs in repository` connect `Erratum Claims` to `Study Guide and Metrics`, `Paper and Release Notes`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `NumericalReliabilityFramework` (e.g. with `evaluate_example()` and `run_experiment()`) actually correct?**
  _`NumericalReliabilityFramework` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ERRATUM.md`, `SmolLM2-1.7B-Instruct`, `all-MiniLM-L6-v2 embedding model` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Experimental Chapter Findings` be split into smaller, more focused modules?**
  _Cohesion score 0.05874125874125874 - nodes in this community are weakly interconnected._
- **Why does `jobs` connect `Cleanup and Replay Jobs` to `Erratum Claims`, `Generate Shards Job`, `Plan Shards Job`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `FinancialRAG` (e.g. with `evaluate_example()` and `run_experiment()`) actually correct?**
  _`FinancialRAG` has 4 INFERRED edges - model-reasoned connections that need verification._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Corrected evidence-support scorer` and `Post claim: accuracy fell 53% to 8% while support score barely moved`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._