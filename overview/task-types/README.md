# Task Types (19)

| Type ID | Name | Steps | Description |
|---------|------|-------|-------------|
| `answer-question` | Answer Question | 5 optional | Answer one or more research questions and register the results as answer assets. |
| `baseline-evaluation` | Baseline Evaluation | all 8 | Run a baseline or benchmark evaluation against standard datasets. |
| `brainstorming` | Brainstorming | 0 optional | Record decisions and outcomes from a brainstorming session. |
| `build-model` | Build/Train Model | all 8 | Build, train, or fine-tune a machine learning model. |
| `code-reproduction` | Code Reproduction | 7 optional | Reproduce results from an existing paper or codebase. |
| `comparative-analysis` | Comparative Analysis | 5 optional | Compare multiple approaches, models, versions, or configurations. |
| `correction` | Correction | 0 optional | Correct aggregated artifacts from earlier completed tasks. |
| `data-analysis` | Data Analysis | 4 optional | Analyze data, compute statistics, and generate visualizations. |
| `deduplication` | Deduplication | 2 optional | Scan for and resolve duplicate assets across tasks. |
| `download-dataset` | Download Dataset | 1 optional | Download one or more datasets and register them as dataset assets. |
| `download-paper` | Download Paper | 0 optional | Download one or more research papers and register them as paper assets. |
| `experiment-run` | Experiment Run | all 8 | Run a designed experiment combining training, inference, and evaluation. |
| `feature-engineering` | Feature Engineering | 5 optional | Extract, compute, or generate features from data for model training. |
| `infrastructure-setup` | Infrastructure Setup | 3 optional | Set up environments, tools, dependencies, or infrastructure. |
| `internet-research` | Internet Research | 1 optional | Research a specific question using internet sources, documentation, and online resources. |
| `literature-survey` | Literature Survey | 4 optional | Survey and summarize academic literature on a specific topic. |
| `tts-benchmark-run` | TTS Benchmark Run | all 8 | Run a TTS inference benchmark measuring TTFB, speaker similarity, and RTF. |
| `tts-finetuning-eval` | TTS Fine-tuning Evaluation | all 8 | Evaluate a Kokoro fine-tuned checkpoint on val_96 and filler corpus. |
| `write-library` | Write Library | 3 optional | Create a reusable Python library for use by downstream tasks. |

<details>
<summary><strong>Answer Question</strong> (<code>answer-question</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `answer-question` |
| **Description** | Answer one or more research questions and register the results as answer assets. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `creative-thinking` |

Covers tasks whose primary deliverable is a well-supported answer to a specific free-form
question. The question may be answered through existing papers, internet research, prior
project findings, or new code experiments, with the final output stored as one or more answer
assets. Examples include clarifying a framework design choice, resolving a methodological
uncertainty, or determining whether an implementation claim is supported by evidence.

</details>

<details>
<summary><strong>Baseline Evaluation</strong> (<code>baseline-evaluation</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `baseline-evaluation` |
| **Description** | Run a baseline or benchmark evaluation against standard datasets. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `creative-thinking`, `compare-literature` |

Covers tasks that implement and evaluate a known baseline method on standard benchmarks. The
goal is establishing a reference point for comparison, not inventing new methods. Baselines
may be simple heuristics, published systems, or standard model architectures. Results must be
compared against published numbers.

</details>

<details>
<summary><strong>Brainstorming</strong> (<code>brainstorming</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `brainstorming` |
| **Description** | Record decisions and outcomes from a brainstorming session. |
| **Optional steps** | — none — |

Covers tasks that capture the results of brainstorming sessions between human researchers and
AI agents. The primary output is a set of suggestions for future tasks, decisions about
project direction, and documented rationale. No research or planning steps are needed since
the brainstorming session itself drives the content.

</details>

<details>
<summary><strong>Build/Train Model</strong> (<code>build-model</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `build-model` |
| **Description** | Build, train, or fine-tune a machine learning model. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `creative-thinking`, `compare-literature` |

Covers tasks that involve designing, implementing, training, and evaluating ML models. This
includes fine-tuning pretrained models, training from scratch, and hyperparameter
optimization. Typically requires GPU compute and produces trained model checkpoints,
evaluation metrics, and analysis. Examples include fine-tuning a pretrained model, training a
bi-encoder, or implementing a new architecture from scratch.

</details>

<details>
<summary><strong>Code Reproduction</strong> (<code>code-reproduction</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `code-reproduction` |
| **Description** | Reproduce results from an existing paper or codebase. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `compare-literature` |

Covers tasks that reproduce published results by running the original code or reimplementing
the method. The goal is verifying that results can be replicated and understanding the method
deeply. This involves finding code repositories, setting up environments, running training or
inference, and comparing outputs to published numbers. Compare-literature is critical for
validating reproduction fidelity.

</details>

<details>
<summary><strong>Comparative Analysis</strong> (<code>comparative-analysis</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `comparative-analysis` |
| **Description** | Compare multiple approaches, models, versions, or configurations. |
| **Optional steps** | `research-papers`, `research-code`, `planning`, `creative-thinking`, `compare-literature` |

Covers tasks that systematically compare two or more alternatives along defined dimensions.
This includes comparing model architectures, dataset versions, evaluation methods, or
experimental configurations. The output is comparative tables, charts, and recommendations.
Examples include comparing fine-tuned models vs LLMs on a benchmark, analyzing which
evaluation dataset best measures capability, or comparing preprocessing strategies.

</details>

<details>
<summary><strong>Correction</strong> (<code>correction</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `correction` |
| **Description** | Correct aggregated artifacts from earlier completed tasks. |
| **Optional steps** | — none — |

Covers tasks whose main purpose is correcting mistakes in the effective aggregated view of
earlier completed-task outputs. Typical outputs are correction files, and sometimes
replacement assets created in the current task and referenced by those corrections. The
default optional step list is empty because many correction tasks are straightforward, but the
execute-task orchestrator may still add research or planning when the correction request is
complex.

</details>

<details>
<summary><strong>Data Analysis</strong> (<code>data-analysis</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `data-analysis` |
| **Description** | Analyze data, compute statistics, and generate visualizations. |
| **Optional steps** | `research-papers`, `research-code`, `planning`, `creative-thinking` |

Covers tasks whose primary purpose is analyzing existing data to extract insights, compute
metrics, generate charts, and summarize findings. The input is typically a dataset produced by
a prior task. The output includes statistical summaries, visualizations, and interpretive
results. Examples include sense frequency analysis, error distribution studies, and dataset
comparison reports.

</details>

<details>
<summary><strong>Deduplication</strong> (<code>deduplication</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `deduplication` |
| **Description** | Scan for and resolve duplicate assets across tasks. |
| **Optional steps** | `research-code`, `planning` |

Covers tasks that identify duplicate papers, suggestions, or other assets that accumulated
from parallel task execution. Uses the corrections mechanism to mark duplicates without
modifying completed task folders. The output is correction files that downstream aggregators
use to filter duplicates. This is typically run at project checkpoints.

</details>

<details>
<summary><strong>Download Dataset</strong> (<code>download-dataset</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `download-dataset` |
| **Description** | Download one or more datasets and register them as dataset assets. |
| **Optional steps** | `planning` |

Covers tasks that download datasets from external sources, verify their integrity, document
their structure and statistics, and register them as dataset assets. The primary output is one
or more dataset assets in the standard format. Examples include downloading evaluation
benchmarks, training corpora, or challenge test sets.

</details>

<details>
<summary><strong>Download Paper</strong> (<code>download-paper</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `download-paper` |
| **Description** | Download one or more research papers and register them as paper assets. |
| **Optional steps** | — none — |

Covers tasks that find, download, and summarize academic papers. Each paper becomes a paper
asset with metadata, PDF file, and detailed summary. This is typically a focused task
downloading a specific set of papers identified by prior research or brainstorming. The output
is paper assets following the paper asset specification.

</details>

<details>
<summary><strong>Experiment Run</strong> (<code>experiment-run</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `experiment-run` |
| **Description** | Run a designed experiment combining training, inference, and evaluation. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `creative-thinking`, `compare-literature` |

Covers tasks that execute a complete experimental pipeline: data preparation, model training
or inference, prediction generation, and evaluation against benchmarks. This type combines
multiple activities into a single coherent experiment. Examples include testing LLM zero-shot
performance across multiple models, running ablation studies, or comparing prompting
strategies.

</details>

<details>
<summary><strong>Feature Engineering</strong> (<code>feature-engineering</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `feature-engineering` |
| **Description** | Extract, compute, or generate features from data for model training. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `creative-thinking` |

Covers tasks that create new features from existing data. This includes extracting linguistic
features, computing statistical measures, generating LLM-based features, or deriving
meta-features from model outputs. The output is typically enriched data files or feature
extraction code that downstream model training tasks consume. Examples include computing word
frequency features, generating LLM embeddings, or extracting phonological distance measures.

</details>

<details>
<summary><strong>Infrastructure Setup</strong> (<code>infrastructure-setup</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `infrastructure-setup` |
| **Description** | Set up environments, tools, dependencies, or infrastructure. |
| **Optional steps** | `research-internet`, `research-code`, `planning` |

Covers tasks that install dependencies, configure environments, set up remote machines, or
create tooling needed by downstream tasks. The primary output is working infrastructure rather
than research results. Examples include installing a lexical resource or knowledge base,
configuring GPU environments, testing remote machine lifecycles, or setting up evaluation
frameworks.

</details>

<details>
<summary><strong>Internet Research</strong> (<code>internet-research</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `internet-research` |
| **Description** | Research a specific question using internet sources, documentation, and online resources. |
| **Optional steps** | `planning` |

Covers tasks whose primary purpose is answering a research question by searching the internet,
reading documentation, forums, blog posts, and other online resources. The output is typically
a structured research document, not code or datasets. This type is appropriate when the task
goal is knowledge gathering rather than asset production. Examples include investigating API
capabilities, comparing tool options, or understanding a technology landscape.

</details>

<details>
<summary><strong>Literature Survey</strong> (<code>literature-survey</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `literature-survey` |
| **Description** | Survey and summarize academic literature on a specific topic. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning` |

Covers tasks that systematically search for, download, and summarize research papers on a
topic. Unlike download-paper tasks which target specific known papers, literature surveys cast
a wide net to discover papers across a research area. The output is a collection of paper
assets plus a synthesis document. Examples include surveying recent approaches to a problem,
collecting papers on a new technique, or mapping a research landscape.

</details>

<details>
<summary><strong>TTS Benchmark Run</strong> (<code>tts-benchmark-run</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `tts-benchmark-run` |
| **Description** | Run a TTS inference benchmark measuring TTFB, speaker similarity, and RTF. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `creative-thinking`, `compare-literature` |

Covers tasks that benchmark a TTS system (ElevenLabs API or local Kokoro model) on the filler
corpus or val_96 set. Measures time-to-first-byte, GE2E speaker similarity against ElevenLabs
David reference, and real-time factor. Results must include per-clip metrics and aggregate
p50/p95/p99 for latency.

</details>

<details>
<summary><strong>TTS Fine-tuning Evaluation</strong>
(<code>tts-finetuning-eval</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `tts-finetuning-eval` |
| **Description** | Evaluate a Kokoro fine-tuned checkpoint on val_96 and filler corpus. |
| **Optional steps** | `research-papers`, `research-internet`, `research-code`, `planning`, `setup-machines`, `teardown`, `creative-thinking`, `compare-literature` |

Covers tasks that evaluate a specific Kokoro-82M fine-tuned checkpoint (StyleTTS2 Stage 2) on
the held-out val_96 set and the filler corpus. Measures speaker similarity, TTFB, and RTF.
Results are compared against the ElevenLabs David baseline and Kokoro base. The task does not
run training — it loads a checkpoint and runs inference only.

</details>

<details>
<summary><strong>Write Library</strong> (<code>write-library</code>)</summary>

| Field | Value |
|---|---|
| **ID** | `write-library` |
| **Description** | Create a reusable Python library for use by downstream tasks. |
| **Optional steps** | `research-internet`, `research-code`, `planning` |

Covers tasks that produce shared Python code registered as library assets. Libraries provide
data loaders, scorers, utilities, or other reusable functionality. Code lives in the task's
code/ directory and is referenced via module_paths in the library asset. Examples include
building a data loader, creating an evaluation scorer, or implementing shared preprocessing
utilities.

</details>
