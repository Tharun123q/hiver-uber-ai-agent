# UberSupport AI Customer Support and Independent Audit

An end-to-end customer-support research prototype built around the Twitter Customer Support dataset. The system classifies incoming customer messages, retrieves similar historical conversations, decides whether a case should be handled automatically or escalated, generates a grounded response, and sends the result through an independent audit stage.

The project is designed to be reproducible and explainable. It runs locally without an external LLM API key, preserves evidence behind each response, and produces evaluation artifacts that make failure cases visible.

## What the project does

The pipeline follows this workflow:

1. Extract and clean customer/support conversation pairs from the raw dataset.
2. Discover and document the support intent taxonomy.
3. Build a balanced golden dataset for evaluation.
4. Train a TF-IDF intent classifier and create a TF-IDF retrieval index.
5. Run the primary support agent over the golden dataset.
6. Audit intent consistency, escalation decisions, and response grounding.
7. Generate classification, escalation, baseline, judge, and failure-analysis reports.

The primary classifier supports these 11 intents:

- Pickup Problem
- Fare / Overcharge Issue
- Refund Request
- Driver Complaint
- Lost Item
- Account Access / Login
- Payment Failure
- App Technical Issue
- Safety Concern
- Rider Complaint
- General Inquiry

## Requirements

- Windows, macOS, or Linux
- Python 3.11 recommended
- The raw Twitter Customer Support CSV dataset

Place the dataset at the repository root as `twcs.csv`, or place it at `data/twcs.csv` or `data/uber.csv`. The preprocessing code looks for these locations when the configured path is unavailable.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The repository already includes a `.venv` in the development environment, but creating a fresh environment is recommended for a new machine.

## Run the complete pipeline

```powershell
python run_pipeline.py
```

This command runs preprocessing, taxonomy generation, golden-dataset creation, model training, retrieval indexing, primary evaluation, reviewer auditing, and report generation. The run is deterministic under the seed configured in `config/config.yaml`.

The reviewer service is started automatically when the audit stage needs it. If the HTTP service cannot be reached, the pipeline falls back to the in-process audit engine.

## Try the interactive demo

After the pipeline has completed, start the live command-line demo:

```powershell
python run_demo.py
```

The demo accepts a customer message or one of five preset examples. It prints the predicted intent, confidence, escalation decision, grounded reply, historical evidence, and secondary audit result.

You can also run one message directly:

```powershell
python src\agent\primary_agent.py --message "I left my phone in the Uber"
```

The classifier model and retrieval index must exist before using either demo command. Run `python run_pipeline.py` first if they are missing.

## Run the tests

```powershell
python -m pytest tests -q
```

The tests cover preprocessing, classifier behavior, retrieval and serialization, escalation rules, grounded responses, configuration loading, and the reviewer API.

## Repository structure

```text
config/       Central configuration and logging setup
data/         Processed, labeled, and golden datasets
docs/         Taxonomy and engineering decision documentation
evaluation/   Metrics, deterministic judge, and human-comparison harness
reviewer/     FastAPI audit service and audit client
src/          Data, NLP, model, retrieval, policy, and agent modules
tests/        Automated component tests
artifacts/    Generated models, indexes, predictions, and evaluation results
reports/      Generated reports and failure analysis
```

The main configuration file is [config/config.yaml](config/config.yaml). It controls dataset paths, output paths, retrieval settings, classifier settings, escalation thresholds, and reviewer endpoints.

## Key outputs

After a successful pipeline run, the most useful files are:

- [artifacts/primary_eval.json](artifacts/primary_eval.json): primary-agent predictions, replies, confidence, and provenance
- [artifacts/final_audit_report.json](artifacts/final_audit_report.json): item-level and batch audit results
- [artifacts/metrics.json](artifacts/metrics.json): classification, escalation, baseline, and judge metrics
- [artifacts/classification_report.json](artifacts/classification_report.json): precision, recall, and F1 by intent
- [artifacts/confusion_matrix.png](artifacts/confusion_matrix.png): intent confusion matrix
- [reports/failure_analysis.md](reports/failure_analysis.md): top observed failure cases and mitigations
- [reports/report.md](reports/report.md): generated project run summary
- [reports/misleading_numbers.md](reports/misleading_numbers.md): interpretation of headline metrics

## Design notes and limitations

The response generator is deliberately grounded in retrieved historical replies and records the conversations that influenced each response. The escalation engine gives priority to safety signals, billing disputes, and low-confidence predictions.

The reviewer and judge are deterministic offline components. They provide repeatable checks, but they are not equivalent to a production LLM review or independent human annotation. Evaluation results should be read together with the failure analysis, class-level metrics, and the limitations described in [reports/misleading_numbers.md](reports/misleading_numbers.md).

Further architectural decisions are documented in [docs/decision_log.md](docs/decision_log.md), and possible next steps are listed in [docs/future_work.md](docs/future_work.md).

## License

This project is intended as an academic and engineering prototype. Add the appropriate license before public distribution.
