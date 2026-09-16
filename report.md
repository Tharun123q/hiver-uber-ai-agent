# UberSupport AI Support and Independent Audit Report

## Executive Summary

This report summarizes a deterministic, offline-capable support-agent run. The primary agent classifies, retrieves historical evidence, applies escalation policy, and produces a grounded response. A separate audit stage checks the result.

## Measured Results

- Evaluation records: `200`
- Intent accuracy: `0.525`
- Intent macro F1: `0.3472`
- Escalation F1: `0.8227`
- Grounded reply judge average: `4.8/5`

## Architecture

Input conversations are cleaned and paired, then used to build the intent taxonomy, classifier, and TF-IDF retrieval index. The primary agent combines those components with the escalation policy and grounded responder. The reviewer independently audits intent, escalation, and evidence grounding.

## Reproducibility

Run `python run_pipeline.py` from the repository root. Configuration is centralized in `config/config.yaml`; generated outputs are written to the configured artifact paths.

## Limitations

The offline judge and human comparison are deterministic proxies, not substitutes for live human annotation or production LLM evaluation. Results should therefore be interpreted with the failure analysis and misleading-number analysis.
