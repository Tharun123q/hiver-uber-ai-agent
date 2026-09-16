import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from reviewer.audit_rules import IndependentAuditEngine

app = FastAPI(
    title="UberSupport Secondary Reviewer Audit Service",
    description="Independent LLM Audit Microservice verifying Primary AI Agent outputs on intent, escalation, and grounding.",
    version="1.0.0"
)

audit_engine = IndependentAuditEngine()


class AuditPayload(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of primary agent evaluation records")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "UberSupport Secondary Reviewer Audit Engine",
        "version": "1.0.0"
    }


@app.post("/audit")
def audit_batch(payload: Union[AuditPayload, List[Dict[str, Any]]]):
    """
    Audits a batch of primary agent decisions and produces the final audit report.
    """
    records = payload.records if isinstance(payload, AuditPayload) else payload
    if not records:
        raise HTTPException(status_code=400, detail="Empty review payload provided.")

    item_audits = []
    total_score = 0.0
    discrepancies = 0

    intent_fails = 0
    escalation_fails = 0
    grounding_fails = 0

    for rec in records:
        audit_res = audit_engine.audit_record(rec)
        item_audits.append(audit_res)
        total_score += audit_res["item_score"]

        if audit_res["item_verdict"] == "FAIL":
            discrepancies += 1
        
        if audit_res["intent_review"]["status"] == "FAIL":
            intent_fails += 1
        if audit_res["escalation_review"]["status"] == "FAIL":
            escalation_fails += 1
        if audit_res["grounding_review"]["status"] == "FAIL":
            grounding_fails += 1

    n = len(records)
    avg_score = round(total_score / n, 1)

    intent_review = "FAIL" if intent_fails > (n * 0.1) else ("WARNING" if intent_fails > 0 else "PASS")
    escalation_review = "FAIL" if escalation_fails > (n * 0.05) else ("WARNING" if escalation_fails > 0 else "PASS")
    grounding_review = "FAIL" if grounding_fails > (n * 0.05) else ("WARNING" if grounding_fails > 0 else "PASS")

    audit_status = "APPROVED" if (avg_score >= 8.5 and discrepancies <= (n * 0.05)) else "FLAGGED"
    reviewer_notes = (
        "Primary agent output is consistent with historical evidence."
        if audit_status == "APPROVED"
        else f"Auditor identified {discrepancies} critical discrepancies requiring policy calibration."
    )

    report = {
        "audit_status": audit_status,
        "intent_review": intent_review,
        "grounding_review": grounding_review,
        "escalation_review": escalation_review,
        "overall_score": avg_score,
        "discrepancies": discrepancies,
        "total_reviewed": n,
        "reviewer_notes": reviewer_notes,
        "discrepancy_breakdown": {
            "intent_failures": intent_fails,
            "escalation_failures": escalation_fails,
            "grounding_failures": grounding_fails
        },
        "sample_audits": item_audits[:5]
    }

    return report


if __name__ == "__main__":
    uvicorn.run("reviewer.reviewer_service:app", host="127.0.0.1", port=8000, log_level="info")
