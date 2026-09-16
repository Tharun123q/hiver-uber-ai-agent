import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from src.config import load_config
from src.agent.primary_agent import PrimarySupportAgent
from reviewer.audit_rules import IndependentAuditEngine


def interactive_demo():
    print("=======================================================================")
    print("      UBERSUPPORT AI CUSTOMER SUPPORT AGENT - LIVE INTERACTIVE DEMO     ")
    print("=======================================================================")
    print("Type a customer query or choose from sample presets below.")
    print("Type 'exit' or 'quit' to exit.\n")
    print("Preset Examples to try:")
    print(" 1. 'My driver never arrived.'")
    print(" 2. 'I was charged twice for my trip.'")
    print(" 3. 'I left my iPhone in the back seat of the car.'")
    print(" 4. 'The driver was driving erratically and I felt in danger!'")
    print(" 5. 'My app keeps crashing when I enter my promo code.'")
    print("-----------------------------------------------------------------------\n")

    cfg = load_config()
    try:
        agent = PrimarySupportAgent.from_config(cfg)
    except Exception as e:
        print(f"Error loading models: {e}. Please run 'python run_pipeline.py' first.")
        return

    audit_engine = IndependentAuditEngine()

    presets = {
        "1": "My driver never arrived.",
        "2": "I was charged twice for my trip.",
        "3": "I left my iPhone in the back seat of the car.",
        "4": "The driver was driving erratically and I felt in danger!",
        "5": "My app keeps crashing when I enter my promo code."
    }

    while True:
        try:
            user_input = input("\nEnter customer message (or preset 1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break

        if not user_input or user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting demo.")
            break

        if user_input in presets:
            user_input = presets[user_input]
            print(f"Selected: \"{user_input}\"")

        print("\nProcessing...")
        result = agent.handle_message(user_input)
        audit_res = audit_engine.audit_record(result)

        print("\n========================= PRIMARY AGENT OUTPUT =========================")
        print(f"Customer Message : \"{result['customer_message']}\"")
        print(f"Predicted Intent : {result['predicted_intent']} (Confidence: {result['intent_confidence']:.2f})")
        print(f"Escalation Policy: {result['escalation']['decision']}")
        print(f"Escalation Reason: {result['escalation']['reason']} (Confidence: {result['escalation']['confidence']:.2f})")
        print("\nGenerated Grounded Reply:")
        print(f"\"{result['grounded_reply']}\"")
        print(f"\nGrounding Rationale: {result['grounding_rationale']}")
        print("Influenced By Historical Conversations:")
        for inf in result["influenced_by"]:
            print(f"  - [{inf['conversation_id']}] (Sim: {inf['similarity_score']:.2f}): {inf['historical_snippet'][:65]}...")

        print("\n====================== SECONDARY REVIEWER AUDIT ======================")
        print(f"Overall Audit Verdict: {audit_res['item_verdict']} (Score: {audit_res['item_score']}/10.0)")
        print(f"  * Intent Review    : {audit_res['intent_review']['status']} - {audit_res['intent_review']['reason']}")
        print(f"  * Escalation Review: {audit_res['escalation_review']['status']} - {audit_res['escalation_review']['reason']}")
        print(f"  * Grounding Review : {audit_res['grounding_review']['status']} - {audit_res['grounding_review']['reason']}")
        print("=======================================================================")


if __name__ == "__main__":
    interactive_demo()
