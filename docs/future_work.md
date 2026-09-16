# Future Work: One More Week

1. Replace TF-IDF retrieval with a hybrid lexical and dense retriever, then compare recall and grounding quality.
2. Add FAISS or another persistent vector index for larger conversation collections.
3. Introduce a reviewed knowledge base for current policies, refunds, safety procedures, and account workflows.
4. Track conversational state across multiple customer messages instead of classifying each message independently.
5. Add active learning so low-confidence or reviewer-disputed examples are prioritized for annotation.
6. Replace simulated human comparison with a blinded annotation workflow and measure inter-rater agreement.
7. Add live provider adapters behind a common interface while retaining the deterministic offline fallback.
8. Add monitoring for drift, retrieval coverage, escalation rates, latency, and reviewer disagreement.
