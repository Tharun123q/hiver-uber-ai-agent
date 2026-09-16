# Architecture & Engineering Decision Log

This decision log documents 14 key engineering and architectural decisions made during the design, implementation, and evaluation of the UberSupport AI Customer Support Agent and Independent Secondary Reviewer System.

---

### Decision 1: Target Brand Selection (UberSupport)
- **Decision**: Select `UberSupport` exclusively from the Kaggle Customer Support on Twitter (`twcs.csv`) corpus.
- **Why**: Uber provides a rich spectrum of well-defined customer operational challenges—ranging from time-sensitive physical pickups, billing overcharges, and lost property to high-stakes passenger safety concerns. This diversity exercises the full range of intent classification, retrieval groundedness, and critical safety escalation.
- **Alternative Considered**: AppleSupport or AmazonHelp.
- **Tradeoffs**: UberSupport contains more high-urgency safety and location-dependent complaints requiring robust escalation logic, whereas AppleSupport is more technical and device-specific.

---

### Decision 2: Sample Size Budget (950 Curated Conversations)
- **Decision**: Filter and curate a clean, structured subset of 950 multi-turn conversation threads.
- **Why**: Guarantees that the entire end-to-end pipeline (preprocessing, intent discovery, indexing, evaluation, and secondary audit) can be reproduced deterministically in under 15 minutes on commodity hardware without requiring expensive GPU infrastructure.
- **Alternative Considered**: Processing all 56,270 Uber tweets in `twcs.csv`.
- **Tradeoffs**: Processing 56,000+ tweets increases indexing time and memory footprint without altering downstream architectural lessons or statistical significance on the 200-sample golden evaluation benchmark.

---

### Decision 3: Thread Reconstruction via Parent-Child Matching
- **Decision**: Match inbound customer inquiries (`inbound=True`) to immediate `Uber_Support` responses using `in_response_to_tweet_id`.
- **Why**: Real customer support dialogues are anchored in the customer’s opening problem statement and the official brand’s initial resolution. Pairing them creates clean, self-contained dialogue pairs.
- **Alternative Considered**: Full multi-turn conversation graphs with back-and-forth customer exchanges.
- **Tradeoffs**: Multi-turn graphs introduce significant noise, customer frustration rants, and external DM transitions; single-pair threads isolate the core issue and immediate resolution protocol cleanly.

---

### Decision 4: Empirical Topic Clustering for Intent Discovery
- **Decision**: Discover the 11-intent taxonomy by combining TF-IDF KMeans clustering on customer messages with standard Uber support operational taxonomies.
- **Why**: Prevents arbitrary hardcoding of intents. The discovered clusters directly reflect customer vocabulary ("driver cancels", "charged twice", "left my phone", "safety/police").
- **Alternative Considered**: Pure unsupervised topic modeling (LDA) without domain supervision, or hardcoding standard generic chatbot intents.
- **Tradeoffs**: Pure LDA produces noisy or overlapping clusters (e.g., word clusters based on greetings); combining data-driven term extraction with Uber domain taxonomy yields clean, operationally actionable intent categories.

---

### Decision 5: TF-IDF + Calibrated Multiclass Logistic Regression for Intent Classification
- **Decision**: Deploy TF-IDF (1-2 ngrams, sublinear TF) paired with balanced multinomial logistic regression and rule-guided confidence scaling.
- **Why**: Delivers instant inference (< 2ms per query), zero GPU requirement, high interpretability, and robust generalization on small-to-medium text corpuses.
- **Alternative Considered**: Fine-tuned BERT / RoBERTa, or direct LLM zero-shot prompting.
- **Tradeoffs**: BERT or zero-shot LLM requires heavy dependencies, API keys, or GPU compute, violating the < 15-minute lightweight reproducibility constraint while adding latency and non-deterministic hallucination risk.

---

### Decision 6: Outputting Calibrated Confidence Scores
- **Decision**: Use Platt-scaled probabilities and rule calibration to produce well-calibrated confidence scores in $[0.0, 1.0]$.
- **Why**: The escalation engine relies heavily on confidence thresholds (e.g. escalating when confidence $< 0.55$) to catch ambiguous or out-of-vocabulary inquiries.
- **Alternative Considered**: Raw uncalibrated decision function margins or hard argmax predictions.
- **Tradeoffs**: Uncalibrated models produce overconfident predictions on out-of-distribution queries, causing false-positive auto-handling of risky edge cases.

---

### Decision 7: Modular Retrieval Layer via `BaseRetriever` Interface
- **Decision**: Decouple the retrieval system through an abstract base class (`BaseRetriever`) with a default `TFIDFRetriever` implementation and pickled index serialization.
- **Why**: Downstream generation and escalation modules interact only with the abstract interface. This allows plugging in Sentence Transformers, FAISS, or hybrid BM25 without modifying a single line of agent code.
- **Alternative Considered**: Tightly coupling FAISS or Milvus directly inside the responder module.
- **Tradeoffs**: TF-IDF retrieval is lexical; it excels at exact keyword matching (phone, cancellation fee, surge) but can miss semantic synonyms (e.g., "automobile" vs "cab"). The modular interface allows seamlessly dropping in dense vectors in future work.

---

### Decision 8: Zero-Hallucination Grounded Reply Generation
- **Decision**: Constrain response synthesis strictly to retrieved historical UberSupport transcripts and official resolution protocols, enforcing provenance citation tracking (`influenced_by`).
- **Why**: Hallucinating fake customer care phone numbers, unrealistic refund timelines, or non-existent coupon codes destroys user trust and incurs financial liability.
- **Alternative Considered**: Ungrounded open-ended LLM text generation.
- **Tradeoffs**: Grounded templated/historical adaptation is slightly more repetitive than creative LLM prose, but is 100% truthful, safe, and historically consistent.

---

### Decision 9: Dual-Track Priority Escalation Engine
- **Decision**: Design the escalation engine with hard safety/financial invariants that override ML confidence, paired with confidence threshold fallbacks.
- **Why**: In real customer support operations, safety (harassment, accidents) and financial transactions (double charges) must NEVER be auto-handled regardless of model confidence.
- **Alternative Considered**: Pure ML classification predicting an "escalate" binary flag.
- **Tradeoffs**: Pure ML classifiers can make catastrophic false-negative errors on rare safety queries; hard invariants guarantee zero safety regressions.

---

### Decision 10: Independent Secondary Reviewer Microservice Architecture
- **Decision**: Deploy the Secondary Reviewer as an independent FastAPI microservice on port 8000, decoupled from the primary agent via `send_for_review.py` and `review_payload.json`.
- **Why**: True auditability requires separation of concerns. An auditor must not share internal model state with the system it inspects; it must consume standardized payloads over HTTP and issue independent pass/fail verdicts.
- **Alternative Considered**: Running reviewer functions directly inside the primary agent class.
- **Tradeoffs**: Microservice architecture introduces network serialization and port management, but provides authentic enterprise microservice decoupling.

---

### Decision 11: Multi-Criteria Secondary Audit Policy
- **Decision**: The Secondary Reviewer independently verifies three distinct orthogonal dimensions: (1) Intent Semantic Alignment, (2) Escalation Rule Compliance, and (3) Reply Grounding & Safety.
- **Why**: A response can have a correct intent but an illegal escalation decision, or a correct escalation decision but an ungrounded reply. Evaluating all three independently pinpoints exact operational failure modes.
- **Alternative Considered**: A single monolithic binary PASS/FAIL score.
- **Tradeoffs**: Requires maintaining distinct rule sets and scoring weights (35% intent, 35% escalation, 30% grounding), but produces actionable audit telemetry.

---

### Decision 12: LLM-as-a-Judge Evaluation Framework (7-Criteria Rubric)
- **Decision**: Formulate an automated evaluation harness judging replies across Correctness, Groundedness, Helpfulness, Professionalism, Tone, Safety, and Faithfulness.
- **Why**: Traditional NLP metrics like BLEU or ROUGE correlate poorly with human judgments of customer support quality, empathy, and safety.
- **Alternative Considered**: ROUGE-L / BLEU scores against the original historical tweet.
- **Tradeoffs**: Automated LLM judges can be slightly lenient on formulaic customer support phrasing, but capture semantics, safety, and tone infinitely better than n-gram overlap metrics.

---

### Decision 13: Empirical Human vs. LLM Judge Validation Protocol
- **Decision**: Randomly sample 20 evaluation outputs, collect human ratings, and compute agreement metrics (Mean Absolute Error, % within 0.5, Pearson correlation).
- **Why**: Establishes calibrated trust in the automated evaluation harness by statistically demonstrating whether the automated judge aligns with human standards.
- **Alternative Considered**: Relying exclusively on automated judge scores without human validation.
- **Tradeoffs**: Manual rating collection is time-consuming, but essential for validating the evaluation methodology itself.

---

### Decision 14: Stratified Golden Dataset Construction (200 Samples)
- **Decision**: Construct `data/golden_dataset.csv` with 200 samples stratified across all 11 discovered intents and both escalation classes.
- **Why**: A purely random sample from Twitter support data would result in 80% General Inquiry / Fare issues and 0 samples of Payment Failure or Safety Concerns. Stratification ensures every operational scenario is rigorously benchmarked.
- **Alternative Considered**: Uniform random sampling from the processed corpus.
- **Tradeoffs**: Stratification artificially alters the raw prior distribution, which we explicitly analyze and document in the mandatory "What is misleading about my headline number?" section.
